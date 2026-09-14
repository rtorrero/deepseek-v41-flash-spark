"""ExFold at chunk scale: the calibrated tables, and the eight tensor ops that apply them.

`engine/prefill_topk.py` holds the environment and the reference arithmetic, and says what ExFold
is and why (arXiv 2608.24938). This module is the tensor half: it loads the tables
`tools/exfold_prepare.py` wrote and turns one prefill chunk's `[T, 6]` routing into `[T, K']`.

The tables, per MoE layer `L` of the checkpoint:

  S[L, s, t]  float32  the directed scalar projector `s*_{s->t}` (Eq. 8)
  N[L, s, t]  float32  the normalised reconstruction loss `l_{s->t}` (Eq. 9/16); 1e30 = unobserved
  h[L, e]     float32  the cached output-norm estimate `h_e` used by the Eq. 20 ranking

At 40 layers and 384 routed experts that is 2 x 40 x 384 x 384 x 4 B = **47.2 MB** on the device,
plus 61 KB for `h`. It is allocated once at start, outside the arena, and never written to.

What this costs per layer per chunk: one multiply, one `topk` over 6 columns, four gathers, one
`argmin` over K' columns, one `scatter_add_`. No host sync, no kernel of its own, no graph -- and
prefill has no captured graphs to invalidate in the first place. Decode does not import this file.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from engine.prefill_topk import UNOBSERVED_LOSS


class FoldTable:
    """The calibrated ExFold state for one checkpoint, resident on the device.

    Built by `load()`; `reduce()` is the only thing the engine calls.
    """

    def __init__(self, scalar: torch.Tensor, loss: torch.Tensor, norm: torch.Tensor,
                 meta: dict, path: str):
        self.S = scalar          # [n_layers, E, E] float32
        self.N = loss            # [n_layers, E, E] float32
        self.h = norm            # [n_layers, E]    float32
        self.meta = meta
        self.path = path

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str, n_layers: int, n_experts: int, device="cuda",
             keep_sig: str | None = None, log=print) -> "FoldTable":
        """Read the `.npz` and put it on the device, refusing anything that does not fit.

        Shape is checked and refused, not warned about: a table calibrated on a different
        checkpoint would index the wrong experts and the only symptom would be worse output.
        The keep-set signature is warned about instead, because a table calibrated under a
        different keep-set is still *sound* -- the pairs it never observed carry the 1e30
        sentinel, their confidence is 0, and their weight takes the proportional fallback. It is
        just less useful, and the coverage line below says how much less.
        """
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"DSV41_PREFILL_FOLD=exfold needs the calibrated tables and there are none at "
                f"{path!r}. Run `python3 tools/exfold_prepare.py` once with the engine stopped, "
                f"or set DSV41_PREFILL_FOLD=none to measure the plain top-k' arm instead.")
        with np.load(path, allow_pickle=False) as z:
            missing = {"scalar", "loss", "norm", "meta"} - set(z.files)
            if missing:
                raise ValueError(f"{path}: not an ExFold table (missing {sorted(missing)})")
            scalar, loss, norm = z["scalar"], z["loss"], z["norm"]
            meta = json.loads(str(z["meta"].item() if z["meta"].shape == () else z["meta"][0]))
        want = (n_layers, n_experts, n_experts)
        for name, arr, shape in (("scalar", scalar, want), ("loss", loss, want),
                                 ("norm", norm, (n_layers, n_experts))):
            if tuple(arr.shape) != shape:
                raise ValueError(f"{path}: {name} is {tuple(arr.shape)}, this checkpoint needs "
                                 f"{shape} -- the table was calibrated on a different model")

        observed = int((loss < UNOBSERVED_LOSS).sum())
        total = int(loss.size) - n_layers * n_experts        # the diagonal is free
        log(f"ExFold tables from {path}: {observed / max(total, 1):.1%} of directed pairs "
            f"observed ({meta.get('observer_tokens', '?')} observer tokens, "
            f"{meta.get('sequences', '?')} sequences, lambda={meta.get('ridge', '?')})")
        if keep_sig and meta.get("keep_sig") and meta["keep_sig"] != keep_sig:
            log("WARNING: these tables were calibrated under a different keep-set than the one "
                "this engine is serving. Pairs the calibration never saw fall back to the "
                "retained routes in proportion to their weights, so this is safe but weaker; "
                "recalibrate to get the coverage back.")
        return cls(torch.from_numpy(np.ascontiguousarray(scalar)).to(device),
                   torch.from_numpy(np.ascontiguousarray(loss)).to(device),
                   torch.from_numpy(np.ascontiguousarray(norm)).to(device),
                   meta, path)

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.S, self.N, self.h))

    # ------------------------------------------------------------------ the fold
    def reduce(self, indices: torch.Tensor, weights: torch.Tensor, L: int, k_prime: int):
        """`[T, k]` routing -> `[T, k']`, with the excluded routes folded onto the retained ones.

        `weights` is what `engine/model.py::moe` already computed: the router's scores gathered at
        `indices`, normalised over the checkpoint's k and multiplied by `route_scale`. That is the
        paper's `alpha_e(x)`, so the fold goes on top of it and nothing about the checkpoint's own
        routing is re-derived here.

        Mirrors `engine.prefill_topk.fold_reference` exactly; `tools/test_exfold.py` holds them to
        each other on a toy.
        """
        T, k = indices.shape
        if k_prime >= k:
            return indices, weights
        S, N, h = self.S[L], self.N[L], self.h[L]

        # (10) rank the router's k routes by weight * output-norm estimate, not by weight alone.
        order = torch.topk(weights * h[indices], k, dim=-1, sorted=True)[1]       # [T, k]
        # Put both halves back in the router's own column order. It changes no arithmetic -- the
        # MoE kernel is a sum over columns -- but it makes `kept_idx` a deterministic function of
        # the routing rather than of `topk`'s internal ordering, so two runs diff cleanly and the
        # reference implementation can be compared to this one column by column.
        keep_cols = order[:, :k_prime].sort(dim=-1)[0]                            # [T, k']
        drop_cols = order[:, k_prime:].sort(dim=-1)[0]                            # [T, m]
        kept_idx = indices.gather(1, keep_cols).contiguous()                      # [T, k']
        kept_w0 = weights.gather(1, keep_cols)                                    # [T, k'] original
        src = indices.gather(1, drop_cols)                                        # [T, m]
        w_src = weights.gather(1, drop_cols)                                      # [T, m]

        # (12) each omitted source picks the retained target with the smallest calibrated loss.
        # [T, m, k'] -- at T=2048, m<=3, k'<=5 that is at most 31k lookups in a [384, 384] table.
        pair_loss = N[src.unsqueeze(-1), kept_idx.unsqueeze(1)]
        best_loss, best = pair_loss.min(dim=-1)                                   # [T, m] each
        tgt = kept_idx.gather(1, best)                                            # [T, m]

        # (26) confidence: an unobserved or badly reconstructed pair sends nothing.
        conf = (1.0 - best_loss).clamp_(0.0, 1.0)                                 # [T, m]

        # (13) the folded part, coalesced onto the target so each retained expert runs once.
        kept_w = kept_w0.clone()
        kept_w.scatter_add_(1, best, w_src * conf * S[src, tgt])

        # (26) and the part that stays behind, spread over the retained routes by original weight.
        rest = (w_src * (1.0 - conf)).sum(dim=-1, keepdim=True)                   # [T, 1]
        kept_w = kept_w + rest * (kept_w0 / (kept_w0.sum(dim=-1, keepdim=True) + 1e-20))

        return kept_idx, kept_w.contiguous()


def reduce_plain(indices: torch.Tensor, weights: torch.Tensor, k_prime: int, route_scale: float):
    """The `none` arm: the router's own first k', renormalised over them.

    `indices` came out of `logits.topk(k)` in descending order, so its first k' columns ARE
    `logits.topk(k')`; renormalising them reproduces the shipped router at a smaller k exactly.
    """
    k = indices.shape[1]
    if k_prime >= k:
        return indices, weights
    idx = indices[:, :k_prime].contiguous()
    w = weights[:, :k_prime]
    return idx, (w / (w.sum(dim=-1, keepdim=True) + 1e-20) * route_scale).contiguous()
