"""Fewer routed experts in PREFILL: `DSV41_PREFILL_TOPK` and `DSV41_PREFILL_FOLD`.

The measurement that made this worth building. `tools/prefill_bound_test.py` ran one 2,048-token
chunk through one MoE layer at k = 6, 4 and 3 and timed `moe_fn` alone (the CB3 -> FP4 unpack plus
the two FP4 grouped launches): **54.1 ms at k=6, 38.3 ms at k=4, 30.9 ms at k=3**. The pair count
at k=3 is 0.50x of k=6 and the time is 0.57x, against a page of reasoning (docs/gemm-dispatch.md)
that predicted ~1.0x because the per-expert unpack does not depend on k. It does depend on k, the
prefill MoE is compute-bound, and the FP4 MoE kernels are ~893 ms of a ~3.7 s chunk. Cutting k
therefore buys real time -- and the only remaining question is what it costs in quality.

Dropping the excluded experts is the cheap answer and a measurably bad one. ExFold (arXiv
2608.24938, "ExFold: Unified Expert Folding for Training-Free MoE Prefill-Decode Acceleration") is
the better one, and its Table 6 is on this model family -- DeepSeek-V4-Flash, 256 routed experts,
Top-6 routing, the same router shape this checkpoint has. At three routed experts per prefill
token, direct Top-3 reduction scores 70.54 against the Top-6 baseline's 72.68 (97.05 % of it);
ExFold-P3 scores 71.53 (98.42 %). The headline "about 99 % of the original average quality" is the
whole paper across both phases; the prefill-only number on this architecture is the 98.42 % above.

What ExFold actually does, in one paragraph. The router still picks its six. Of those six, `K'`
are kept -- ranked not by router weight alone but by `w_e * h_e`, the weight times a calibrated
estimate of that expert's output norm, because the paper's Figure 3 shows expert outputs within a
layer are nearly co-directional but differ in magnitude by over 3x, so the weight alone is a bad
estimate of how much a route contributes (their Table 5: ranking by `w * h` beats ranking by
either alone). Each excluded expert `s` is then *folded* onto the kept expert `t` that reconstructs
it best, by adding `w_s * S[s,t]` to `t`'s weight -- `S[s,t]` being a single scalar, calibrated
offline, that answers "by how much must t's output be rescaled to stand in for s's?". Nothing is
renormalised afterwards: the scalar IS the magnitude correction, and renormalising would throw it
away. No expert weights change, the router does not change, and the fused MoE kernel still runs
exactly `K'` routes per token. The tables come from `tools/exfold_prepare.py`; the arithmetic is
`fold_reference` below and `engine/exfold.py`.

This module is torch-free on purpose -- it parses the environment and holds the reference
implementation of the routing arithmetic, so `tools/test_exfold.py` exercises the shipped object
on a CPU runner rather than a copy of it. Everything that needs a tensor lives in
`engine/exfold.py`.

Variables:

  DSV41_PREFILL_TOPK   an integer 1..k, or unset / "off" / "0" for the shipped k. Prefill only.
  DSV41_PREFILL_FOLD   `exfold` (the default whenever the top-k is reduced) or `none`.
  DSV41_EXFOLD_TABLE   path to the calibrated tables; defaults to `<model_dir>/exfold_table.npz`.

Off is the default and off is byte-identical: with `DSV41_PREFILL_TOPK` unset `OVERRIDE` is None,
`prefill_topk` returns the k it was handed, and `engine/model.py::moe` issues the same topk and the
same weights it always did. `tools/test_prefill_topk.py` pins both halves of that.

`DSV41_PREFILL_FOLD=none` is the plain top-k' reduction -- the paper's "Prefill TopK=k" baseline,
and bit-for-bit the behaviour of the measurement hook this module replaces. It is kept because it
is the control the ExFold arm has to be compared against, not because anything should serve on it.

Why this is not `DSV41_TOPK`. That switch (tools/v41_ref.py::routed_topk) rewrites
`Args.n_activated_experts` as the config is read, so it changes the checkpoint's routing
everywhere: the decode path's static buffers, the CUDA graphs, the prune fallback table and the
served output all follow it. This one changes nothing but a PREFILL call's routing, and decode is
untouched by construction -- `engine/fastdecode.py` never imports this module.
"""

from __future__ import annotations

import os

ENV_K = "DSV41_PREFILL_TOPK"
ENV_FOLD = "DSV41_PREFILL_FOLD"
ENV_TABLE = "DSV41_EXFOLD_TABLE"

#: The default filename `tools/exfold_prepare.py` writes and the engine looks for, next to the
#: checkpoint, so a box that has calibrated once does not need the variable set at all.
TABLE_FILENAME = "exfold_table.npz"

FOLD_MODES = ("none", "exfold")

#: The paper's sentinel for a directed pair its calibration never observed (Appendix B: "unobserved
#: pairs receive loss 1e30 so that an observed target is preferred whenever one exists"). Any loss
#: at or above 1 drives the confidence below to exactly 0, so an unobserved source does not get
#: folded anywhere -- its weight falls back to the retained routes instead.
UNOBSERVED_LOSS = 1e30


def parse_k(raw, default_k: int = 6):
    """The env value -> a prefill top-k override, or None for "leave the routing alone"."""
    v = (raw or "").strip().lower()
    if v in ("", "off", "0", "none", "default"):
        return None
    try:
        k = int(v)
    except ValueError:
        raise ValueError(f"{ENV_K}: {v!r}; use an integer 1..{default_k}, "
                         f"or leave it unset") from None
    if not 1 <= k <= int(default_k):
        raise ValueError(f"{ENV_K}: {k}; must be 1..{default_k} "
                         f"(the checkpoint's n_activated_experts)")
    return k


def parse_fold(raw, k_override=None) -> str:
    """The env value -> `none` or `exfold`.

    The default is `exfold` whenever the top-k is actually reduced, because dropping a routed
    expert outright is the arm the paper measures as the worse one and nobody should reach it by
    forgetting a variable. With no reduction in force the mode is inert and reported as `none`.
    """
    v = (raw or "").strip().lower()
    if v == "":
        return "exfold" if k_override is not None else "none"
    if v in ("off", "drop", "0"):
        v = "none"
    if v not in FOLD_MODES:
        raise ValueError(f"{ENV_FOLD}: {v!r}; use one of {', '.join(FOLD_MODES)}")
    return v


def table_path(raw=None, model_dir: str | None = None) -> str | None:
    """Where the calibrated tables are. Explicit path wins; otherwise next to the checkpoint."""
    v = (raw or "").strip()
    if v:
        return os.path.expanduser(v)
    if model_dir:
        return os.path.join(model_dir, TABLE_FILENAME)
    return None


#: Read once at import. A harness that wants to sweep k inside ONE process (which is the only way
#: to A/B anything on this box -- see NOTES, "any A/B that changes a resident allocation's size has
#: to be run inside one process") assigns these directly between runs. Nothing in the engine writes
#: them.
OVERRIDE = parse_k(os.environ.get(ENV_K), 6)
FOLD = parse_fold(os.environ.get(ENV_FOLD), OVERRIDE)


def prefill_topk(k: int, drop: bool = False) -> int:
    """The number of routed experts a PREFILL call should activate.

    Returns `k` unchanged whenever the override is off, which is every shipped configuration.
    """
    if OVERRIDE is None:
        return k
    if drop:
        raise ValueError(f"{ENV_K} cannot be combined with DSV41_PRUNE_MODE=drop: the "
                         f"per-column fallback table is sized for the checkpoint's k")
    return min(OVERRIDE, k)


# --------------------------------------------------------------------------- the arithmetic
# The reference implementation of one token's routing under ExFold. `engine/exfold.py` does the
# same thing to a whole chunk with tensors; `tools/test_exfold.py` checks the two agree and this
# one against the equations, on a toy, with no torch in the room.

def fold_reference(weights, experts, k_prime, h, scalar, loss, eps: float = 1e-12):
    """One token: the router's k routes -> the K' that will actually run, with folded weights.

    Arguments, all plain Python sequences:

      weights  the router weights of this token's k routes, AFTER the checkpoint's own
               normalisation and `route_scale` -- i.e. exactly the `alpha_e(x)` of the paper's
               Eq. 3/4, the numbers `engine/model.py::moe` hands the MoE kernel today.
      experts  the expert ids of those k routes, same order.
      k_prime  how many to keep.
      h        h[e] -- the calibrated output-norm estimate of expert e (Eq. 20's `h_e`).
      scalar   scalar[s][t] -- the directed projector `s*_{s->t}` of Eq. 8.
      loss     loss[s][t]   -- the normalised reconstruction loss `l_{s->t}` of Eq. 9/16.

    Returns `(kept_experts, kept_weights)`, both length `k_prime`, in the order the retained
    routes had in the input -- the MoE kernel does not care about the order, but a stable one
    makes a diff of two runs readable.

    The three steps, with the paper's equation numbers:

      (10)  B = TopK_{e in S_K(x)} ( w_e * h_e , K' )        -- rank by weight TIMES output norm
      (12)  pi(s) = argmin_{t in B} l_{s->t}                 -- each omitted route picks its target
      (13)  w~_t = w_t + sum_{s: pi(s)=t} w_s * S[s,t]       -- and is folded onto it

    plus the confidence-aware safeguard the paper adds *specifically for this model family*
    (Eq. 26; DeepSeek-V4-Flash "exhibits a wider range of router scores and expert-output
    magnitudes than the primary Qwen model"):

      c_s = clip(1 - min_t l_{s->t}, 0, 1)

    Only `w_s * c_s * S[s,t]` is folded onto the minimum-loss target; the remainder
    `w_s * (1 - c_s)` falls back to the retained routes in proportion to their ORIGINAL router
    weights. A pair the calibration never saw has loss 1e30, hence c_s = 0, hence all of its
    weight takes the proportional fallback and nothing is folded onto a target that was never
    shown to stand in for it. This is what makes a stale or partial table degrade into the `none`
    arm rather than into noise.

    Nothing is renormalised at the end. `S[s,t]` is the magnitude correction; dividing the folded
    weights by their new sum would undo exactly the thing that was calibrated.
    """
    k = len(weights)
    if k_prime >= k:
        return list(experts), list(weights)
    if k_prime < 1:
        raise ValueError("k_prime must be at least 1")

    # (10) rank by w_e * h_e, keep the best K'. Ties break on the router's own order, which is
    # descending logit -- so with a flat h this is the plain top-k' and the `none` arm.
    order = sorted(range(k), key=lambda j: (-weights[j] * h[experts[j]], j))
    keep_cols = sorted(order[:k_prime])
    drop_cols = sorted(order[k_prime:])

    kept_experts = [experts[j] for j in keep_cols]
    kept_w0 = [weights[j] for j in keep_cols]          # the ORIGINAL weights, for the fallback
    kept_w = list(kept_w0)
    mass0 = sum(kept_w0)

    for j in drop_cols:
        s = experts[j]
        w_s = weights[j]
        # (12) the retained target that reconstructs this omitted expert best
        best = min(range(k_prime), key=lambda c: (loss[s][kept_experts[c]], c))
        l = loss[s][kept_experts[best]]
        # (26) how much of it we are willing to send there
        c = 1.0 - l
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
        # (13) the folded part
        kept_w[best] += w_s * c * scalar[s][kept_experts[best]]
        # (26) and the part that does not go there, spread over the retained routes by weight
        rest = w_s * (1.0 - c)
        if rest and mass0 > eps:
            for c2 in range(k_prime):
                kept_w[c2] += rest * (kept_w0[c2] / mass0)

    return kept_experts, kept_w


def reduce_reference(weights, experts, k_prime, route_scale: float, eps: float = 1e-20):
    """The `none` arm, for the same toy: keep the router's first k' and renormalise over them.

    `weights` arrive already scaled by `route_scale`, so the renormalisation has to put it back;
    the result is identical to running `logits.topk(k')` in the first place, which is what makes
    this mode the exact control the fold arm is measured against.
    """
    k = len(weights)
    if k_prime >= k:
        return list(experts), list(weights)
    kept_experts = list(experts[:k_prime])
    kept_w = list(weights[:k_prime])
    total = sum(kept_w)
    return kept_experts, [w / (total + eps) * route_scale for w in kept_w]
