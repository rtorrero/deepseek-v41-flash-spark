#!/usr/bin/env python3
"""exfold_prepare.py -- calibrate the ExFold fold tables for this checkpoint. Run once.

    python3 tools/exfold_prepare.py                      # writes <model_dir>/exfold_table.npz
    python3 tools/exfold_prepare.py --sequences 32       # a faster, coarser table
    python3 tools/exfold_prepare.py --out /tmp/t.npz --sequences 8 --observer-tokens 16  # a smoke run

**Run it with the engine stopped.** It builds its own engine and lays down the same ~89 GB arena
the server does; two of those at once wedge this box hard enough to need a power cycle. `./stop.sh`
first, and hold the box for the duration.

---------------------------------------------------------------------------------------------
What is being calibrated

ExFold (arXiv 2608.24938) replaces "drop the routed experts we cannot afford" with "fold them onto
the ones we keep". Folding expert `s` onto expert `t` means adding `w_s * S[s,t]` to `t`'s router
weight, where `S[s,t]` is one scalar per ordered pair per layer. The paper's justification is its
Figure 3: within a layer, expert outputs are nearly co-directional -- normalising them lifts the
mean pairwise cosine from 0.335 to 0.529 at one layer and 0.251 to 0.471 at another -- while their
magnitudes differ by more than 3x. So the mismatch between an excluded expert and a retained one is
"almost purely radial", and a single scalar per direction is enough to correct it.

That scalar is a weighted least-squares fit. For every ordered pair `(s, t)` that the router
co-routes on the same token, collect `u_i = E_s(x_i)` and `v_i = E_t(x_i)` over the calibration
tokens, weight each by `w_i = ||u_i||` (Appendix B, "source-output-norm weighting"), and solve
`min_s sum_i w_i ||u_i - s v_i||^2` with a ridge term:

    (8)   S[s,t] = sum_i w_i <v_i, u_i> / ( sum_i w_i ||v_i||^2 + lambda )

Store, alongside it, how well that scalar actually worked -- the normalised residual, which is what
picks the fold target at inference time and what the confidence safeguard reads:

    (9)   L[s,t] = sum_i w_i ||u_i - S[s,t] v_i||^2 / ( sum_i w_i ||u_i||^2 + eps )

and one more number per expert, the output-norm estimate `h_e` that Eq. 20's ranking needs:

          h[e]   = sqrt( mean_i ||E_e(x_i)||^2 )

Three accumulators do all of it. Writing `n_p = ||o_p||` for the norm of this token's p-th routed
expert output and `G[p,q] = <o_p, o_q>` for their Gram matrix:

    A[s,t] += n_p * G[q,p]        (= sum_i w_i <v_i, u_i>)
    B[s,t] += n_p * G[q,q]        (= sum_i w_i ||v_i||^2)
    C[s,t] += n_p * n_p^2         (= sum_i w_i ||u_i||^2)

so one 6x6 Gram per token per layer is the entire measurement, and the pass costs no more than the
expert outputs themselves.

---------------------------------------------------------------------------------------------
How it runs, and what it costs

Stage 1 -- observe. Prefill each calibration sequence and tap, at every MoE layer the prefill
actually runs, the MoE input rows and the router's choice at a sample of token positions (the paper
collects "at most 64 observer tokens per sequence"; so does this, evenly spaced over the second
half of the sequence, where the context is no longer cold). Held on the host: at the defaults,
40 layers x 4,096 rows x 5,120 dims in bf16 = 1.7 GB.

  * With `DSV41_SWA_REPLAY=1` (the default) a prompt only runs layers 0..`candidate_source_layer`
    -- layer 20's global KV feeds 20..39, so prefill stops there. Those are exactly the layers the
    prefill fold will ever apply to. The table is still indexed by absolute layer; the rest stay
    unobserved and are never read.
  * With a keep-set on (the shipped configuration), the router can only pick resident experts, so
    the pairs observed here are exactly the pairs prefill will meet. The table records which
    keep-set it was calibrated under.

Stage 2 -- fit. Per layer, recover each routed expert's output for the observed rows by calling the
engine's own MoE kernel once per routing column with a weight of 1. That is six `moe_fn` calls over
the observer rows per layer -- the same kernel prefill uses, so no second implementation of the
expert FFN exists to disagree with the first.

Cost, measured in what it moves rather than in minutes, because minutes depend on the keep-set:
stage 1 is one ordinary prefill per sequence (64 x 2,048 tokens at the shipped ~370 tok/s is about
six minutes); stage 2's six calls per layer each unpack that layer's resident experts, so it is
~6 x `resident_experts_per_layer` x 33 MB per layer unless `DSV41_PREFILL_UNPACK_CACHE_GB` is set
large enough to hold one layer's working set (2.6 GB at keep 0.36), in which case it is ~1x. **Set
it.** With the cache on, budget under half an hour end to end on top of the warm start; without it,
budget a few hours.

The output is one `.npz` of about 47 MB: `scalar`, `loss`, `norm`, and a `meta` blob recording the
corpus, the sizes, the ridge and the keep-set signature.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine.prefill_topk import TABLE_FILENAME, UNOBSERVED_LOSS  # noqa: E402


# --------------------------------------------------------------------------- calibration text
def load_texts(path: str, limit: int) -> list[tuple[str, str]]:
    """`(id, text)` pairs from a JSONL corpus, or from every file under a directory.

    The default is this repository's own trace corpus: mixed code and prose, chat-formatted,
    unlabeled. Worth stating plainly, because the paper does not get to say it -- their released
    DeepSeek matrix is calibrated partly on benchmark inputs and they disclose it as "transductive,
    benchmark-aware calibration". This one sees no benchmark, no answer and no evaluator.
    """
    out: list[tuple[str, str]] = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            p = os.path.join(path, name)
            if os.path.isfile(p):
                out.append((name, open(p, encoding="utf-8", errors="replace").read()))
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("text")
                if t:
                    out.append((str(d.get("id") or len(out)), t))
    if not out:
        raise SystemExit(f"no calibration text in {path}")
    # Deterministic, and spread across the corpus rather than taking its first n -- the corpus is
    # written in category order, so the head of it is all one kind of text.
    step = max(1, len(out) // limit)
    return out[::step][:limit]


# --------------------------------------------------------------------------- stage 1
def observer_positions(n_tokens: int, want: int) -> list[int]:
    """Evenly spaced positions over the second half of the sequence.

    The second half because an observation is only worth having where the routing is driven by
    context rather than by the first few tokens of a template; evenly spaced because a contiguous
    block would over-weight one passage of one document.
    """
    lo = n_tokens // 2
    n = min(want, max(1, n_tokens - lo))
    if n == 1:
        return [n_tokens - 1]
    step = (n_tokens - 1 - lo) / (n - 1)
    return sorted({lo + int(round(i * step)) for i in range(n)})


def observe(eng, texts, max_len: int, want_obs: int, log=print):
    """Prefill each sequence and keep the MoE input and routing at the observer positions.

    Returns `{layer: (rows bf16 [n, D] cpu, idx int64 [n, k] cpu, w float32 [n, k] cpu)}`.
    """
    m = eng.model
    keep: dict[int, list] = {}
    # `moe_in` fires once per chunk per layer, and the tap is not told the chunk's offset, so it
    # is counted here: the rows a layer has already seen in this sequence ARE its offset.
    seen: dict[int, int] = {}
    wanted: set[int] = set()

    def tap(name, L, t):
        if name == "moe_in":
            base = seen.get(L, 0)
            seen[L] = base + t.shape[0]
            sel = sorted(p - base for p in wanted if base <= p < base + t.shape[0])
            # Set or CLEAR, every time: a chunk with no observers in it must not leave the
            # previous chunk's row selection standing for the routing taps that follow.
            if sel:
                tap.sel[L] = torch.tensor(sel, device=t.device)
                keep.setdefault(L, [[], [], []])[0].append(t[tap.sel[L]].detach().to("cpu"))
            else:
                tap.sel.pop(L, None)
        elif name in ("route_idx", "route_w") and L in tap.sel:
            slot = 1 if name == "route_idx" else 2
            keep[L][slot].append(t[tap.sel[L]].detach().to("cpu"))

    total = 0
    for n, (sid, text) in enumerate(texts, 1):
        ids = eng.tokenizer.encode(text, add_special_tokens=False)[:max_len]
        if len(ids) < 8:
            continue
        wanted = set(observer_positions(len(ids), want_obs))
        seen.clear()
        tap.sel = {}
        eng._reset()
        m.begin_prompt()
        m.tap = tap
        t0 = time.perf_counter()
        try:
            m.forward(torch.tensor(ids, dtype=torch.long, device=eng.device), 0, prefill=True,
                      need_logits=False, encoder_only=bool(eng.swa_replay))
        finally:
            m.tap = None
        total += len(wanted)
        log(f"  [{n}/{len(texts)}] {sid}: {len(ids)} tokens, {len(wanted)} observers, "
            f"{time.perf_counter() - t0:.1f} s")

    if not keep:
        raise SystemExit("no MoE layer produced a tap -- is this an engine with routed experts?")
    out = {}
    for L, (rows, idx, w) in sorted(keep.items()):
        out[L] = (torch.cat(rows), torch.cat(idx), torch.cat(w))
    log(f"observed {total} tokens across {len(out)} MoE layers "
        f"(layers {min(out)}..{max(out)})")
    return out


# --------------------------------------------------------------------------- stage 2
# Inference mode is scoped to the two compute functions and NOT to the engine's construction.
# The engine's own entry points (engine/model.py forward, decode) run under inference mode, so the
# scratch they allocate lazily is an inference tensor and a kernel that writes it from outside
# fails ("Inplace update to inference tensor outside InferenceMode"; the third calibration run).
# The arena, in turn, must be a normal tensor, because the slot loader writes it from worker
# threads that are outside any mode (the fourth run, with main() wrapped, failed there).
@torch.inference_mode()
def expert_outputs(eng, L: int, rows: torch.Tensor, idx: torch.Tensor):
    """`[n, k, D]` -- each observed token's routed expert outputs, unweighted.

    One call to the engine's own `moe_fn` per routing column, with a weight of exactly 1, so the
    result is `E_e(x)` and not `alpha_e(x) E_e(x)`. Using the shipped kernel rather than a
    dequantise-and-GEMM reference is the point: the scalars are being fitted to the outputs prefill
    will actually produce, quantisation included.
    """
    a = eng.args
    n, k = idx.shape
    lut = getattr(eng.model, "slot_lut", None)
    ones = torch.ones((n, 1), dtype=torch.float32, device=rows.device)
    out = torch.empty((n, k, rows.shape[1]), dtype=torch.float32, device=rows.device)
    for j in range(k):
        col = idx[:, j:j + 1]
        slots = (lut[L][col] if lut is not None else eng.store.resolve(L, col, True))
        out[:, j, :] = eng.moe_fn(rows, slots.contiguous().to(torch.int32), ones,
                                  eng.store.arena, a.swiglu_limit).float()
    return out


@torch.inference_mode()
def accumulate(out: torch.Tensor, idx: torch.Tensor, n_experts: int, batch: int = 256):
    """The three pairwise sums plus the per-expert norm statistics, for one layer.

    Everything is derived from the per-token Gram matrix of the routed expert outputs, so the cost
    is `n * k * k * D` multiply-adds and not one pass per pair.
    """
    dev = out.device
    E = n_experts
    A = torch.zeros(E * E, dtype=torch.float64, device=dev)
    B = torch.zeros(E * E, dtype=torch.float64, device=dev)
    C = torch.zeros(E * E, dtype=torch.float64, device=dev)
    sq = torch.zeros(E, dtype=torch.float64, device=dev)
    cnt = torch.zeros(E, dtype=torch.float64, device=dev)
    n, k, _ = out.shape

    for s0 in range(0, n, batch):
        o = out[s0:s0 + batch]                                  # [b, k, D]
        e = idx[s0:s0 + batch]                                  # [b, k]
        G = torch.einsum("bpd,bqd->bpq", o, o).double()         # [b, k, k]
        diag = torch.diagonal(G, dim1=-2, dim2=-1)              # [b, k] = ||o_p||^2
        nrm = diag.clamp_min(0).sqrt()                          # [b, k] = n_p

        sq.index_add_(0, e.reshape(-1), diag.reshape(-1))
        cnt.index_add_(0, e.reshape(-1), torch.ones_like(diag).reshape(-1))

        # source p (the omitted one, u), target q (the retained one, v); weight w_i = n_p
        w = nrm.unsqueeze(-1)                                   # [b, k, 1] broadcast over q
        flat = (e.unsqueeze(-1) * E + e.unsqueeze(1)).reshape(-1)   # [b*k*k] -> s*E + t
        A.index_add_(0, flat, (w * G).reshape(-1))                        # n_p * <o_p, o_q>
        B.index_add_(0, flat, (w * diag.unsqueeze(1)).reshape(-1))        # n_p * ||o_q||^2
        C.index_add_(0, flat, (w * diag.unsqueeze(-1)).reshape(-1))       # n_p * ||o_p||^2

    return (A.view(E, E), B.view(E, E), C.view(E, E), sq, cnt)


def solve(A, B, C, sq, cnt, ridge: float, clip: float | None, eps: float = 1e-12):
    """Eq. 8 and Eq. 9, plus `h_e`. Returns numpy `(scalar, loss, norm)` for one layer."""
    E = A.shape[0]
    observed = B > 0
    scalar = torch.where(observed, A / (B + ridge), torch.zeros_like(A))
    if clip is not None:
        # The Qwen configuration clips to [-4, 4]; the released DeepSeek artifact does not
        # ("unbounded least-squares scalars"), which is why this is off by default.
        scalar = scalar.clamp(-clip, clip)
    resid = C - 2.0 * scalar * A + scalar * scalar * B
    loss = torch.where(observed, (resid / (C + eps)).clamp_min(0.0),
                       torch.full_like(A, UNOBSERVED_LOSS))
    # "self-pair coefficients are fixed to one" (Appendix B): an expert reconstructs itself
    # exactly, so folding onto yourself is a no-op rather than a fit.
    d = torch.arange(E, device=A.device)
    scalar[d, d] = 1.0
    loss[d, d] = 0.0

    h = torch.where(cnt > 0, (sq / cnt.clamp_min(1)).clamp_min(0).sqrt(), torch.zeros_like(sq))
    seen = h[cnt > 0]
    if seen.numel():
        # An expert this calibration never saw must not be ranked first or last by accident, so it
        # gets the layer's median norm -- which makes Eq. 20 fall back to ranking it by weight.
        h = torch.where(cnt > 0, h, torch.full_like(h, float(seen.median())))
    return (scalar.float().cpu().numpy(), loss.float().cpu().numpy(), h.float().cpu().numpy())


# --------------------------------------------------------------------------- driver
def build_engine(args):
    from engine.v41_engine import V41Engine
    keep = os.environ.get("PRUNE_KEEP", "")
    return V41Engine(
        args.model_dir, max_seq=args.max_seq,
        trace_stats=os.environ.get("TRACE_STATS") or None,
        spec=os.environ.get("DSV41_SPEC", "1") == "1",
        prune_keep=float(keep) if keep.replace(".", "", 1).isdigit() else None,
        arena_gb=float(os.environ["ARENA_GB"]) if os.environ.get("ARENA_GB") else None,
        transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "8")),
        keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "10")),
        expert_format=os.environ.get("EXPERT_FORMAT", "cb3"),
        expert_topics=os.environ.get("EXPERT_TOPICS") or None,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR")
                    or os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--out", default=None, help=f"default <model_dir>/{TABLE_FILENAME}")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "corpus", "trace_corpus_v3.jsonl"),
                    help="JSONL with a `text` field, or a directory of text files")
    ap.add_argument("--sequences", type=int, default=64, help="calibration sequences (64)")
    ap.add_argument("--observer-tokens", type=int, default=64,
                    help="observed tokens per sequence, the paper's cap (64)")
    ap.add_argument("--max-len", type=int, default=2048, help="tokens per sequence (2048)")
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--ridge", type=float, default=1e-3, help="lambda in Eq. 8 (1e-3, the paper's)")
    ap.add_argument("--clip", type=float, default=None,
                    help="clip the scalars to +-CLIP; off, as in the paper's DeepSeek artifact")
    ap.add_argument("--fit-batch", type=int, default=256, help="observer rows per Gram batch")
    a = ap.parse_args()

    # The routing taps this reads fire AFTER the prefill reduction, so calibrating with the
    # reduction already on would fit the scalars to routing that had already been folded once.
    for var in ("DSV41_PREFILL_TOPK", "DSV41_TOPK"):
        if os.environ.get(var, "").strip() not in ("", "0", "off"):
            raise SystemExit(f"unset {var} before calibrating: the tables must be fitted to the "
                             "checkpoint's own routing, not to a reduced one")
    if os.environ.get("DSV41_PRUNE_MODE", "").strip() == "drop":
        raise SystemExit("DSV41_PRUNE_MODE=drop zeroes displaced routes, so a calibration run "
                         "under it would fit scalars to weights the fold arm never sees")

    out_path = a.out or os.path.join(a.model_dir, TABLE_FILENAME)
    texts = load_texts(a.corpus, a.sequences)
    print(f"calibration corpus: {a.corpus} -> {len(texts)} sequences, "
          f"<= {a.max_len} tokens each, <= {a.observer_tokens} observers each")

    eng = build_engine(a)
    cfg = eng.config()
    print("config:", json.dumps({k: cfg[k] for k in (
        "expert_format", "kernel", "routed_topk", "prune_keep", "prune_mode", "arena_slots",
        "prefill_chunk", "unpack_cache_gb", "swa_replay") if k in cfg}))
    if not cfg.get("unpack_cache_gb"):
        print("NOTE: DSV41_PREFILL_UNPACK_CACHE_GB is not set. Stage 2 re-unpacks every expert of "
              "a layer six times instead of once; set it to one layer's working set and rerun.")

    n_layers, E, K = eng.args.n_layers, eng.args.n_routed_experts, eng.args.n_activated_experts
    print(f"\n--- stage 1: observe ({n_layers} layers, {E} experts, top-{K})")
    t0 = time.perf_counter()
    obs = observe(eng, texts, a.max_len, a.observer_tokens)
    t_observe = time.perf_counter() - t0

    print(f"\n--- stage 2: fit")
    scalar = np.zeros((n_layers, E, E), dtype=np.float32)
    loss = np.full((n_layers, E, E), UNOBSERVED_LOSS, dtype=np.float32)
    norm = np.zeros((n_layers, E), dtype=np.float32)
    # Layers prefill never runs keep the sentinel, and a sentinel loss is a confidence of 0, so a
    # fold that somehow reached one of them would degrade to the proportional fallback rather than
    # to a scalar of zero. The identity diagonal still has to be right there too.
    d = np.arange(E)
    scalar[:, d, d] = 1.0
    loss[:, d, d] = 0.0

    t1 = time.perf_counter()
    covered = []
    for L, (rows, idx, w) in sorted(obs.items()):
        rows = rows.to(eng.device).contiguous()
        idx = idx.to(eng.device)
        out = expert_outputs(eng, L, rows, idx)
        A, B, C, sq, cnt = accumulate(out, idx, E, batch=a.fit_batch)
        s_l, l_l, h_l = solve(A, B, C, sq, cnt, a.ridge, a.clip)
        scalar[L], loss[L], norm[L] = s_l, l_l, h_l
        seen = int((l_l < UNOBSERVED_LOSS).sum()) - E
        covered.append(seen)
        print(f"  layer {L:>2}: {rows.shape[0]} rows, {int((cnt > 0).sum())} experts, "
              f"{seen} directed pairs, median loss "
              f"{float(np.median(l_l[l_l < UNOBSERVED_LOSS])):.3f}, "
              f"scalar {float(np.median(s_l)):.3f} median")
        del rows, idx, out, A, B, C
        torch.cuda.empty_cache()
    t_fit = time.perf_counter() - t1

    meta = {
        "paper": "arXiv:2608.24938",
        "corpus": os.path.relpath(a.corpus, ROOT) if a.corpus.startswith(ROOT) else a.corpus,
        "sequences": len(texts),
        "observer_tokens": int(sum(v[0].shape[0] for v in obs.values()) / max(len(obs), 1)),
        "max_len": a.max_len,
        "ridge": a.ridge,
        "clip": a.clip,
        "layers_observed": sorted(obs),
        "keep_sig": eng.keep_signature(),
        "prune_keep": cfg.get("prune_keep"),
        "expert_format": cfg.get("expert_format"),
        "swa_replay": cfg.get("swa_replay"),
        "observe_s": round(t_observe, 1),
        "fit_s": round(t_fit, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(out_path, scalar=scalar, loss=loss, norm=norm, meta=np.array(json.dumps(meta)))
    size = os.path.getsize(out_path)
    print(f"\nwrote {out_path} ({size / 1e6:.1f} MB)")
    print(f"  {sum(covered)} directed pairs over {len(covered)} layers "
          f"({sum(covered) / max(len(covered) * E * (E - 1), 1):.1%} of the observed layers' pairs)")
    print(f"  observe {t_observe / 60:.1f} min, fit {t_fit / 60:.1f} min")
    print(f"\nTo use it:  DSV41_PREFILL_TOPK=4  (DSV41_PREFILL_FOLD=exfold is the default)")


if __name__ == "__main__":
    main()
