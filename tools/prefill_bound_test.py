"""Is a prefill chunk's MoE compute-bound or bandwidth-bound? Force the router's k and see.

Run on the box, with the engine's own environment:

    python3 tools/prefill_bound_test.py
    python3 tools/prefill_bound_test.py --chunk 2048 --layer 10 --k 6,4,3 --repeats 3
    python3 tools/prefill_bound_test.py --json results/prefill_bound.json

What it does. It runs one real prefill chunk, taps the MoE input and the router's choice at ONE
backbone layer, and then re-runs that single layer's MoE at k = 6, 4 and 3 -- three repeats,
CUDA-synchronised, median reported. Two timings per k:

  * `layer`  -- the whole `Model.moe`: router, slot lookup, the routed kernels and the shared
                expert. This is what a prefill chunk actually pays per layer. The shared expert
                does not move with k, so this number can only ever scale sub-linearly.
  * `routed` -- `moe_fn` alone: on this configuration that is the CB3 -> FP4 unpack plus the two
                FP4 grouped launches, and nothing else. This is the number the decision rule reads.

It also reports, per k, how many DISTINCT experts the chunk still touches, because that is the
quantity the unpack cost is proportional to -- and at 2,048 tokens it barely moves with k.

The decision rule (also written down in docs/gemm-dispatch.md):

    routed(3) / routed(6) <= 0.65   -> time scales with k: the prefill MoE is COMPUTE-bound, and
                                      an ExFold-style top-k reduction in prefill (arXiv 2608.24938,
                                      validated on DeepSeek-V4-Flash) is worth implementing.
    routed(3) / routed(6) >= 0.90   -> flat: the chunk is BANDWIDTH/UNPACK-bound. Cutting k buys
                                      nothing; the levers are the ones that cut bytes moved per
                                      chunk, not pairs computed.
    in between                      -> partly each. Report the number, do not round it to a verdict.

The k override is `DSV41_PREFILL_TOPK_TEST` (engine/prefill_topk.py): prefill-only, off by
default, byte-identical off, pinned by tools/test_prefill_topk.py. This script drives it by
assigning the module attribute rather than the environment variable, because an A/B on this box
has to happen inside ONE process -- a second process lays the 89 GB arena down at a different
offset and that alone moves a streaming kernel's time (NOTES, 2026-09-11 "the head" section).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

from engine import prefill_topk as PT  # noqa: E402


def median_ms(fn, repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def build_engine(args):
    from engine.v41_engine import V41Engine
    keep = os.environ.get("PRUNE_KEEP", "")
    eng = V41Engine(
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
    cfg = eng.config()
    print("config:", json.dumps({k: cfg[k] for k in (
        "expert_format", "kernel", "dense_fp4", "head_fmt", "routed_topk", "prune_keep",
        "arena_slots", "prefill_chunk") if k in cfg}))
    return eng


def prompt_ids(eng, n: int):
    para = ("The engine streams three-bit routed experts out of an arena on the device, unpacks "
            "the ones a chunk touches back into packed FP4, and runs the grouped kernel over "
            "them. Attention keeps a sliding window of 128 positions plus every completed "
            "compressed group, selected by an indexer that scores in fixed blocks. ")
    ids = []
    while len(ids) < n:
        ids += eng.tokenizer.encode(para, add_special_tokens=False)
    return ids[:n]


def capture_layer(eng, ids, layer: int):
    """Run one prefill chunk and keep layer `layer`'s MoE input and router output.

    `route_w` comes out of the tap already normalised over the checkpoint's six and scaled by
    `route_scale`; since those weights are proportional to the router's raw scores, renormalising
    the first k columns of them IS the top-k' routing this test wants, with no second copy of the
    router's arithmetic living in this file.
    """
    m = eng.model
    got = {}

    def tap(name, L, t):
        if L == layer and name in ("moe_in", "route_idx", "route_w"):
            got.setdefault(name, t.detach().clone())

    eng._reset()
    m.begin_prompt()
    m.tap = tap
    try:
        m.forward(ids, 0, prefill=True, need_logits=False, encoder_only=bool(eng.swa_replay))
    finally:
        m.tap = None
    missing = {"moe_in", "route_idx", "route_w"} - set(got)
    if missing:
        raise SystemExit(f"layer {layer} produced no {sorted(missing)} tap -- is it inside the "
                         f"encoder half (0..{eng.args.candidate_source_layer}) with SWA replay on?")
    return got


def route_for_k(eng, layer, idx6, w6, k):
    """(slots int32 [T, k], weights fp32 [T, k]) for the top-k' of the router's own six."""
    a = eng.args
    idx = idx6[:, :k].contiguous()
    w = w6[:, :k].float()
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
    lut = getattr(eng.model, "slot_lut", None)
    slots = lut[layer][idx] if lut is not None else eng.store.resolve(layer, idx, True)
    return slots.contiguous().to(torch.int32), w.contiguous()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR")
                    or os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--layer", type=int, default=10, help="backbone layer to measure (default 10)")
    ap.add_argument("--k", default="6,4,3", help="top-k values to force, comma separated")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ks = [int(x) for x in a.k.split(",") if x.strip()]
    if max(ks) > 6:
        raise SystemExit("--k above the checkpoint's 6 would need weights the router never produced")

    eng = build_engine(a)
    m, args_ = eng.model, eng.args
    ids = torch.tensor(prompt_ids(eng, a.chunk), dtype=torch.long, device=eng.device)

    print(f"\nprefill chunk {a.chunk} tokens, layer {a.layer}, {a.repeats} repeats, median")
    got = capture_layer(eng, ids, a.layer)
    y = got["moe_in"]
    idx6, w6 = got["route_idx"], got["route_w"]
    w = m.W.layers[a.layer]
    print(f"captured moe_in {tuple(y.shape)} {y.dtype}, route_idx {tuple(idx6.shape)}")

    rows = []
    for k in ks:
        slots, wk = route_for_k(eng, a.layer, idx6, w6, k)
        distinct = int(torch.unique(slots[slots >= 0]).numel())

        def routed():
            eng.moe_fn(y, slots, wk, eng.store.arena, args_.swiglu_limit)

        def layer_moe():
            m.moe(y, w, a.layer, True, eng.store, eng.store.arena, args_.n_routed_experts)

        PT.OVERRIDE = k
        try:
            routed(); layer_moe()                      # warm: first call at a new k compiles nothing
            t_routed = median_ms(routed, a.repeats)     # but does allocate the scratch
            t_layer = median_ms(layer_moe, a.repeats)
        finally:
            PT.OVERRIDE = None
        rows.append({"k": k, "routed_ms": round(t_routed, 3), "layer_ms": round(t_layer, 3),
                     "pairs": int(slots.numel()), "distinct_experts": distinct})
        print(f"  k={k}  routed {t_routed:8.3f} ms   layer {t_layer:8.3f} ms   "
              f"pairs {slots.numel():7d}   distinct experts {distinct}")

    base = next(r for r in rows if r["k"] == max(ks))
    print(f"\n  {'k':>3}  {'routed rel':>11}  {'layer rel':>10}  {'k rel':>7}  {'experts rel':>11}")
    for r in rows:
        print(f"  {r['k']:>3}  {r['routed_ms'] / base['routed_ms']:>11.3f}  "
              f"{r['layer_ms'] / base['layer_ms']:>10.3f}  {r['k'] / base['k']:>7.3f}  "
              f"{r['distinct_experts'] / max(base['distinct_experts'], 1):>11.3f}")

    lo = min(rows, key=lambda r: r["k"])
    ratio = lo["routed_ms"] / base["routed_ms"]
    k_ratio = lo["k"] / base["k"]
    if ratio <= 0.65:
        verdict = (f"COMPUTE-BOUND: routed time at k={lo['k']} is {ratio:.2f}x of k={base['k']} "
                   f"(k itself is {k_ratio:.2f}x). An ExFold-style top-k reduction in prefill is "
                   f"worth implementing.")
    elif ratio >= 0.90:
        verdict = (f"BANDWIDTH/UNPACK-BOUND: routed time at k={lo['k']} is {ratio:.2f}x of "
                   f"k={base['k']} while k is {k_ratio:.2f}x. Cutting k buys nothing here; the "
                   f"levers are the ones that cut bytes moved per chunk.")
    else:
        verdict = (f"MIXED: routed time at k={lo['k']} is {ratio:.2f}x of k={base['k']}, against "
                   f"{k_ratio:.2f}x of the pairs. Partly compute, partly bytes -- report the "
                   f"number, do not round it to a verdict.")
    print(f"\n{verdict}")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"config": eng.config(), "chunk": a.chunk, "layer": a.layer,
                       "repeats": a.repeats, "rows": rows,
                       "routed_ratio": round(ratio, 4), "verdict": verdict}, f, indent=2, default=str)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
