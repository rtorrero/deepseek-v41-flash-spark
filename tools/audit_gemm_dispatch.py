"""Which CUDA kernels actually run in a prefill chunk and in a decode step.

Run on the box, with the engine's own environment (`.env` / `./start.sh`'s variables):

    python3 tools/audit_gemm_dispatch.py                      # 2,048-token chunk + one decode step
    python3 tools/audit_gemm_dispatch.py --chunk 512
    python3 tools/audit_gemm_dispatch.py --phase prefill --json results/gemm_dispatch.json
    python3 tools/audit_gemm_dispatch.py --expect _fp4_linear_kernel,_moe_up_kernel,cutlass

Why. Every quantised matmul in this engine dispatches in Python, on the TYPE of the weight object
and on the row count of the call (tools/v41_ref.py `dense`, `wo_a_proj`, `head_logits`;
tools/cb3_moe.py `moe_forward_v3`). Each of those dispatch points has a branch that does NOT run
the packed kernel -- a transient bf16 dequant into cuBLAS, a blocked dequant of the LM head, an
unpack of the 3-bit experts back to FP4 -- and the branch taken is invisible from the outside: the
answer is correct either way, only slower. The public SGLang win on this same checkpoint (35 ->
118 tok/s) was exactly one such misdispatch found by reading kernel names in a profile. So read
kernel names.

What it prints, per phase:

  * wall time of the region measured WITHOUT the profiler attached (the profiler's own overhead is
    large when a phase issues many small launches, which prefill does),
  * total GPU kernel time and the GPU-busy fraction of that wall. A phase whose kernels account
    for a small part of its wall clock is launch-bound, and no kernel-level change will help it,
  * the top-20 CUDA kernels by total device time, with call counts,
  * anything above --threshold (default 5 %) of GPU time whose name matches nothing in --expect,
  * every kernel whose name matches a known fallback shape (`*fallback*`, `*unpack*`,
    `*elementwise*`, memcpy/memset), at any share.

`--expect` is the list of kernel-name substrings this configuration SHOULD be spending its time
in. The default is the packed-format kernels plus cuBLAS/cutlass; pass your own when auditing a
different DSV41_DENSE_FP4 / DSV41_HEAD_FMT / EXPERT_FORMAT combination.

Nothing here writes to the model, the caches or the arena beyond what a normal request does, and
it never starts a server.
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
from torch.profiler import ProfilerActivity, profile  # noqa: E402

# The kernels the shipped configuration (EXPERT_FORMAT=cb3, DSV41_DENSE_FP4=attn,wo_a,
# DSV41_HEAD_FMT=fp8) is supposed to be spending its time in.
DEFAULT_EXPECT = (
    "_cb3v3_up_kernel", "_cb3v3_down_kernel",      # CB3 experts, decode
    "_moe_up_kernel", "_moe_down_kernel",          # FP4 experts (and the prefill path's scratch)
    "_route_kernel",                               # routing, one small launch per MoE call
    "_fp4_linear_kernel", "_fp4_grouped_kernel",   # dense attn projections + wo_a
    "_fp8_linear_kernel", "_fp8_grouped_kernel",   # whatever stayed in the stored fp8 format
    "_skinny_kernel", "_hc_kernel",                # the fp32 HC mix GEMM and the fused Sinkhorn
    "cutlass", "gemm", "sgemm", "wgrad", "ampere", "sm80", "sm90", "sm120",
)

# Named and shamed at any share: these are what a misdispatch LOOKS like in a kernel list.
SUSPECT = ("fallback", "unpack", "elementwise", "memcpy", "memset", "copy_",
           "dequant", "repeat_interleave", "cat_", "index_select")


def _dev_time(evt) -> float:
    """Self device time in microseconds, across the attribute renames of the last few torch versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total", "device_time_total", "cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _is_cuda(evt) -> bool:
    dt = getattr(evt, "device_type", None)
    return dt is not None and str(dt).endswith("CUDA")


def kernel_table(prof):
    """[(name, calls, total_us)] for the CUDA kernels of a profile, largest first.

    `key_averages()` carries both the aten-op rows and the CUDA kernel rows; only the second group
    has a CUDA device type, and only those rows carry the names a dispatch bug shows up in.
    """
    rows = []
    for evt in prof.key_averages():
        if not _is_cuda(evt):
            continue
        us = _dev_time(evt)
        if us <= 0:
            continue
        rows.append((str(getattr(evt, "key", evt.__class__.__name__)), int(getattr(evt, "count", 0)), us))
    rows.sort(key=lambda r: -r[2])
    return rows


def matches(name: str, needles) -> bool:
    low = name.lower()
    return any(n.lower() in low for n in needles)


def report(title: str, rows, wall_s: float, expect, threshold: float, top: int = 20):
    total_us = sum(r[2] for r in rows)
    calls = sum(r[1] for r in rows)
    busy = (total_us / 1e6) / wall_s if wall_s > 0 else 0.0
    print(f"\n=== {title} ===")
    print(f"wall (no profiler attached) {wall_s * 1e3:9.1f} ms")
    print(f"GPU kernel time             {total_us / 1e3:9.1f} ms   "
          f"({busy * 100:.1f} % of wall)   {calls} launches, {len(rows)} distinct kernels")
    if busy < 0.75:
        print(f"  NOTE: only {busy * 100:.0f} % of this phase is GPU-busy time. It is launch- or "
              f"host-bound; kernel-level work will not move it.")
    print(f"\n  {'%GPU':>6}  {'ms':>9}  {'calls':>7}  kernel")
    for name, n, us in rows[:top]:
        flag = " " if matches(name, expect) else "*"
        print(f"{flag} {us / max(total_us, 1) * 100:6.2f}  {us / 1e3:9.2f}  {n:7d}  {name}")
    unexpected = [r for r in rows
                  if not matches(r[0], expect) and r[2] / max(total_us, 1) >= threshold]
    print(f"\n  unexpected above {threshold * 100:.0f} % of GPU time "
          f"(not matched by --expect): {len(unexpected)}")
    for name, n, us in unexpected:
        print(f"    {us / max(total_us, 1) * 100:6.2f} %  {us / 1e3:9.2f} ms  {n:7d} calls  {name}")
    susp = [r for r in rows if matches(r[0], SUSPECT)]
    print(f"  fallback-shaped kernel names (any share): {len(susp)}"
          + ("  -- none" if not susp else ""))
    for name, n, us in susp[:12]:
        print(f"    {us / max(total_us, 1) * 100:6.2f} %  {us / 1e3:9.2f} ms  {n:7d} calls  {name}")
    return {"wall_s": round(wall_s, 4), "gpu_ms": round(total_us / 1e3, 2),
            "gpu_busy_frac": round(busy, 4), "launches": calls,
            "kernels": [{"name": k, "calls": c, "ms": round(u / 1e3, 3),
                         "pct": round(u / max(total_us, 1) * 100, 2),
                         "expected": matches(k, expect)} for k, c, u in rows[:top]],
            "unexpected": [{"name": k, "calls": c, "ms": round(u / 1e3, 3)} for k, c, u in unexpected]}


def timed(fn, repeats: int = 3) -> float:
    """Median wall seconds of `fn`, each call CUDA-synchronised. No profiler attached."""
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def profiled(fn):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        fn()
        torch.cuda.synchronize()
    return prof


def build_engine(args):
    from engine.v41_engine import V41Engine
    md = args.model_dir
    eng = V41Engine(
        md, max_seq=args.max_seq,
        trace_stats=os.environ.get("TRACE_STATS") or None,
        spec=os.environ.get("DSV41_SPEC", "1") == "1",
        prune_keep=float(os.environ["PRUNE_KEEP"]) if os.environ.get("PRUNE_KEEP", "").replace(".", "", 1).isdigit() else None,
        arena_gb=float(os.environ["ARENA_GB"]) if os.environ.get("ARENA_GB") else None,
        transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "8")),
        keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "10")),
        expert_format=os.environ.get("EXPERT_FORMAT", "cb3"),
        expert_topics=os.environ.get("EXPERT_TOPICS") or None,
    )
    cfg = eng.config()
    print("config:", json.dumps({k: cfg[k] for k in (
        "expert_format", "kernel", "dense_fp4", "head_fmt", "routed_topk", "prune_keep",
        "arena_slots", "prefill_chunk", "swa_replay") if k in cfg}))
    return eng


def long_prompt_ids(eng, n_tokens: int):
    """`n_tokens` real token ids. Any ordinary text will do -- this measures kernel dispatch, not
    routing quality -- but it must be real text, because the router's choices decide how many
    distinct experts a chunk touches and that is what the MoE path costs."""
    para = ("The engine streams three-bit routed experts out of an arena on the device, unpacks "
            "the ones a chunk touches back into packed FP4, and runs the grouped kernel over "
            "them. Attention keeps a sliding window of 128 positions plus every completed "
            "compressed group, selected by an indexer that scores in fixed blocks. ")
    ids = []
    while len(ids) < n_tokens:
        ids += eng.tokenizer.encode(para, add_special_tokens=False)
    return ids[:n_tokens]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR")
                    or os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("DSV41_PREFILL_CHUNK", 2048)),
                    help="prefill chunk length to profile (default: the engine's own chunk)")
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--phase", choices=("both", "prefill", "decode"), default="both")
    ap.add_argument("--expect", default=",".join(DEFAULT_EXPECT),
                    help="comma-separated kernel-name substrings this config should be running")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="flag unexpected kernels above this share of GPU time (default 0.05)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", default=None, help="also write the numbers here")
    a = ap.parse_args()
    expect = [s.strip() for s in a.expect.split(",") if s.strip()]

    eng = build_engine(a)
    m = eng.model
    out = {"config": eng.config(), "chunk": a.chunk, "expect": expect}

    # ---------------------------------------------------------------- prefill
    if a.phase in ("both", "prefill"):
        ids_t = torch.tensor(long_prompt_ids(eng, a.chunk), dtype=torch.long, device=eng.device)
        enc_only = bool(eng.swa_replay)

        def one_chunk():
            eng._reset()
            m.begin_prompt()
            m.forward(ids_t, 0, prefill=True, need_logits=not enc_only, encoder_only=enc_only)

        one_chunk()  # compile the Triton kernels and warm the arena; never measured
        wall = timed(one_chunk, repeats=3)
        prof = profiled(one_chunk)
        out["prefill"] = report(f"prefill, one {a.chunk}-token chunk"
                                + (" (encoder-only: SWA bounded replay is on)" if enc_only else ""),
                                kernel_table(prof), wall, expect, a.threshold, a.top)
        out["prefill"]["encoder_only"] = enc_only
        del prof

    # ---------------------------------------------------------------- decode
    if a.phase in ("both", "decode"):
        short = long_prompt_ids(eng, 64)
        for _ in eng.generate(short, max_tokens=24, temperature=0.0):
            pass
        fd = eng.fast
        if fd is None:
            pos = m.c.len
            tok = torch.tensor([128799], dtype=torch.long, device=eng.device)

            def one_step():
                m.c.len = pos
                m.forward(tok, pos, prefill=False, need_logits=True)
            label = "decode, one un-graphed 1-token forward (DSV41_GRAPHS=0 / spec off)"
        else:
            pos = m.c.len
            tok = 128799
            drafts, _q = fd.draft(tok, pos - 1, 0.0)
            block = torch.cat([torch.tensor([tok], device=eng.device), drafts.clone()])
            hashes = m.hash_state(block[None], pos)[0]
            rows = {L: eng.tables[L].rows(hashes[:, li, :])
                    for li, L in enumerate(eng.args.engram_layer_ids)}

            def one_step():
                m.c.len = pos
                fd.step(block, pos, rows)
            label = f"decode, one graphed verify step ({block.numel()} tokens)"

        one_step()
        wall = timed(one_step, repeats=3)
        prof = profiled(one_step)
        out["decode"] = report(label, kernel_table(prof), wall, expect, a.threshold, a.top)
        del prof

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
