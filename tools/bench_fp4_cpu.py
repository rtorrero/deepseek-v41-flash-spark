"""bench_fp4_cpu.py -- what the RAM expert tier actually costs, measured.

The tier planner (`tools/tier_plan.py`) needs one number from the real hardware: how fast the
CPU side can walk an expert. This measures it, on whatever box it is run on, with no checkpoint
and no GPU -- the experts are synthetic bytes with the real shapes, because the kernel's cost is
a function of the byte count, not of what the bytes mean.

Reported per configuration:

    ms/expert      one expert forward: the fused GEMV for gate and up, the clamped SwiGLU, the
                   fused GEMV for down
    GB/s           the 18,800,640 stored bytes of that expert divided by that time -- the honest
                   denominator, because the fused kernel reads each stored byte exactly once.
                   The torch path's denominator is the same, but it touches ~6x the bytes, so
                   the comparison is a memory-traffic comparison, not a micro-optimisation.
    tok/s          projected from 240 expert passes per generated token (40 layers x 6 routed
                   experts), which is the model's routing, not an assumption.

Usage:
    bash tools/build_fp4_cpu.sh
    python3 tools/bench_fp4_cpu.py [--reps N] [--threads 1,2,4,8] [--ram-fraction 0.6]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fp4_expert_cpu as C  # noqa: E402

EXPERT_BYTES = 18_800_640
LAYERS, TOPK = 40, 6  # config.json: 40 MoE layers, n_activated_experts = 6
PASSES_PER_TOKEN = LAYERS * TOPK


def cpu_banner() -> str:
    name, cores, flags = "unknown", os.cpu_count(), []
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and name == "unknown":
                name = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not flags:
                f = set(line.split(":", 1)[1].split())
                flags = [x for x in ("avx2", "fma", "avx512f", "amx_int8") if x in f]
            if name != "unknown" and flags:
                break
    except OSError:
        pass
    return f"{name} | {cores} threads | {' '.join(flags) or 'no vector flags found'}"


def mem_bandwidth_gbs(nbytes: int = 512 << 20) -> float:
    """A practical ceiling for this box: a 512 MB copy (read + write) timed in torch."""
    a = torch.empty(nbytes // 4, dtype=torch.float32)
    b = torch.empty_like(a)
    for _ in range(2):
        b.copy_(a)
    t = time.perf_counter()
    reps = 5
    for _ in range(reps):
        b.copy_(a)
    dt = (time.perf_counter() - t) / reps
    return 2 * nbytes / dt / 1e9  # read + write


def make_expert(gen, scale_lo=120, scale_hi=132):
    def r(n, k):
        return (torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8),
                torch.randint(scale_lo, scale_hi, (n, k // 32), generator=gen, dtype=torch.uint8))
    w1, s1 = r(C.INTER, C.DIM)
    w3, s3 = r(C.INTER, C.DIM)
    w2, s2 = r(C.DIM, C.INTER)
    return w1, s1, w2, s2, w3, s3


def timed(fn, reps: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def child_measure(threads: int, reps: int) -> dict:
    """One measurement, in a fresh process.

    libgomp reads OMP_NUM_THREADS once, at the first parallel region, and never again, so a
    sweep that sets it in a loop measures the first value for every row -- which is exactly the
    nonsense the first version of this file printed (1 thread "slower" than 2 by 20x). The only
    honest way to sweep thread counts is one process each.
    """
    import json

    torch.set_num_threads(1)  # this child measures the C kernel, not BLAS
    arena = C.ExpertArenaCPU(1)
    gen = torch.Generator().manual_seed(21)
    arena.load_slot(0, *make_expert(gen))
    x = torch.randn(1, C.DIM, generator=gen) * 0.5
    cache = C.kernels()
    f = lambda: C.expert_forward_slot(x, 0, arena, weights=torch.ones(1), out_dtype=torch.float32, cache=cache)
    ms = timed(f, reps) * 1e3
    print("JSON " + json.dumps({"threads": threads, "ms": ms}))
    return {"threads": threads, "ms": ms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=str, default="", help="comma list; default: 1,2,4,8,12 and all")
    ap.add_argument("--ram-fraction", type=float, default=0.6,
                    help="share of the routed experts assumed to live in the RAM tier")
    ap.add_argument("--gpu-gbs", type=float, default=400.0,
                    help="effective GPU expert bandwidth for the projection; replace with your own measured number")
    ap.add_argument("--child", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        child_measure(args.child, args.reps)
        return 0

    cache = C.kernels()
    print("bench_fp4_cpu.py -- the RAM expert tier, measured\n")
    print(f"  host        {cpu_banner()}")
    print(f"  kernel      {'compiled' if cache.available() else 'MISSING -- ' + C.CpuKernels.build_hint()}")
    if not cache.available():
        print("\nnothing to measure without the kernel; the torch fallback is benchmarked in the tests.")
        return 1
    copy_gbs = mem_bandwidth_gbs()
    print(f"  bandwidth   {copy_gbs:.1f} GB/s (512 MiB copy, read+write; a read-only stream is the ceiling below)")

    all_threads = os.cpu_count() or 1
    thread_list = [int(t) for t in args.threads.split(",") if t] or [1, 2, 4, 8, 12, all_threads]
    thread_list = sorted(set(min(t, all_threads) for t in thread_list))

    # One synthetic expert with the real shapes: the kernel's cost is a function of the byte
    # count, so nothing about the numbers depends on what the weights mean.
    arena = C.ExpertArenaCPU(1)
    gen = torch.Generator().manual_seed(21)
    arena.load_slot(0, *make_expert(gen))
    x = torch.randn(1, C.DIM, generator=gen) * 0.5

    print(f"\n  one expert forward ({EXPERT_BYTES / 1e6:.2f} MB stored), T=1\n")
    print(f"  {'ask':>8} {'libgomp':>8} {'ms/expert':>11} {'GB/s':>7} {'vs copy':>8} {'tok/s (all CPU)':>17}")
    rows = []
    for nt in thread_list:
        actual = cache.set_threads(nt)
        ms = timed(lambda: C.expert_forward_slot(x, 0, arena, weights=torch.ones(1),
                                                 out_dtype=torch.float32, cache=cache), args.reps) * 1e3
        gbs = EXPERT_BYTES / (ms / 1e3) / 1e9
        tps = 1000.0 / (ms * PASSES_PER_TOKEN)
        rows.append((actual, ms, gbs))
        print(f"  {nt:>8} {actual:>8} {ms:>11.3f} {gbs:>7.1f} {gbs / copy_gbs * 100:>7.0f}% {tps:>17.1f}")

    # Breakdown: which of the three GEMVs and the epilogue the time actually goes to.
    cache.set_threads(all_threads)
    print(f"\n  where the time goes (T=1, {cache.max_threads()} OpenMP threads)\n")
    w1r, s1r, w2r, s2r, w3r, s3r = (arena.w1[0], arena.s1[0], arena.w2[0], arena.s2[0],
                                    arena.w3[0], arena.s3[0])
    x0 = x[0].contiguous()
    g = torch.empty(C.INTER)
    u = torch.empty(C.INTER)
    h = torch.empty(C.INTER)
    y = torch.empty(C.DIM)
    parts = [
        ("w1 gate  2304x5120", lambda: cache.gemv(w1r, s1r, x0, g)),
        ("w3 up    2304x5120", lambda: cache.gemv(w3r, s3r, x0, u)),
        ("swiglu   2304", lambda: cache.swiglu(g, u, h, 10.0, 1.0)),
        ("w2 down  5120x2304", lambda: cache.gemv(w2r, s2r, h, y)),
    ]
    tot = 0.0
    for name, fn in parts:
        ms = timed(fn, args.reps) * 1e3
        tot += ms
        print(f"  {name:<20} {ms:>8.3f} ms")
    print(f"  {'sum':<20} {tot:>8.3f} ms")

    # The projection the tier planner consumes.
    best = min(rows, key=lambda r: r[1])
    f = args.ram_fraction
    cpu_ms = best[1] * PASSES_PER_TOKEN * f
    gpu_ms = (PASSES_PER_TOKEN * (1 - f) * EXPERT_BYTES) / (args.gpu_gbs * 1e9) * 1e3
    print(f"\n  projection at {f:.0%} of the routed experts in the RAM tier ({best[0]} CPU threads, "
          f"{best[2]:.1f} GB/s)\n")
    print(f"  {'CPU tier':<26} {cpu_ms:>8.1f} ms/token  -> {1000 / cpu_ms:>6.1f} tok/s on its own")
    print(f"  {'GPU tier @ ' + str(args.gpu_gbs) + ' GB/s':<26} {gpu_ms:>8.1f} ms/token  -> {1000 / gpu_ms:>6.1f} tok/s on its own")
    print(f"  {'serialised (worst case)':<26} {cpu_ms + gpu_ms:>8.1f} ms/token  -> {1000 / (cpu_ms + gpu_ms):>6.1f} tok/s")
    print(f"  {'overlapped (best case)':<26} {max(cpu_ms, gpu_ms):>8.1f} ms/token  -> {1000 / max(cpu_ms, gpu_ms):>6.1f} tok/s")
    print("\n  The overlap case assumes the CPU expert work of layer L runs while layer L's attention")
    print("  is on the GPU. That is a scheduling property of the runtime, not a property of these")
    print("  kernels, so treat it as the target the integration has to reach, not as a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
