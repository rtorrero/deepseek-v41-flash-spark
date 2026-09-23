"""test_fp4_moe_sm70.py -- the ported grouped-MoE kernels, on the real card.

Runs on a V100 with no checkpoint: the experts are synthetic bytes with the real shapes, because
what is being tested is the kernel -- the decode, the dtypes, the launch configs -- not the
weights. `tools/test_fp4_moe.py` is the upstream harness that uses real layer-0 experts, and it is
the right thing to run once the checkpoint has landed; this one runs now.

Four questions, in order:

 1. Does it compile at all on sm_70? That is really "does fp16 `tl.dot` work on Volta", which the
    decode check deliberately left open by using `tl.sum`.
 2. Does it match? Against `moe_forward_reference_dtype` in fp16 (the same numerics class as the
    kernel) and, for a single expert, against an fp64 oracle over the same bytes.
 3. What does it cost? Milliseconds and effective GB/s per expert, which is the number the tier
    planner wants for `--gpu-gbs`.
 4. Which launch config? The GB10 table in `fp4_moe.py` was swept against 48 SMs and 228 KB of
    shared memory; this sweeps the same knobs on 4 warp schedulers and 64 KB.

Footprint on a busy card: an 8-slot arena is 150 MB, and the fp64 oracle is run on one expert at a
time so it never holds more than ~300 MB. Total measured at the end.

Usage (on the V100 host):
    docker run --rm --gpus device=0 --user $(id -u):$(id -g) -e HOME=/tmp \
      -e CUDA_VISIBLE_DEVICES=0 -e TRITON_CACHE_DIR=/triton-cache \
      --entrypoint python3 -v ~/Workspace/dsv41-sm70:/work -w /work \
      -v /mnt/12tb/dsv41-sm70/scratch/triton:/triton-cache \
      geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4 \
      tools/test_fp4_moe_sm70.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import budget as B  # noqa: E402
import fp4_moe as M  # noqa: E402
import fp4_moe_sm70 as S  # noqa: E402

TOPK = B.TOPK  # config.json n_activated_experts, via the repo's single source for it

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def build_arena(n_slots: int, dev, gen, scale_lo: int = 120, scale_hi: int = 126) -> M.ExpertArena:
    """Random packed experts with the real shapes and a realistic UE8M0 exponent range.

    The scale is chosen at quantisation time so that the block's largest weight lands on the top
    of the E2M1 grid: scale ~ amax/6. For expert weights that is a small number, so the bytes here
    are 120..126, i.e. 2**-7 .. 2**-1, not an arbitrary spread. The range is not decoration: the
    kernels return fp16, and a scale of 2**5 pushed the output past 65504 in the pre-flight on
    CPU -- the accumulator scaling in `_chunk_dot` protects the *products*, but the final cast is
    the engine's fp16 contract, and this is the range that keeps it honest. If the real
    checkpoint's scales turn out larger, that is a finding worth acting on, not a test to loosen.
    """
    arena = M.ExpertArena(n_slots, dev)
    for s in range(n_slots):
        def r(n, k):
            return torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8, device=dev)
        def sc(n, k):
            return torch.randint(scale_lo, scale_hi, (n, k // 32), generator=gen, dtype=torch.uint8, device=dev)
        w1, s1 = r(M.INTER, M.DIM), sc(M.INTER, M.DIM)
        w3, s3 = r(M.INTER, M.DIM), sc(M.INTER, M.DIM)
        w2, s2 = r(M.DIM, M.INTER), sc(M.DIM, M.INTER)
        arena.load_slot(s, w1, s1, w2, s2, w3, s3)
    return arena


def routing(T: int, K: int, n_slots: int, gen, dev, cpu_gen=None):
    """Distinct experts per token, and routing weights that sum to one.

    Two generators on purpose, and this is the second time this trap cost a GPU round trip: a
    device generator is required when the tensor is created on the device (`torch.randint(...,
    device=dev)`, `torch.randn(..., device=dev)`), and a *CPU* generator is required by
    `torch.randperm`, which builds its tensor on the CPU whatever the generator is. Passing the
    device generator to randperm raises "Expected a 'cpu' device type for generator but found
    'cuda'"; passing a CPU generator to a device tensor raises the mirror image.
    """
    cpu_gen = cpu_gen or torch.Generator().manual_seed(11)
    slots = torch.stack([torch.randperm(n_slots, generator=cpu_gen)[:K] for _ in range(T)])
    slots = slots.to(torch.int32).to(dev)
    w = torch.rand((T, K), generator=gen, device=dev)
    return slots, w / w.sum(dim=1, keepdim=True)


def timed(fn, reps: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tol", type=float, default=2e-2, help="relative tolerance vs the fp16 reference")
    args = ap.parse_args()

    print("test_fp4_moe_sm70.py -- the ported MoE kernels on this GPU\n")
    if not torch.cuda.is_available():
        print("no CUDA device: this test exists to run on the V100.")
        return 2
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    free0 = torch.cuda.mem_get_info()[0] / 1e6
    print(f"  device      {torch.cuda.get_device_name()} (capability {cap[0]}.{cap[1]})")
    print(f"  free VRAM   {free0:.0f} MB")
    print(f"  triton      {__import__('triton').__version__}, torch {torch.__version__}")

    gen = torch.Generator(device=dev)
    gen.manual_seed(7)
    arena = build_arena(args.slots, dev, gen)
    print(f"  arena       {args.slots} slots, {arena.bytes_per_slot / 1e6:.2f} MB each = "
          f"{arena.slots * arena.bytes_per_slot / 1e9:.2f} GB")

    # ---------------------------------------------------------------- 1/2. compile and match
    print("\n[1] does it compile and match? (T=6, decode-sized routing)\n")
    x = (torch.randn(6, M.DIM, generator=gen, device=dev) * 0.5).to(torch.float16).contiguous()
    slots, w = routing(6, TOPK, args.slots, gen, dev)
    ref = S.moe_forward_reference_dtype(x, slots, w, arena, dtype=torch.float16)
    try:
        got = S.moe_forward_sm70(x, slots, w, arena)
        ok = True
        err = rel_err(got, ref)
    except Exception as e:  # noqa: BLE001 - the point of the test is to see the failure
        ok, err, got = False, float("nan"), None
        print(f"  FAIL  the kernel raised: {type(e).__name__}: {str(e)[:400]}")
        FAILS.append("kernel did not compile or run")
    if ok:
        check("fp16 tl.dot works on this SM and the kernel runs", True)
        check(f"matches the fp16 reference (tol {args.tol:g})", err < args.tol, f"rel {err:.3e}")
        check("output is fp16 and correctly shaped",
              got.dtype == torch.float16 and tuple(got.shape) == (6, M.DIM), f"{got.dtype} {tuple(got.shape)}")

    # ---------------------------------------------------------------- 1b. both inner loops
    print("\n[1b] both inner loops, and the guard that picks between them\n")
    if ok:
        got_g = S.moe_forward_sm70(x, slots, w, arena, fold_scale=False)
        e = rel_err(got_g, ref)
        check("the grouped path (scale on the accumulator) also matches", e < args.tol, f"rel {e:.3e}")
        check("this arena may fold (all scales <= 140)", S.arena_fold_ok(arena),
              f"max scale byte {int(max(int(arena.s1.max()), int(arena.s3.max()), int(arena.s2.max())))}")
        # An arena whose scales are past the fold's bound. The input is scaled down so the check is
        # about the code path, not about fp16's range: at 2**14 the output of the grouped path is
        # fine in fp32 and would overflow fp16, which is the engine's contract rather than the
        # guard's business.
        big = build_arena(2, dev, gen, scale_lo=141, scale_hi=145)
        check("an arena with scales above 140 refuses to fold", not S.arena_fold_ok(big))
        bs = torch.zeros((1, TOPK), dtype=torch.int32, device=dev)
        bw = torch.full((1, TOPK), 1.0 / TOPK, device=dev)
        xb = (x[:1] * 0.01).contiguous()
        bref = S.moe_forward_reference_dtype(xb, bs, bw, big, dtype=torch.float16)
        bgot = S.moe_forward_sm70(xb, bs, bw, big)
        e = rel_err(bgot, bref)
        check("the guard's fallback is correct on an unfoldable arena", e < args.tol, f"rel {e:.3e}")

    # ---------------------------------------------------------------- 3. the fp64 oracle
    print("\n[2] one expert against an fp64 oracle (the strongest check available here)\n")
    x1 = (torch.randn(1, M.DIM, generator=gen, device=dev) * 0.5).to(torch.float16).contiguous()
    slots1 = torch.zeros((1, TOPK), dtype=torch.int32, device=dev)
    w1_ = torch.full((1, TOPK), 1.0 / TOPK, device=dev)
    oracle = S.moe_forward_reference_dtype(x1, slots1, w1_, arena, oracle=torch.float64)
    if ok:
        got1 = S.moe_forward_sm70(x1, slots1, w1_, arena)
        e = rel_err(got1, oracle)
        # fp16 weights and an fp16 down GEMM over K=2304: ~1e-3 is the arithmetic, not a bug. The
        # point of comparing against fp64 is to catch a systematic error, which would be orders
        # of magnitude larger.
        check("kept expert vs fp64 oracle", e < 1e-2, f"rel {e:.3e}")
    print(f"       (the fp16 reference itself sits {rel_err(ref[0:1], oracle):.3e} from the oracle)")

    # ---------------------------------------------------------------- 4. the sweep
    print("\n[3] launch config sweep (the GB10 numbers do not transfer)\n")
    print(f"  {'BM':>3} {'BN':>4} {'warps':>6} {'stages':>7} {'ms':>9} {'GB/s':>7} {'rel':>10}")
    best = None
    for T in (6, 32):
        xs = (torch.randn(T, M.DIM, generator=gen, device=dev) * 0.5).to(torch.float16).contiguous()
        sl, ww = routing(T, TOPK, args.slots, gen, dev)
        r16 = S.moe_forward_reference_dtype(xs, sl, ww, arena, dtype=torch.float16)
        BM = S._pick_bm(T * TOPK)
        n_distinct = int(torch.unique(sl).numel())
        nbytes = n_distinct * arena.bytes_per_slot
        for bn in (64, 128):
            for nw in (4, 8):
                for ns in (1, 2):
                    cfg = (bn, nw, ns)
                    try:
                        y = S.moe_forward_sm70(xs, sl, ww, arena, block_m=BM, up_cfg=cfg, down_cfg=cfg)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {BM:>3} {bn:>4} {nw:>6} {ns:>7} {'--':>9} {'--':>7}   {type(e).__name__}: {str(e)[:60]}")
                        continue
                    e = rel_err(y, r16)
                    ms = timed(lambda: S.moe_forward_sm70(xs, sl, ww, arena, block_m=BM, up_cfg=cfg, down_cfg=cfg), args.reps) * 1e3
                    gbs = nbytes / (ms / 1e3) / 1e9
                    print(f"  {BM:>3} {bn:>4} {nw:>6} {ns:>7} {ms:>9.3f} {gbs:>7.1f} {e:>10.2e}  (T={T}, {n_distinct} experts)")
                    if e < args.tol and (best is None or gbs > best[0]):
                        best = (gbs, cfg, BM, T, ms)
    if best:
        gbs, cfg, BM, T, ms = best
        print(f"\n  best: BN={cfg[0]} warps={cfg[1]} stages={cfg[2]} at BM={BM} (T={T}) "
              f"-> {ms:.3f} ms, {gbs:.1f} GB/s")
        print(f"  that GB/s is what tools/tier_plan.py wants for --gpu-gbs")
    else:
        FAILS.append("no launch config both matched and ran")

    print(f"\n  peak GPU memory during this test: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    print(f"  free VRAM now: {torch.cuda.mem_get_info()[0] / 1e6:.0f} MB")
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
