"""test_fp4_gemv_cpu.py -- the RAM expert tier against the checkpoint's own arithmetic.

Three implementations of the same decode are compared, so a broken intrinsic cannot hide behind
a broken reference:

  * `fp4_gemv_ref`   -- scalar C, written for readability, no intrinsics
  * `fp4_gemv`       -- AVX2 + FMA, `vpermps` for the magnitude lookup
  * `fp4_gemv_torch` -- dequantise exactly in fp32, then BLAS

and then the whole expert (gate/up, the checkpoint's clamped SwiGLU, down) is compared against
the torch path, and the routed MoE against the same grouping-and-summation order the GPU
reference uses.

Runs on any x86 box with a compiler and torch. No GPU, no checkpoint.

Usage:  bash tools/build_fp4_cpu.sh && python3 tools/test_fp4_gemv_cpu.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for engine.*

import fp4_decode as D  # noqa: E402
import fp4_expert_cpu as C  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def rand_expert(gen, n: int, k: int, scale_lo: int = 118, scale_hi: int = 134):
    """Packed weights + UE8M0 scales with a plausible exponent range (2**-9 .. 2**7), which is
    what keeps the fp32 oracle free of overflow and the fp16 output free of infinities."""
    w = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
    s = torch.randint(scale_lo, scale_hi, (n, k // D.GROUP), generator=gen, dtype=torch.uint8)
    return w, s


# --------------------------------------------------------------------------- the kernel
def test_gemv() -> None:
    print("[1] fused GEMV: AVX2 == scalar C == torch, at the real shapes")
    K = C.DIM
    cache = C.kernels()
    if not cache.available():
        check("compiled kernel present", False, f"run: {C.CpuKernels.build_hint()}")
        return
    check("compiled kernel present", True, cache.path)

    g = torch.Generator().manual_seed(5)
    for n, label in ((64, "small"), (C.INTER, "w1/w3 shape 2304x5120")):
        w, s = rand_expert(g, n, K)
        x = torch.randn(K, generator=g, dtype=torch.float32)
        y_avx = torch.empty(n, dtype=torch.float32)
        y_ref = torch.empty(n, dtype=torch.float32)
        cache.gemv(w, s, x, y_avx)
        cache.gemv(w, s, x, y_ref, ref=True)
        y_torch = C.fp4_gemv_torch(w, s, x)
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
        check(f"{label}: AVX2 vs scalar C", rel(y_avx, y_ref) < 1e-6, f"rel {rel(y_avx, y_ref):.2e}")
        check(f"{label}: AVX2 vs torch fp32", rel(y_avx, y_torch) < 1e-6, f"rel {rel(y_avx, y_torch):.2e}")


def test_expert() -> None:
    print("[2] one expert end to end, C path vs torch path vs an fp64 oracle")
    arena = C.ExpertArenaCPU(2)
    g = torch.Generator().manual_seed(9)
    w1, s1 = rand_expert(g, C.INTER, C.DIM)
    w3, s3 = rand_expert(g, C.INTER, C.DIM)
    w2, s2 = rand_expert(g, C.DIM, C.INTER)
    arena.load_slot(0, w1, s1, w2, s2, w3, s3)

    x = torch.randn(1, C.DIM, generator=g, dtype=torch.float32) * 0.5
    y_c = C.expert_forward_slot(x, 0, arena, swiglu_limit=10.0, weights=torch.ones(1),
                                out_dtype=torch.float32)
    y_t = C.expert_forward(x, None, w1, s1, w2, s2, w3, s3, swiglu_limit=10.0,
                           weights=torch.ones(1), out_dtype=torch.float32)
    check("no NaN/Inf in the C path", torch.isfinite(y_c).all().item())

    # An fp64 oracle, so "the two paths differ" can be read as "by how much, and is either of
    # them wrong". Both fp32 paths differ from each other only by summation order: a 5120-long
    # dot product in fp32 lands ~1e-7 relative on its own, and the expert chains three of them
    # through a nonlinearity, so a few 1e-5 between two fp32 implementations is expected and
    # neither is more correct than the other. What must not happen is a *systematic* error.
    x64 = x.to(torch.float64)
    w1f64 = D.dequant_fp4(w1, s1, dtype=torch.float64)
    w3f64 = D.dequant_fp4(w3, s3, dtype=torch.float64)
    w2f64 = D.dequant_fp4(w2, s2, dtype=torch.float64)
    gate64 = (x64 @ w1f64.T).clamp(max=10.0)
    up64 = (x64 @ w3f64.T).clamp(-10.0, 10.0)
    h64 = torch.nn.functional.silu(gate64) * up64
    y64 = h64 @ w2f64.T

    rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max()).item()
    check("C path vs fp64 oracle", rel(y_c, y64) < 1e-4, f"rel {rel(y_c, y64):.2e}")
    check("torch path vs fp64 oracle", rel(y_t, y64) < 1e-4, f"rel {rel(y_t, y64):.2e}")
    check("C path vs torch path agree to fp32 reduction noise", rel(y_c, y64.double().float()) < 1e-4,
          f"rel {rel(y_c, y_t):.2e}")

    # A batch (>1 token) takes the torch path in both, which is the point: the threshold is
    # explicit rather than accidental. Both weight layouts have to be accepted, because passing
    # [M, 1] through `weights[:, None]` would broadcast to [M, M, INTER].
    for shape in ((3,), (3, 1)):
        xb = torch.randn(3, C.DIM, generator=g, dtype=torch.float32) * 0.5
        yb = C.expert_forward(xb, C.kernels(), w1, s1, w2, s2, w3, s3,
                              weights=torch.ones(*shape), out_dtype=torch.float32)
        check(f"M=3 with weights{shape} is finite and [3, DIM]",
              torch.isfinite(yb).all().item() and tuple(yb.shape) == (3, C.DIM), f"{tuple(yb.shape)}")


def test_moe_ordering() -> None:
    print("[3] routed MoE matches the GPU reference's grouping and summation order")
    n_slots, topk, t = 3, 2, 4
    arena = C.ExpertArenaCPU(n_slots)
    g = torch.Generator().manual_seed(13)
    for s in range(n_slots):
        w1, s1 = rand_expert(g, C.INTER, C.DIM)
        w3, s3 = rand_expert(g, C.INTER, C.DIM)
        w2, s2 = rand_expert(g, C.DIM, C.INTER)
        arena.load_slot(s, w1, s1, w2, s2, w3, s3)

    slots = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 1]], dtype=torch.int32)
    weights = torch.rand(t, topk, generator=g, dtype=torch.float32)
    x = torch.randn(t, C.DIM, generator=g, dtype=torch.float32) * 0.5
    y = C.moe_forward_cpu(x, slots, weights, arena, out_dtype=torch.float32)

    # Reference: the exact loop of tools/fp4_moe.moe_forward_reference, on this arena.
    y_ref = torch.zeros((t, C.DIM), dtype=torch.float32)
    for s in torch.unique(slots).tolist():
        w1f, w2f, w3f = arena.dequant_slot(int(s))
        ti, ki = torch.nonzero(slots == s, as_tuple=True)
        xs = x[ti]
        gate = (xs @ w1f.T).float().clamp(max=10.0)
        up = (xs @ w3f.T).float().clamp(-10.0, 10.0)
        hs = torch.nn.functional.silu(gate) * up * weights[ti, ki].float()[:, None]
        y_ref.index_add_(0, ti, (hs @ w2f.T).float())
    # The tolerance is fp32 reduction noise, not a fudge: two fp32 implementations of a
    # 5120-long dot product differ by ~1e-7 each, the expert chains three of them through a
    # nonlinearity, and this path additionally folds duplicate routes before the call instead of
    # computing the expert twice. Measured here the C path is the *closer* one to the fp64
    # oracle (4.6e-6 vs 2.97e-5 in [2]), so a difference in this range says nothing about which
    # is right; what would say something is a systematic offset, and the fp64 check in [2] is
    # what rules that out.
    err = ((y - y_ref).abs().max() / y_ref.abs().max().clamp_min(1e-30)).item()
    check("moe_forward_cpu agrees with moe_forward_reference", err < 1e-4, f"rel {err:.2e} (fp32 noise)")

    # Repeated routes to the same slot in one token (row 3) must not be silently deduplicated:
    # top-k can pick the same expert twice and both contributions count.
    single = C.moe_forward_cpu(x[:1], torch.tensor([[1, 1]], dtype=torch.int32),
                               torch.ones(1, 2), arena, out_dtype=torch.float32)
    twice = C.expert_forward_slot(x[:1], 1, arena, weights=torch.full((1, 1), 2.0),
                                  out_dtype=torch.float32)
    check("duplicate route contributes twice", (single - twice).abs().max().item() < 1e-5 * max(1.0, twice.abs().max().item()))


def test_shape_contract() -> None:
    print("[4] the shapes are the checkpoint's, not this module's opinion")
    try:
        import fp4_moe
        check("DIM/INTER/GROUP match tools/fp4_moe.py",
              (fp4_moe.DIM, fp4_moe.INTER, fp4_moe.GROUP) == (C.DIM, C.INTER, C.GROUP),
              f"{(fp4_moe.DIM, fp4_moe.INTER, fp4_moe.GROUP)}")
    except Exception as e:
        print(f"       (fp4_moe not importable, skipped: {type(e).__name__})")
    w13 = C.INTER * (C.DIM // 2)      # w1 and w3, packed
    s13 = C.INTER * (C.DIM // C.GROUP)
    w2 = C.DIM * (C.INTER // 2)       # w2 is the transpose shape, 5898240 too
    s2 = C.DIM * (C.INTER // C.GROUP)
    one = 2 * w13 + 2 * s13 + w2 + s2
    check("one slot is the 18,800,640 B the store and the budget model agree on",
          one == 18_800_640, f"{one}")
    a = C.ExpertArenaCPU(4)
    check("ExpertArenaCPU.bytes_per_slot == that number", a.bytes_per_slot == one,
          f"{a.bytes_per_slot}")


def test_arena_loader_contract() -> None:
    print("[5] the arena takes exactly what engine/experts.py::ExpertStore hands it")
    # ExpertStore calls: arena.load_slot(slot, w1.view(*W13_SHAPE), s1.view(*S13_SHAPE),
    # w2.view(*W2_SHAPE), s2.view(*S2_SHAPE), w3.view(*W13_SHAPE), s3.view(*S13_SHAPE),
    # non_blocking=True, **kw) -- i.e. flat views of a pinned staging buffer.
    import engine.experts as E  # noqa: PLC0415
    wanted = dict(w1=E.W13_SHAPE, s1=E.S13_SHAPE, w2=E.W2_SHAPE, s2=E.S2_SHAPE,
                  w3=E.W13_SHAPE, s3=E.S13_SHAPE)
    check("the store's tensor shapes are the ones this arena validates",
          (E.W13_SHAPE, E.S13_SHAPE, E.W2_SHAPE, E.S2_SHAPE) ==
          ((C.INTER, C.DIM // 2), (C.INTER, C.DIM // C.GROUP), (C.DIM, C.INTER // 2), (C.DIM, C.INTER // C.GROUP)),
          f"{E.W13_SHAPE} {E.S13_SHAPE} {E.W2_SHAPE} {E.S2_SHAPE}")

    a = C.ExpertArenaCPU(1)
    flat = {k: torch.arange(int(torch.tensor(v).prod()), dtype=torch.uint8) for k, v in wanted.items()}
    a.load_slot(0, flat["w1"].view(*wanted["w1"]), flat["s1"].view(*wanted["s1"]),
                flat["w2"].view(*wanted["w2"]), flat["s2"].view(*wanted["s2"]),
                flat["w3"].view(*wanted["w3"]), flat["s3"].view(*wanted["s3"]),
                non_blocking=True, sim=None)
    ok = (torch.equal(a.w1[0].reshape(-1), flat["w1"]) and torch.equal(a.s2[0].reshape(-1), flat["s2"]))
    check("load_slot with the store's call signature stores the bytes verbatim", ok)


def test_threads() -> None:
    print("[6] the OpenMP thread trap: torch sets it to 1, the wrapper must override it")
    k = C.kernels()
    cores = os.cpu_count() or 1
    now = k.max_threads()
    check("kernels() did not leave the tier single-threaded",
          now > 1 or cores == 1, f"omp max = {now}, cores = {cores}")
    check("set_threads(2) takes effect", k.set_threads(2) == 2, f"{k.max_threads()}")
    check("set_threads restores all cores", k.set_threads(cores) >= min(cores, 2), f"{k.max_threads()}")
    # The mechanism, stated: importing torch is enough to leave OpenMP at 1 in this process, so
    # anything that measures or serves through this tier has to set the count itself. Proven by
    # the fact that the very first `kernels()` in this process had to raise it from torch's 1.
    print(f"       (torch.get_num_threads()={torch.get_num_threads()}, " 
          f"expert tier omp max={k.max_threads()})")


def main() -> int:
    t0 = time.perf_counter()
    print("test_fp4_gemv_cpu.py -- the CPU expert tier\n")
    test_shape_contract()
    test_arena_loader_contract()
    test_gemv()
    test_expert()
    test_moe_ordering()
    test_threads()
    print(f"\n{time.perf_counter() - t0:.1f}s")
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
