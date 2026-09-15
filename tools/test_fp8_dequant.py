"""Does the cheap fp8 dequant produce the same bf16 as the expensive one?

Run: python3 tools/test_fp8_dequant.py

The claim `engine/prefill_fp8.py` makes is not "close enough", it is BIT-IDENTICAL, so that is what
is tested: `dequant_views` (torch, a broadcast over a blocked view) and `dequant_fused` (one Triton
launch) against `FP8Weight.dequant()`, compared as raw bits and not as a norm. The argument behind
the claim is that e4m3 -> bf16 loses nothing (four significand bits into eight), the block scale is
a power of two so the fp32 product is exact wherever the shipped path's is, and the closing
round-to-nearest-even to bf16 is the same rounding.

Small tensors on the CPU, so this needs no GB10 and no checkpoint -- but it does need torch, and
the CPU half needs triton only because `tools/fp8_linear.py` imports it at module scope. Both are
absent on a hosted runner, so the script SKIPS there (exit 0) rather than failing. On a CUDA box it
additionally runs `dequant_fused` itself, which is the path the engine takes.

Shapes cover the five weights that are still fp8 under DSV41_DENSE_FP4=attn,wo_a -- and one whose
N is not a multiple of 32, which is the only case `dequant_views` handles with a second statement.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"skip: torch is not importable here ({e.__class__.__name__}); "
          f"this test needs it and says so rather than failing")
    sys.exit(0)

try:
    from fp8_linear import FP8Weight, dequant_fused, dequant_views
except Exception as e:  # noqa: BLE001
    print(f"skip: tools/fp8_linear.py did not import ({e.__class__.__name__}: {e}); "
          f"it needs triton, which a CPU runner does not have")
    sys.exit(0)

import v41_ref as R  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


def make(N, K, device, seed=0):
    """An FP8Weight with the checkpoint's layout: e4m3 codes and one UE8M0 scale per 32x32 block.

    The scales are spread over a wide exponent range on purpose -- a scale table of all 127s would
    make every path agree for the wrong reason.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = (torch.randn(N, K, generator=g) * 3).to(torch.float8_e4m3fn).to(device)
    s = torch.randint(100, 150, ((N + 31) // 32, (K + 31) // 32), generator=g,
                      dtype=torch.uint8).to(device)
    return FP8Weight(codes, s)


def bits(t):
    return t.contiguous().view(torch.int16)


SHAPES = (
    ("shared_experts.w1/w3", 2304, 5120),
    ("shared_experts.w2", 5120, 2304),
    ("indexer.wq_b", 4096, 1280),
    ("engram.wkv (a slice of it)", 2560, 6144),
    ("N not a multiple of 32", 100, 256),
    ("one block", 32, 32),
)

# The real shapes are up to 157 M weights; at fp32 intermediates that is several GB on a CPU, and
# this test is about bits, not about size. Cut every N down, keep K, keep the block structure.
CPU_ROWS = 128

print("=== CPU: dequant_views against FP8Weight.dequant()")
for name, N, K in SHAPES:
    n = min(N, CPU_ROWS) if N > CPU_ROWS else N
    if n % 32 and N % 32 == 0:
        n = (n // 32) * 32 or 32
    W = make(n, K, "cpu")
    ref = W.dequant()
    got = dequant_views(W)
    check(f"{name} [{n}, {K}]: same dtype and shape",
          got.dtype == ref.dtype and got.shape == ref.shape, f"{got.dtype} {tuple(got.shape)}")
    check(f"{name} [{n}, {K}]: bit-identical", bool(torch.equal(bits(got), bits(ref))),
          f"max abs diff {float((got.float() - ref.float()).abs().max()):.3e}")

print("\n=== CPU: dequant_fused falls back to the torch path off the device")
W = make(64, 256, "cpu")
check("dequant_fused runs on a CPU tensor and matches",
      bool(torch.equal(bits(dequant_fused(W)), bits(W.dequant()))))

print("\n=== CPU: the mode switch in v41_ref")
W = make(64, 256, "cpu")
ref = W.dequant()
try:
    R.PREFILL_FP8_MODE = "off"
    R.prefill_fp8_cache_clear()
    check("off returns the shipped dequant", bool(torch.equal(bits(R.prefill_dequant(W)), bits(ref))))
    check("off caches nothing", R.prefill_fp8_cache_bytes() == 0)

    R.PREFILL_FP8_MODE = "fused"
    a = R.prefill_dequant(W)
    b = R.prefill_dequant(W)
    check("fused returns the same bits", bool(torch.equal(bits(a), bits(ref))))
    check("fused caches nothing and re-does the work", R.prefill_fp8_cache_bytes() == 0
          and a is not b)

    R.PREFILL_FP8_MODE = "cached"
    a = R.prefill_dequant(W)
    b = R.prefill_dequant(W)
    check("cached returns the same bits", bool(torch.equal(bits(a), bits(ref))))
    check("cached hands back the SAME tensor the second time", a is b)
    check("cached holds exactly the bytes it produced",
          R.prefill_fp8_cache_bytes() == a.numel() * a.element_size(),
          str(R.prefill_fp8_cache_bytes()))
    R.prefill_fp8_cache_clear()
    check("clearing gives them back", R.prefill_fp8_cache_bytes() == 0)
finally:
    R.PREFILL_FP8_MODE = "fused"
    R.prefill_fp8_cache_clear()
check("the sweep leaves v41_ref back in its shipped state", R.PREFILL_FP8_MODE == "fused")

if not torch.cuda.is_available():
    print("\nskip: no CUDA here, so the Triton kernel itself was not run "
          "(the CPU half above covers the arithmetic)")
else:
    print("\n=== CUDA: the Triton kernel against FP8Weight.dequant()")
    for name, N, K in SHAPES:
        W = make(N, K, "cuda")
        ref = W.dequant()
        got = dequant_fused(W)
        check(f"{name} [{N}, {K}]: bit-identical on the device",
              bool(torch.equal(bits(got), bits(ref))),
              f"max abs diff {float((got.float() - ref.float()).abs().max()):.3e}")
    # A weight whose K is not a multiple of the kernel's 128-wide tile, to exercise the mask.
    W = make(64, 160, "cuda")
    check("K = 160 (not a multiple of BLOCK_K): bit-identical",
          bool(torch.equal(bits(dequant_fused(W)), bits(W.dequant()))))

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
