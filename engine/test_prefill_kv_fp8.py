"""Unit tests for the fp8 prefill gather (DSV41_PREFILL_KV_FP8, engine/model.py `_gather_kv_fp8`).

Four things are checked, in the order they can go wrong:

  1. the e4m3 round trip itself, on values shaped like the ones the engine feeds it -- the output of
     an rmsnorm, so O(1), which is why e4m3's range is never the binding constraint and its extra
     mantissa bit is what matters;
  2. `_gather_kv_fp8` agrees EXACTLY with quantising what the bf16 path gathers, at several chunk
     lengths and gather-tile sizes: the tiling is an allocation strategy and must not be visible in
     a single value;
  3. what the rounding does to an attention output once the softmax has averaged it over head_dim,
     which is the number that matters for quality;
  4. that the clamp holds: an outlier above 448 saturates instead of turning into a NaN that the
     softmax then spreads across the whole row.

Skips itself with exit 0 wherever there is no CUDA (or no torch, or a torch without float8), so it
is safe to run from a checkout on any machine.

Run:  python engine/test_prefill_kv_fp8.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"SKIP: no torch ({type(e).__name__})")
    sys.exit(0)

if not torch.cuda.is_available():
    print("SKIP: no CUDA device")
    sys.exit(0)
if not hasattr(torch, "float8_e4m3fn"):
    print(f"SKIP: torch {torch.__version__} has no float8_e4m3fn")
    sys.exit(0)

# The flag only selects the path inside `attention`; `_gather_kv_fp8` is called directly here, so
# the module is imported with the shipped default (off) and nothing about this file depends on it.
from engine.model import (FP8_KV_DTYPE, FP8_KV_MAX, PREFILL_KV_FP8,  # noqa: E402
                          RING, Model)
import v41_ref as R  # noqa: E402

DEV = "cuda"
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def quantise(x):
    """What the engine does, elementwise: saturate into range, then round to e4m3."""
    return x.clamp(-FP8_KV_MAX, FP8_KV_MAX).to(FP8_KV_DTYPE)


def bare_model(args):
    """A Model with only the two attributes the gather and the softmax touch. Instantiating the
    real one would load 19 GB of weights to test an index_select."""
    m = Model.__new__(Model)
    m.args = args
    m.dev = DEV
    m.tap = None
    return m


print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
      f"DSV41_PREFILL_KV_FP8={'1' if PREFILL_KV_FP8 else '0'} (irrelevant here)")
print()

args = R.Args()
d = args.head_dim
W = args.window_size
K = args.index_topk

# --- 1. the round trip -------------------------------------------------------
# e4m3 keeps 1 implicit + 3 explicit significand bits, so inside its normal range round-to-nearest
# is within half an ulp = 2^-4 of the value: a hard 6.25 % worst case, and far less in RMS. e5m2
# would be 2^-3 on the same values and buys only range, which an rmsnorm output does not need.
torch.manual_seed(0)
for name, x in (
    ("rmsnorm-like N(0,1)", torch.randn(64, 4096, device=DEV, dtype=torch.bfloat16)),
    ("with RoPE-scale tails", (torch.randn(64, 4096, device=DEV) * 4).to(torch.bfloat16)),
    ("small, near the subnormals", (torch.randn(64, 4096, device=DEV) * 0.01).to(torch.bfloat16)),
):
    ref = x.float()
    got = quantise(x).float()
    err = (got - ref).abs()
    live = ref.abs() >= 2.0 ** -6          # e4m3's smallest normal; below it the step is absolute
    rel = (err[live] / ref.abs()[live]).max().item()
    rms = float((got - ref).norm() / ref.norm())
    check(f"e4m3 round trip, {name}", rel <= 0.0625 + 1e-6 and rms <= 0.02,
          f"max rel {rel:.4f} (bound 0.0625), rms rel {rms:.4f}")
    check(f"  finite, {name}", bool(torch.isfinite(got).all()))

# --- 2. the gather is the bf16 gather, quantised -----------------------------
# `_gather_kv_fp8` fills one [T, W+K, d] fp8 buffer tile by tile instead of building two bf16
# gathers and concatenating them. Quantisation is elementwise, so the two have to be bit-identical,
# at every chunk length and whatever the tile.
m = bare_model(args)
import engine.model as M  # noqa: E402

for T in (256, 1024, 2048, 4096):
    if T + W > RING:
        print(f"skip chunk {T}: RING is {RING} (raise DSV41_RING to test it)")
        continue
    S = 4096
    ring = (torch.randn(RING, d, device=DEV) * 1.5).to(torch.bfloat16)
    n_c = (S + T) // 2
    ckv = (torch.randn(n_c + 8, d, device=DEV) * 1.5).to(torch.bfloat16)
    pos = torch.arange(S, S + T, device=DEV)
    wpos = m._window_positions(pos)
    cidx = torch.randint(-1, n_c, (T, K), device=DEV)

    class _Sh:
        pass
    sh = _Sh(); sh.ckv = ckv

    ref = quantise(torch.cat([ring[wpos.clamp_min(0) % RING], ckv[cidx.clamp_min(0)]], dim=1))
    for tile in (128, 512, T):
        old, M.KV_GATHER_TILE = M.KV_GATHER_TILE, tile
        try:
            got = m._gather_kv_fp8(ring, wpos, sh, cidx, T)
        finally:
            M.KV_GATHER_TILE = old
        same = bool(torch.equal(got.view(torch.uint8), ref.view(torch.uint8)))
        check(f"gather T={T} tile={tile} is the bf16 gather quantised", same)
        check(f"  shape and dtype T={T} tile={tile}",
              got.shape == (T, W + K, d) and got.dtype == FP8_KV_DTYPE, str(tuple(got.shape)))
    # the ratio-0 layers (0 and 1) have no compressed rows at all
    win_only = m._gather_kv_fp8(ring, wpos, sh, None, T)
    check(f"window-only gather T={T}",
          win_only.shape == (T, W, d)
          and bool(torch.equal(win_only.view(torch.uint8),
                               quantise(ring[wpos.clamp_min(0) % RING]).view(torch.uint8))))

# --- 3. what it costs an attention output ------------------------------------
# The rounding lands in the KV of a softmax that reduces over head_dim = 512 and then mixes n
# values, so the error at the output is a good deal smaller than the 6.25 % at the input. This is
# the number quality risk is argued from: it feeds the residual stream, and through it the window
# KV the decoder layers write and decode later reads.
T = 512
q = (torch.randn(T, args.n_heads, d, device=DEV) * 0.2).to(torch.bfloat16)
kv_bf16 = (torch.randn(T, W + K, d, device=DEV) * 1.5).to(torch.bfloat16)
mask = torch.rand(T, W + K, device=DEV) > 0.3
mask[:, 0] = True                                     # never an all-masked row here
sink = torch.zeros(args.n_heads, device=DEV)
o_ref = m._softmax_attn(q, kv_bf16, mask, sink).float()
o_fp8 = m._softmax_attn(q, quantise(kv_bf16), mask, sink).float()
rel = float((o_fp8 - o_ref).norm() / o_ref.norm())
check("attention output moves less than 1 % relative", rel <= 1e-2, f"rel {rel:.2e}")
check("  and stays finite", bool(torch.isfinite(o_fp8).all()))

# --- 4. the clamp, not a NaN -------------------------------------------------
# torch's cast to float8_e4m3fn has no infinity: anything above 448 becomes NaN, and one NaN in a
# gathered row turns that query's whole softmax into NaN. The clamp is what stops it.
big = torch.tensor([[-1e4, -449.0, -448.0, 0.0, 1.0, 448.0, 449.0, 1e4]],
                   device=DEV, dtype=torch.bfloat16)
qb = quantise(big).float()
check("outliers saturate instead of turning into NaN", bool(torch.isfinite(qb).all()), str(qb.tolist()))
check("  and saturate at +-448", float(qb.max()) == 448.0 and float(qb.min()) == -448.0)
raw = big.to(FP8_KV_DTYPE).float()
if not bool(torch.isfinite(raw).all()):
    print("     (confirmed: the unclamped cast produces NaN on this build -- the clamp is load-bearing)")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
