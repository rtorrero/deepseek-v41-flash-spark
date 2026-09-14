"""Does the fused Sinkhorn agree with the tiled torch path at PREFILL row counts?

Run on a CUDA box: python3 tools/test_hc_sinkhorn_prefill.py

`engine/hc_sinkhorn.py` has been decode's Sinkhorn since the fast path existed, at 1-6 rows.
DSV41_PREFILL_FUSED_SINKHORN points prefill at the same kernel, where a call carries up to
DSV41_PREFILL_CHUNK rows instead. Two things have to hold and neither is provable by reading:

  * the kernel is correct at those row counts -- it is one program per row with a tail guard, so
    2,048 rows is 2,048 programs and nothing about the arithmetic changes, but a grid that large
    has never been launched here;
  * the two paths agree closely enough that routing the shipped prefill through the fused one is a
    speed change and not a model change. They are NOT expected to be bit-identical: 130 torch
    kernels and one Triton program do the same softmax and the same 20 alternating normalisations
    in a different order, and `tl.exp` is not `torch.exp`. What is checked is that the disagreement
    stays at the level the fp32 arithmetic itself is uncertain to, at every row count, and that it
    does not grow with the number of rows (which is what a cross-row bug would look like).

It also checks the property that makes the tiling unnecessary on this path: one program per row
means a row's result does not depend on how many rows were in the call or where it sat, so the
fused output is row-count- and row-offset-invariant by construction. The tiled path manufactures
the same property at MM_TILE = 16; the fused one has it for free, at any T.

Self-skips (exit 0) without torch or without CUDA, so it can sit in the ordinary sweep.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"skip: torch is not importable here ({e.__class__.__name__})")
    sys.exit(0)

if not torch.cuda.is_available():
    print("skip: no CUDA device; the fused Sinkhorn is a Triton kernel and there is nothing to run")
    sys.exit(0)

try:
    from engine.hc_sinkhorn import hc_split_sinkhorn as fused
except Exception as e:  # noqa: BLE001
    print(f"skip: engine/hc_sinkhorn.py did not import ({e.__class__.__name__}: {e})")
    sys.exit(0)

import v41_ref as R  # noqa: E402

HC = 4
ITERS = 20
EPS = 1e-6
TOL = 2e-5          # fp32, 20 normalisations deep; decode has run on this kernel all along
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


def inputs(n, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mixes = (torch.randn(n, (2 + HC) * HC, generator=g) * 2).cuda()
    scale = torch.tensor([0.7, 1.3, 0.9], device="cuda")
    base = torch.randn(HC * (2 + HC), generator=g).cuda()
    return mixes, scale, base


def tiled(mixes, scale, base, tile=16):
    """What `hc_mixes` does today: the torch port under MM_TILE row tiles."""
    old, R.MM_TILE = R.MM_TILE, tile
    try:
        return R.tiled_rows(lambda t: R.hc_split_sinkhorn(t, scale, base, HC, ITERS, EPS), mixes)
    finally:
        R.MM_TILE = old


print("=== the two paths, at the row counts a prefill chunk produces")
worst = {}
for n in (1, 6, 16, 17, 128, 512, 1024, 2048):
    mixes, scale, base = inputs(n)
    a = tiled(mixes, scale, base)
    b = fused(mixes, scale, base, HC, ITERS, EPS)
    for x, y, nm in zip(a, b, ("pre", "post", "comb")):
        d = float((x - y).abs().max())
        worst[nm] = max(worst.get(nm, 0.0), d)
        check(f"T={n:5d} {nm:4s} agrees to {TOL:.0e}", d <= TOL, f"max abs diff {d:.3e}")
print(f"    worst over every row count: " + "  ".join(f"{k} {v:.2e}" for k, v in worst.items()))

print("\n=== the disagreement does not grow with the number of rows")
mixes, scale, base = inputs(2048, seed=1)
small = max(float((x - y).abs().max()) for x, y in
            zip(tiled(mixes[:16], scale, base), fused(mixes[:16], scale, base, HC, ITERS, EPS)))
large = max(float((x - y).abs().max()) for x, y in
            zip(tiled(mixes, scale, base), fused(mixes, scale, base, HC, ITERS, EPS)))
check("2,048 rows are no worse than 16", large <= max(small * 4, TOL), f"16 rows {small:.3e}, 2048 rows {large:.3e}")

print("\n=== the fused path is row-count- and row-offset-invariant, which is why it needs no tiling")
whole = fused(mixes, scale, base, HC, ITERS, EPS)
for lo, hi in ((0, 16), (7, 23), (1000, 1064), (2047, 2048)):
    part = fused(mixes[lo:hi].contiguous(), scale, base, HC, ITERS, EPS)
    same = all(torch.equal(p, w[lo:hi]) for p, w in zip(part, whole))
    check(f"rows [{lo}:{hi}] alone are bit-identical to those rows of the whole call", same)

print("\n=== and the tiled path it replaces is only invariant because it is tiled")
t16 = tiled(mixes, scale, base, tile=16)
t16b = tiled(mixes[:1024], scale, base, tile=16)
check("the tiled path agrees with itself across chunk lengths (the property MM_TILE buys)",
      all(torch.equal(a[:1024], b) for a, b in zip(t16, t16b)))

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
