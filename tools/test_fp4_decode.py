"""test_fp4_decode.py -- the decode contract that the SM70 port depends on.

No GPU, no checkpoint, no CUDA: this runs anywhere torch does, which is the point. The
Blackwell instruction the kernels currently use (`cvt.rn.f16x2.e2m1x2`) cannot run on V100,
so the replacement has to be provably identical *before* anyone waits for a GPU to tell them.

Checks, in order:

 1. `values_bits` == `values_lut` == the E2M1 grid, for all 16 codes, in fp16 and fp32.
 2. `codes_from_packed` agrees with the packing order the reference uses (low nibble = even K).
 3. `decode_even_odd` is the same values, split the way `_chunk_dot` consumes them.
 4. The K-permutation trick the kernels rely on: a dot product assembled from the even and
    odd tiles with a stride-2 activation equals the plain dot product over the interleaved
    weights. This is the one non-obvious contract in tools/fp4_moe.py, and on a machine with
    no GPU it is otherwise untestable.
 5. `ue8m0_bits` == `exp2(b - 127)` exactly, for every byte except the two corners b=0 and
    b=255, which are called out rather than papered over.
 6. `dequant_fp4` against an fp64 computation from the raw codes and scale bytes, and against
    `v41_ref.dequant_fp4_packed` on the same bytes.

Usage:  python3 tools/test_fp4_decode.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fp4_decode as D  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------------------- 1. the grid
def test_grid() -> None:
    print("[1] decode agrees with the E2M1 grid, all 16 codes")
    codes = torch.arange(16, dtype=torch.uint8)
    lut16 = D.values_lut(codes, dtype=torch.float16)
    bit16 = D.values_bits(codes, dtype=torch.float16)
    check("values_lut == FP4_TABLE (fp16)", torch.equal(lut16.float(), D.FP4_TABLE), f"{lut16.tolist()}")
    check("values_bits == values_lut, bit for bit (fp16)", torch.equal(bit16.view(torch.int16), lut16.view(torch.int16)))
    check("values_bits == values_lut (fp32)", torch.equal(D.values_bits(codes, dtype=torch.float32), D.FP4_TABLE))
    # The sign bit is code bit 3 and fp16 bit 15; get that wrong and the negatives become
    # large positives, which is a plausible-looking but catastrophically wrong model.
    neg = D.values_bits(codes, dtype=torch.float32)[8:]
    check("codes 8..15 are the negated magnitudes", torch.equal(neg, -D.FP4_TABLE[:8]))


# --------------------------------------------------------------------------- 2. packing order
def test_packing() -> None:
    print("[2] low nibble is the even K element")
    packed = torch.tensor([[0x1F, 0x08, 0xA5]], dtype=torch.uint8)  # low, high per byte
    codes = D.codes_from_packed(packed)
    check("codes_from_packed interleaves low then high", codes.tolist() == [[0x0F, 0x01, 0x08, 0x00, 0x05, 0x0A]], f"{codes.tolist()}")
    ev, od = D.decode_even_odd(packed)
    # packed = [[0x1F, 0x08, 0xA5]]: even nibbles 0xF,0x8,0x5 = -6.0,0.0,3.0; odd 0x1,0x0,0xA = 0.5,0.0,-1.0
    check("decode_even_odd returns (even, odd) halves as values",
          ev.tolist() == [[-6.0, 0.0, 3.0]] and od.tolist() == [[0.5, 0.0, -1.0]], f"{ev.tolist()} {od.tolist()}")

    ref = None
    try:
        import v41_ref
        ref = v41_ref.dequant_fp4_packed
    except Exception as e:  # pragma: no cover - v41_ref is in the same directory
        print(f"       (v41_ref not importable: {e})")
    if ref is not None:
        g = torch.Generator().manual_seed(7)
        w = torch.randint(0, 256, (64, 128), generator=g, dtype=torch.uint8)
        s = torch.randint(100, 160, (64, 8), generator=g, dtype=torch.uint8)
        mine = D.dequant_fp4(w, s, dtype=torch.float32)
        theirs = ref(w, s).float()
        check("dequant_fp4 == v41_ref.dequant_fp4_packed", torch.equal(mine.to(torch.bfloat16).float(), theirs), "")


# --------------------------------------------------------------------------- 3/4. kernel contract
def test_even_odd_dot() -> None:
    print("[3/4] the even/odd split plus a per-group accumulator scale == the real dot product")
    g = torch.Generator().manual_seed(11)
    N, M = 32, 5
    NG = 2  # K groups; K = NG * GROUP, two scale columns
    K = NG * D.GROUP
    packed = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.uint8)
    scale = torch.full((N, NG), 127 + 3, dtype=torch.uint8)  # 2**3, exact in fp16 and fp32
    w = D.dequant_fp4(packed, scale, dtype=torch.float32)  # [N, K], exact
    x = torch.randn(M, K, generator=g)
    ref = x @ w.T

    # Now the kernel's exact shape: decode each 32-wide group's even/odd halves, load the
    # activation with stride 2 inside the group, and apply the group's scale to the fp32
    # partial (tools/fp4_moe.py::_chunk_dot). This is the contract that has to survive the
    # port, and on a box with no GPU this is the only place it gets checked.
    ev, od = D.decode_even_odd(packed)  # [N, K/2] each, in K order within each half
    acc = torch.zeros(M, N)
    for gi in range(NG):
        sl = slice(gi * (D.GROUP // 2), (gi + 1) * (D.GROUP // 2))
        xe = x[:, gi * D.GROUP + 0 :: 2][:, : D.GROUP // 2]
        xo = x[:, gi * D.GROUP + 1 :: 2][:, : D.GROUP // 2]
        s = D.ue8m0_bits(scale[:, gi])[None, :]
        acc = acc + (xe @ ev[:, sl].float().T + xo @ od[:, sl].float().T) * s
    err = (acc - ref).abs().max().item()
    check("grouped even+odd dot == interleaved dot", err < 1e-3, f"max abs err {err:.2e}")


# --------------------------------------------------------------------------- 5. scales
def test_scales() -> None:
    print("[5] UE8M0 byte -> 2**(b-127)")
    b = torch.arange(256, dtype=torch.uint8)
    want = torch.exp2(b.to(torch.float32) - 127.0)
    got = D.ue8m0_bits(b)
    mid = torch.arange(1, 255, dtype=torch.uint8)
    check("exact for every ordinary exponent (1..254)",
          torch.equal(got[mid.long()], want[mid.long()]), f"{got[mid.long()][:3].tolist()} ...")
    # Corners, stated instead of hidden: the bit trick and exp2 disagree at the two ends of
    # the byte range. Both are degenerate scales (0 vs 2**-127, and inf vs inf) and the
    # kernel's `_ue8m0` does the shift, so `ue8m0_bits` is the behaviour to match.
    check("b=255 -> inf, both ways", torch.isinf(got[255]) and torch.isinf(want[255]))
    check("b=0 differs (0.0 by shift vs 2**-127 by exp2) -- documented, not a bug",
          got[0].item() == 0.0 and want[0].item() == 2.0**-127)


# --------------------------------------------------------------------------- 6. exactness
def test_exactness() -> None:
    print("[6] dequant_fp4 is exact against an fp64 computation from the raw bytes")
    g = torch.Generator().manual_seed(3)
    N, K = 48, 256
    packed = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.uint8)
    scale = torch.randint(110, 145, (N, K // D.GROUP), generator=g, dtype=torch.uint8)

    codes = D.codes_from_packed(packed)
    exact = D.FP4_TABLE.double()[codes.long()] * torch.exp2(
        scale.view(torch.uint8).double().repeat_interleave(D.GROUP, 1) - 127.0)
    mine = D.dequant_fp4(packed, scale, dtype=torch.float32)
    check("fp32 decode == fp64 ground truth", torch.equal(mine.double(), exact), f"max err {(mine.double() - exact).abs().max().item():.2e}")
    # fp16 is exact only for the *unscaled* values: all 16 E2M1 values are representable, and
    # no product has been formed yet. This is the form the kernel's tl.dot consumes.
    uns = D.dequant_fp4(packed, scale, dtype=torch.float16, scaled=False)
    check("fp16 decode is exact when the scale is left for the accumulator",
          torch.equal(uns.float().double(), D.FP4_TABLE.double()[codes.long()]))
    # And the hazard, demonstrated rather than described: a large scale times the largest
    # E2M1 value leaves fp16's range, which is why `scaled=True` defaults to a dtype the
    # caller has to think about and why the kernels scale the accumulator instead.
    big = torch.full((1, 1), 127 + 14, dtype=torch.uint8)
    sixes = torch.full((1, D.GROUP // 2), 0x77, dtype=torch.uint8)  # every nibble = 6.0
    overflow = D.dequant_fp4(sixes, big, dtype=torch.float16)  # 6.0 * 2**14 = 98304 > 65504
    check("scaled fp16 overflows at a scale of 2**14 (documented, not silent)",
          torch.isinf(overflow).all().item(), f"{overflow.tolist()[:3]} ...")
    # The layout the kernels index: column 0 is the code of byte 0's low nibble, times that
    # column's group scale.
    want0 = D.values_lut(codes[:, 0], dtype=torch.float32) * D.ue8m0_bits(scale[:, 0])
    check("column 0 is code 0 of byte 0 times group 0's scale", torch.equal(mine[:, 0], want0))


def main() -> int:
    print("test_fp4_decode.py -- portable FP4 E2M1 + UE8M0 decode\n")
    test_grid()
    test_packing()
    test_even_odd_dot()
    test_scales()
    test_exactness()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
