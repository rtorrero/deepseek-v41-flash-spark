"""fp4_decode_triton.py -- the SM70-safe FP4 decode, as a Triton kernel, with a check you can run.

The upstream kernels decode with `cvt.rn.f16x2.e2m1x2` (tools/fp4_moe.py), which is
sm_100a/120a/121a only. On V100 it does not assemble. `tools/fp4_decode.py` is the portable
replacement, held bit-exact against the reference on CPU; this file is the same arithmetic
expressed for Triton, plus one deliberately simple kernel to prove it on the actual GPU.

The proof kernel is a plain GEMV and it uses `tl.sum`, not `tl.dot`. That is on purpose:

  * if the decode is wrong, this kernel says so, and nothing else is in the way;
  * it does not depend on fp16 `tl.dot` working on sm70, which is a separate question that the
    real MoE kernel will answer on its own;
  * it is short enough to read.

So the order of validation on a V100 is:

    1. python3 tools/fp4_decode_triton.py --check      # does the decode work on this GPU?
    2. bash tools/build_fp4_cpu.sh && python3 tools/test_fp4_gemv_cpu.py   # no GPU needed
    3. (only then) the fp16/portable variant of tools/fp4_moe.py, which is the retuning work

What has NOT been done, stated plainly: none of this has been run on a GPU. This machine has no
NVIDIA driver and no nvcc (see docs/sm70-port.md), so the kernel here is written against the
documented behaviour of the intrinsics and the arithmetic that the CPU tests pin down, and the
`--check` path exists to be the first thing that runs on your box. If it fails, the failure is
informative: the two decodes are compared code by code, so a wrong bit is a wrong bit, not a
diffuse numerical complaint.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton is a hard dependency of the GPU path
    HAVE_TRITON = False
    triton = None
    tl = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if HAVE_TRITON:

    @triton.jit
    def _decode_codes(codes):
        """4-bit E2M1 codes (int32 tile) -> fp16 values, by building the fp16 bit pattern.

        The derivation is in tools/fp4_decode.py and each step is a test there:

            mag        0     1     2     3     4     5     6     7
            value      0.0   0.5   1.0   1.5   2.0   3.0   4.0   6.0
            exp field  --    14    15    15    16    16    17    17
            mant bit   --    0     0     1     0     1     0     1

        Note the mantissa bit is `mag & 1` *except for mag 1*: 0.5 is exactly 2**-1, not
        1.5 * 2**-1. Writing the naive form decodes code 1 as 0.75, which is one sixteenth of
        the weights being quietly wrong.
        """
        mag = codes & 7
        sign = (codes & 8) << 12  # code bit 3 -> fp16 bit 15
        e = 14 + (mag >> 1)
        mant = ((mag & 1) & ((mag >> 1) != 0).to(tl.int32)) << 9
        bits = (e << 10) | mant
        bits = tl.where(mag == 0, 0, bits)  # +0.0, not 0.5
        bits = bits | sign
        # Canonicalise -0.0 to +0.0 so this matches vl_ref/fp4_decode bit for bit.
        bits = tl.where((bits & 0x7FFF) == 0, 0, bits)
        # Build the signed 16-bit value explicitly rather than relying on the truncation
        # behaviour of a narrowing cast (PyTorch saturates int32->int16; Triton truncates).
        signed = tl.where(bits >= 0x8000, bits - 0x10000, bits)
        return signed.to(tl.int16).to(tl.float16, bitcast=True)

    @triton.jit
    def _decode_even_odd(packed):
        """uint8 tile [..., 16] -> (even, odd) fp16 tiles, the contract of fp4_moe::_fp4_decode.

        One byte is two K elements, low nibble first. The kernels never interleave them back:
        a dot product is order-invariant along K, so the even values and the odd values are
        returned separately and the activation is read with stride 2 to match.
        """
        p = packed.to(tl.int32)
        return _decode_codes(p & 0x0F), _decode_codes((p >> 4) & 0x0F)

    @triton.jit
    def _decode_probe_kernel(packed_ptr, out_ptr, N: tl.constexpr):
        """Loads N packed bytes, decodes them, and writes the 2N values back in K order.

        This exists because `_decode_even_odd` is an inline helper, not a kernel: in Triton a
        kernel's tensor parameter arrives as a *pointer*, so calling the helper directly as a
        kernel fails with "cannot cast pointer<uint8>[] to int32" (learned on the first real run
        on a V100). The helper is right as written -- a kernel loads the tile and then calls it --
        so the harness needed a kernel around it, which is what this is.
        """
        offs = tl.arange(0, N)
        packed = tl.load(packed_ptr + offs)
        ev, od = _decode_even_odd(packed)
        tl.store(out_ptr + 2 * offs, ev)
        tl.store(out_ptr + 2 * offs + 1, od)

    @triton.jit
    def _gemv_sm70_kernel(w_ptr, s_ptr, x_ptr, y_ptr, N, NG, BN: tl.constexpr):
        """y[n] = sum over K groups of scale[n,g] * sum_k x[k] * value(code(w[n,k])).

        `tl.sum` over a decoded 32-wide group, so the kernel needs nothing from sm70 beyond
        fp16 load/store and fp32 arithmetic. One program handles BN rows.
        """
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        rmask = rows < N
        offs16 = tl.arange(0, 16)
        acc = tl.zeros([BN], dtype=tl.float32)
        for g in range(0, NG):
            packed = tl.load(w_ptr + rows[:, None] * (NG * 16) + g * 16 + offs16[None, :],
                             mask=rmask[:, None], other=0)
            ev, od = _decode_even_odd(packed)  # [BN, 16] each
            xe = tl.load(x_ptr + g * 32 + 2 * offs16, mask=(2 * offs16) < 32, other=0.0)
            xo = tl.load(x_ptr + g * 32 + 2 * offs16 + 1, mask=(2 * offs16 + 1) < 32, other=0.0)
            part = tl.sum(ev * xe[None, :], axis=1) + tl.sum(od * xo[None, :], axis=1)
            sb = tl.load(s_ptr + rows * NG + g, mask=rmask, other=127).to(tl.int32) << 23
            acc += part * sb.to(tl.float32, bitcast=True)
        tl.store(y_ptr + rows, acc, mask=rmask)


def fp4_gemv_triton(w: torch.Tensor, s: torch.Tensor, x: torch.Tensor, bn: int = 16) -> torch.Tensor:
    """Reference-shaped GEMV through the portable decode. w [N, K/2] uint8, s [N, K/32] uint8."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not importable")
    n, kb = w.shape
    k = kb * 2
    ng = s.shape[1]
    assert ng == k // 32, (ng, k)
    y = torch.empty(n, dtype=torch.float32, device=w.device)
    _gemv_sm70_kernel[(triton.cdiv(n, bn),)](
        w.contiguous(), s.contiguous(), x.contiguous(), y, n, ng, BN=bn, num_warps=4)
    return y


# --------------------------------------------------------------------------- the check
def check(rows: int = 512, k: int = 1024, seed: int = 4, device: str = "cuda") -> int:
    """Decode + GEMV on the GPU against the CPU-validated reference, code by code first.

    Two stages, so a failure says which half is wrong:

      1. the decode itself, compared against `fp4_decode.values_bits` on all 16 codes -- this is
         the piece that `cvt.rn.f16x2.e2m1x2` used to do, and the piece that must be identical;
      2. the GEMV, against a torch fp32 matmul over the same bytes.
    """
    import fp4_decode as D

    if not torch.cuda.is_available():
        print("no CUDA device: this check exists to run on the V100, and it needs the GPU.")
        return 2
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"device {name}, capability {cap[0]}.{cap[1]}")
    if cap[0] < 8:
        print("  sm < 80: bf16 has no hardware path here, so the ported kernels must use fp16.")
    dev = torch.device(device)

    # 1. the decode, every code, through the same path the kernel uses.
    #    Eight packed bytes carry sixteen codes: byte i is code i in the low nibble and code i+8 in
    #    the high one, so all sixteen values are covered by one tile.
    packed = torch.tensor([i | ((i + 8) << 4) for i in range(8)], dtype=torch.uint8, device=dev)
    codes = D.codes_from_packed(packed.cpu())  # [16], in K order
    want = D.values_bits(codes, dtype=torch.float16)
    if HAVE_TRITON:
        out = torch.empty(16, dtype=torch.float16, device=dev)
        _decode_probe_kernel[(1,)](packed, out, N=8)
        got = out.cpu()
        ok = torch.equal(got, want)
        print(f"  {'PASS' if ok else 'FAIL'}  decode matches the reference on all 16 codes")
        if not ok:
            print(f"        got  {got.tolist()}")
            print(f"        want {want.tolist()}")
            return 1
        print(f"        codes -> {got.tolist()}")

    # 2. the GEMV.
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (rows, k // 2), generator=g, dtype=torch.uint8, device=dev)
    s = torch.randint(110, 145, (rows, k // 32), generator=g, dtype=torch.uint8, device=dev)
    x = torch.randn(k, generator=g).to(dev)
    y_tri = fp4_gemv_triton(w, s, x)
    wf = D.dequant_fp4(w.cpu(), s.cpu(), dtype=torch.float64).to(dev)
    y_ref = (wf @ x.double()).float()
    rel = ((y_tri - y_ref).abs().max() / y_ref.abs().max()).item()
    ok = rel < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}  GEMV vs fp64 over the same bytes  (rel {rel:.2e})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the GPU check (needs an NVIDIA GPU)")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--k", type=int, default=1024)
    args = ap.parse_args()
    if not HAVE_TRITON:
        print("triton is not importable: install it before using the GPU path")
        return 2
    if args.check:
        return check(rows=args.rows, k=args.k)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
