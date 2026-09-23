"""fp4_decode.py -- the FP4 E2M1 + UE8M0 decode, expressed portably.

Why this file exists
--------------------
`tools/fp4_moe.py` decodes the packed experts with one PTX instruction:

    cvt.rn.f16x2.e2m1x2 t0, b0     ; sm_100a / sm_120a / sm_121a only

That instruction is Blackwell-only. On V100 (SM70) it does not assemble, and there is no
FP4 hardware path at all, so the decode has to be done with integer arithmetic. This module
is the specification of that decode, written twice and held against each other:

  * `values_lut`        -- a 16-entry table lookup. This is what `v41_ref.dequant_fp4_packed`
                           already does and what the tests use as ground truth.
  * `values_bits`       -- the same value built out of the fp16 bit pattern with shifts and
                           selects, no table, no memory, no `cvt`. This is the form that can be
                           dropped into a Triton kernel for SM70, because it is pure integer
                           arithmetic (`tl.where` + shifts + `bitcast`).

The two must agree bit for bit on all 16 codes. `tools/test_fp4_decode.py` asserts that, and
also that both agree with the reference implementation on real checkpoint tensors.

The E2M1 grid
-------------
    code  0    1    2    3    4    5    6    7
    value 0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0

Read as a sign, a magnitude index `mag = code & 7`, and the fp16 fields that produce it:

    mag        0     1     2     3     4     5     6     7
    value      0.0   0.5   1.0   1.5   2.0   3.0   4.0   6.0
    exp field  --    14    15    15    16    16    17    17
    mant bit   --    0     0     1     0     1     0     1

Three things follow, and the first two bit me while writing this:

  * the grid is *not* linear in the code (`0.5 * code` is wrong from code 5 on), so a
    shift-and-add bitplane decomposition does not exist and the decode has to go through the
    exponent and mantissa fields;
  * the mantissa bit is **not** `mag & 1`. It is `mag & 1` *except for mag 1*: 0.5 is exactly
    2**-1, not 1.5 * 2**-1 = 0.75. Writing the naive form makes code 1 decode as 0.75 -- a
    model that loads, runs, and is quietly wrong on one sixteenth of its weights;
  * `mag == 0` is +0.0, and the exponent formula alone would make it 0.5.

So: `exp = 14 + (mag >> 1)`, `mant = (mag & 1) and (mag >= 3)`, zero masked to 0, sign from
code bit 3. Every value on the grid is exactly representable in fp16, so the two
implementations agree bit for bit: there is no rounding to reason about, only equality.

Packing order
-------------
Two values per byte along K, **low nibble = even K element**, matching
`convert.py:cast_e2m1fn_to_e4m3fn` and `v41_ref.dequant_fp4_packed`. The kernels never
interleave the two halves back: a dot product is order-invariant along K, so they decode
"all even elements" and "all odd elements" as two separate tiles and load the activation
with stride 2 to match (`_chunk_dot` in tools/fp4_moe.py). `decode_even_odd` below reproduces
that split exactly, so the kernel's contract can be tested without a GPU.

UE8M0 block scales
------------------
One byte per 32 consecutive K elements, value 2**(b-127), i.e. the byte *is* the biased
exponent of an fp32. `ue8m0_bits` builds it with one shift and a bitcast -- the same trick as
`_ue8m0` in the kernel -- instead of `exp2`, which keeps the exponent exact and avoids a
transcendental for every scale.
"""

from __future__ import annotations

import torch

# The E2M1 grid, indexed by code. Same order as tools/v41_ref.py::FP4_TABLE and
# tools/fp4_moe.py::FP4_TABLE; kept here so the tests do not depend on an import path.
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

GROUP = 32  # K elements per UE8M0 scale


# --------------------------------------------------------------------------- nibble split
def codes_from_packed(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> uint8 [..., K] codes, low nibble first (even K element).

    The returned tensor is *twice as wide along the last axis*: element 2i is the low nibble
    of byte i and element 2i+1 is the high nibble. Callers that want the kernel's even/odd
    split use `decode_even_odd` instead.
    """
    x = packed.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).flatten(-2)


def decode_even_odd(packed: torch.Tensor, dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 [..., K/2] -> (even, odd) fp16 tiles, each [..., K/2].

    This is the contract of `_fp4_decode` in tools/fp4_moe.py: the two nibbles of one byte come
    back as two separate tiles, in K order within each tile, and the dot product against an
    activation loaded with stride 2 reassembles the sum.
    """
    x = packed.view(torch.uint8)
    even = values_bits(x & 0x0F, dtype=dtype)
    odd = values_bits((x >> 4) & 0x0F, dtype=dtype)
    return even, odd


# --------------------------------------------------------------------------- the two decodes
def values_lut(codes: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Code -> value through the 16-entry table. The reference form."""
    return FP4_TABLE.to(device=codes.device)[codes.long()].to(dtype)


def values_bits(codes: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Code -> value by building the fp16 bit pattern. No table, no `cvt`, SM70-safe.

    Only integer ops and selects, so this is exactly what a Triton kernel for SM70 can do:

        mag  = c & 7
        half = (mag & 1) != 0                 ; upper half of the binade ...
        top  = (mag >> 1) != 0                ; ... except mag 1 (0.5 = 2**-1 exactly)
        bits = ((14 + (mag >> 1)) << 10) | ((half & top) << 9)
        bits = mag == 0 ? 0 : bits            ; +0.0, the formula would say 0.5
        bits = bits | ((c & 8) << 12)         ; code bit 3 is the sign, fp16 sign is bit 15
        value = bitcast(bits, fp16)

    Verified bit-exact against `values_lut` for all 16 codes in the tests. Returns fp16 by
    default because that is the only dtype the kernel's `tl.dot` can use on SM70; for fp32
    the caller widens afterwards.
    """
    c = codes.to(torch.int32)
    mag = c & 0x07
    sign = (c & 0x08) << 12  # code bit 3 is the sign; fp16 sign is bit 15
    exp = 14 + (mag >> 1)
    mant = ((mag & 1) & ((mag >> 1) != 0).to(torch.int32)) << 9
    bits = (exp << 10) | mant
    bits = torch.where(mag == 0, torch.zeros_like(bits), bits)  # +0.0, not 0.5
    bits = bits | sign
    # Canonicalise negative zero. Codes 0 and 8 are both the value 0, and OR-ing the sign onto
    # the zero magnitude yields -0.0 for code 8 -- numerically inert, but a different bit
    # pattern, and `v41_ref.dequant_fp4_packed` (which the checkpoint's own correctness runs
    # were checked against) produces +0.0. Canonicalising here keeps the portable decode
    # bit-identical to the reference instead of merely equal.
    bits = torch.where((bits & 0x7FFF) == 0, torch.zeros_like(bits), bits)
    out = bits.to(torch.int16).view(torch.float16)
    return out if dtype == torch.float16 else out.to(dtype)


def ue8m0_bits(scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """UE8M0 byte -> 2**(b-127) by building the fp32 bit pattern (b << 23). Exact.

    b=0 is the 2**-127 subnormal-ish corner and b=255 is NaN/inf in fp32; the checkpoint's
    values are ordinary exponents, and the kernel's `_ue8m0` does the same shift, so this
    matches it byte for byte.
    """
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32).to(dtype)


def dequant_fp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    group: int = GROUP,
    dtype: torch.dtype = torch.float16,
    bits: bool = True,
    scaled: bool = True,
) -> torch.Tensor:
    """Packed FP4 [N, K/2] + UE8M0 [N, K/group] -> [N, K] in `dtype`.

    Drop-in for `v41_ref.dequant_fp4_packed` with three differences that matter for the port:
    the output dtype is a parameter (fp16 on SM70, bf16 on Blackwell), `bits=True` uses the
    table-free decode, and `scaled=False` returns the raw E2M1 values with the block scale
    *not* applied.

    `scaled=False` is not a convenience: it is the layout the kernels actually consume, and the
    reason is overflow. An E2M1 value is at most 6.0, so folding a block scale of 2**k into the
    weight makes the largest weight 6 * 2**k -- which leaves fp16's 65504 behind at k = 14. The
    kernels never materialise that product: `_chunk_dot` computes the fp32 partial sum of the
    32-wide group and multiplies *it* by the group's scale, where fp32 has the range to hold
    any scale the format can express. Materialising scaled weights in fp16 is therefore only
    safe when the checkpoint's scale bytes are known to be small, and this module does not
    assume that. Use fp32 for materialised weights, or `scaled=False` in fp16.
    """
    codes = codes_from_packed(weight)  # [N, K]
    vals = values_bits(codes, dtype=dtype) if bits else values_lut(codes, dtype=dtype)
    if not scaled:
        return vals
    s = ue8m0_bits(scale, dtype=dtype).repeat_interleave(group, dim=1)  # [N, K]
    if s.shape[-1] != vals.shape[-1]:
        raise ValueError(f"scale covers {s.shape[-1]} K elements but the weights have {vals.shape[-1]}")
    return vals * s
