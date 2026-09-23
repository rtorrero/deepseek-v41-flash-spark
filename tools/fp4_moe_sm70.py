"""fp4_moe_sm70.py -- the grouped-MoE kernels of tools/fp4_moe.py, ported to SM70 (V100).

Three changes, and only three:

 1. **The decode is integer arithmetic.** Upstream uses `cvt.rn.f16x2.e2m1x2`, which is
    sm_100a/120a/121a only; on V100 it does not assemble. `tools/fp4_decode.py` is the
    replacement and it is bit-exact against the reference (`tools/test_fp4_decode.py`, 15 checks),
    and it compiles and runs on the real card (`tools/fp4_decode_triton.py --check`, V100,
    capability 7.0, GEMV vs fp64 at 1.8e-07). This module reuses that decode rather than
    restating it, so there is one definition of the E2M1 grid in the tree.

 2. **fp16 instead of bf16.** Volta has fp16 tensor cores and no bf16 at all, so the activation
    tile, the stored `h` and the kernels' outputs are fp16. The accumulators stay fp32, which is
    what the upstream epilogue already does.

 3. **The launch configs are parameters, not the GB10 table.** `_UP_CFG`/`_DOWN_CFG` upstream were
    swept against 48 SMs and 228 KB of shared memory per SM; a V100 SM has 4 warp schedulers and
    64 KB per block. The defaults here are conservative guesses and
    `tools/test_fp4_moe_sm70.py` sweeps them.

Everything else is deliberately line-by-line parallel to the original -- the routing, the arena
layout, the K-permutation trick (`_split4`, `_quad_dot`), the clamps, the SwiGLU, the per-pair
output buffer that removes the atomics, the summation order -- so the two files can be diffed and
a reviewer can see that the port changed nothing it did not have to.

Not ported, and deliberately: `tools/cb3_moe.py` (the 3-bit codebook path) uses `prmt.b32`, which
is sm_20+ and therefore fine on Volta, but its bf16 sites still need the same sweep. It is a
separate change.
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fp4_decode as D  # noqa: E402
import fp4_moe  # noqa: E402  (the palace: arena layout, routing, DIM/INTER, the reference)
from fp4_decode_triton import _decode_even_odd  # noqa: E402  (the SM70-safe decode)
from fp4_moe import (  # noqa: E402
    DIM, INTER, ExpertArena, build_routing, build_routing_small,
)

# --------------------------------------------------------------------------- device helpers
@triton.jit
def _ue8m0(scale_u8):
    """UE8M0 byte -> fp32 2**(b-127), by building the float bit pattern. Same trick as the
    upstream `_ue8m0` and as `fp4_decode.ue8m0_bits`, which the tests hold equal."""
    return (scale_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _split4(t, BN: tl.constexpr, W: tl.constexpr):
    """[BN, 4*W] register tile -> four [BN, W] column chunks, in order. Identical to upstream."""
    t = tl.reshape(t, [BN, 2, 2, W])
    t = tl.permute(t, [0, 3, 2, 1])
    a, b = tl.split(t)
    c0, c1 = tl.split(a)
    c2, c3 = tl.split(b)
    return c0, c1, c2, c3


@triton.jit
def _chunk_dot(x_base, xk, mask_m, packed, scale_u8):
    """One 32-wide K group: x[BM, 32] . w[BN, 32]^T, the packed FP4 chunk decoded in registers.

    The only difference from upstream's `_chunk_dot` is where the two fp16 tiles come from: the
    integer decode instead of the PTX instruction. The K-permutation trick is unchanged -- the
    even K elements and the odd ones are computed as separate dots and the activation is read with
    stride 2 to match, because a dot product is order-invariant along K.
    """
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    we, wo = _decode_even_odd(packed)
    p = tl.dot(xe, tl.trans(we))
    p = tl.dot(xo, tl.trans(wo), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _quad_dot(x_base, xk, mask_m, w_tile_ptr, s_ptr, BN: tl.constexpr):
    """Four consecutive K groups (128 logical K) from one 64-byte-wide packed tile. Same as
    upstream: the 64-byte row tile is what reaches the practical copy bandwidth on the original
    box, and there is no reason to change it before measuring."""
    packed = tl.load(w_tile_ptr)
    c0, c1, c2, c3 = _split4(packed, BN, 16)
    s = tl.load(s_ptr)
    sa, sb = tl.split(tl.permute(tl.reshape(s, [BN, 2, 2]), [0, 2, 1]))
    s0, s1 = tl.split(sa)
    s2, s3 = tl.split(sb)
    acc = _chunk_dot(x_base, xk, mask_m, c0, s0)
    acc += _chunk_dot(x_base + 32, xk, mask_m, c1, s1)
    acc += _chunk_dot(x_base + 64, xk, mask_m, c2, s2)
    acc += _chunk_dot(x_base + 96, xk, mask_m, c3, s3)
    return acc


# --------------------------------------------------------------------------- kernel 1: gate/up
@triton.jit
def _moe_up_kernel_sm70(
    x_ptr, w1_ptr, s1_ptr, w3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    KB: tl.constexpr = K // 2
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)

    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_j = tl.arange(0, 64)
    offs_q = tl.arange(0, 4)

    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    w1_tile = w1_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    w3_tile = w3_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    s1_tile = s1_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]
    s3_tile = s3_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]

    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        acc_g += _quad_dot(x_base + q * 128, xk, mask_m[:, None], w1_tile + q * 64, s1_tile + q * 4, BN)
        acc_u += _quad_dot(x_base + q * 128, xk, mask_m[:, None], w3_tile + q * 64, s3_tile + q * 4, BN)

    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    # fp16 here, not bf16: this is the one line of the epilogue the port had to change.
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.float16), mask=mask_m[:, None])


# --------------------------------------------------------------------------- kernel 2: down
@triton.jit
def _moe_down_kernel_sm70(
    h_ptr, w2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr,
):
    KB: tl.constexpr = K // 2
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)

    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_j = tl.arange(0, 64)
    offs_q = tl.arange(0, 4)

    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    w2_tile = w2_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    s2_tile = s2_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        acc += _quad_dot(h_base + q * 128, xk, mask_m[:, None], w2_tile + q * 64, s2_tile + q * 4, BN)

    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


def _pick_bm(P: int) -> int:
    if P <= 64:
        return 16
    if P <= 1024:
        return 32
    return 64


# Conservative first guesses for SM70: 4 warp schedulers, 64 KB per block. The test sweeps these
# rather than trusting them; upstream's table was measured on a different SM entirely.
_UP_CFG_SM70 = {16: (64, 4, 1), 32: (64, 4, 1), 64: (64, 4, 1)}
_DOWN_CFG_SM70 = {16: (64, 4, 1), 32: (64, 4, 1), 64: (64, 4, 1)}


def moe_forward_sm70(
    x: torch.Tensor,
    slots: torch.Tensor,
    weights: torch.Tensor,
    arena: ExpertArena,
    swiglu_limit: float = 10.0,
    block_m: int | None = None,
    up_cfg: tuple[int, int, int] | None = None,
    down_cfg: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """x fp16 [T, 5120], slots int32 [T, K], weights fp32 [T, K] -> fp16 [T, 5120].

    Same contract and same summation order as `fp4_moe.moe_forward`; the differences are the
    decode and the dtype.
    """
    assert x.dtype == torch.float16 and x.shape[1] == DIM and x.is_contiguous(), (
        f"expected contiguous fp16 x [T, {DIM}], got {x.dtype} {tuple(x.shape)}")
    T, K = slots.shape
    assert weights.shape == (T, K)
    P = T * K
    dev = x.device
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = up_cfg or _UP_CFG_SM70[BM]
    bn2, nw2, ns2 = down_cfg or _DOWN_CFG_SM70[BM]
    if P <= 64:
        block_slot, block_pair, NB = build_routing_small(slots, BM)
    else:
        block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()

    h = torch.empty((P, INTER), dtype=torch.float16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    _moe_up_kernel_sm70[(NB, INTER // bn1)](
        x, arena.w1, arena.s1, arena.w3, arena.s3, h,
        wgt, block_slot, block_pair,
        x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1,
    )
    _moe_down_kernel_sm70[(NB, DIM // bn2)](
        h, arena.w2, arena.s2, parts,
        block_slot, block_pair,
        h.stride(0), parts.stride(0),
        TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2,
    )
    return parts.view(K, T, DIM).sum(dim=0).to(torch.float16)


# --------------------------------------------------------------------------- reference
@torch.no_grad()
def moe_forward_reference_dtype(
    x: torch.Tensor,
    slots: torch.Tensor,
    weights: torch.Tensor,
    arena: ExpertArena,
    swiglu_limit: float = 10.0,
    dtype: torch.dtype = torch.float16,
    oracle: torch.dtype | None = None,
) -> torch.Tensor:
    """The same loop as `fp4_moe.moe_forward_reference`, with the working dtype as a parameter.

    `oracle=torch.float64` dequantises and multiplies in fp64, which is the check that actually
    says whether a mismatch is a port bug or fp16 rounding: on a V100 there is no bf16 to compare
    against, so the upstream reference cannot be the arbiter here.
    """
    T, K = slots.shape
    work = oracle or dtype
    y = torch.zeros((T, DIM), dtype=torch.float32, device=x.device)
    for s in torch.unique(slots).tolist():
        w1 = D.dequant_fp4(arena.w1[s], arena.s1[s], dtype=work)
        w2 = D.dequant_fp4(arena.w2[s], arena.s2[s], dtype=work)
        w3 = D.dequant_fp4(arena.w3[s], arena.s3[s], dtype=work)
        t_idx, k_idx = torch.nonzero(slots == s, as_tuple=True)
        xs = x[t_idx].to(work)
        gate = (xs @ w1.T).to(torch.float32)
        up = (xs @ w3.T).to(torch.float32)
        if swiglu_limit > 0:
            up = up.clamp(-swiglu_limit, swiglu_limit)
            gate = gate.clamp(max=swiglu_limit)
        hs = torch.nn.functional.silu(gate.float()) * up * weights[t_idx, k_idx].float()[:, None]
        y.index_add_(0, t_idx, (hs.to(work) @ w2.T).float())
    return y.to(oracle or dtype)
