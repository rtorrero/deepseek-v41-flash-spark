"""DSV41_PREFILL_ATTN_GEMM -- the math type of prefill's two attention GEMMs.

What this is about. A prefill chunk's single largest kernel cost on this box is not a quantised
kernel at all: it is two *fp32* GEMMs inside the attention softmax, and fp32 on this hardware means
the SIMT pipeline, not the tensor cores. The profiler run behind
[`docs/gemm-dispatch.md`](../docs/gemm-dispatch.md) put them at

    cutlass_80_simt_sgemm_128x32_8x5_tn_align1   559.5 ms over 672 calls   (the score product)
    cutlass_80_simt_sgemm_128x32_8x5_nn_align1   295.7 ms over 672 calls   (the PV product)

for one 2,048-token chunk -- 855 ms together, more than the FP4 MoE kernels the chunk exists to
run (`_moe_up` 483 ms + `_moe_down` 410 ms). The call counts say exactly which matmuls they are:
`engine/model.py::Model._softmax_attn` runs its query tiles at `ATTN_TILE = 64` rows, so a
2,048-token chunk is 32 tiles, each tile issues one score einsum and one PV einsum, and the
encoder is 21 layers -- 32 x 21 = 672 of each, 1,344 together, and nothing else in the prefill
path has that count.

The modes.

  fp32            -- what the engine did until 0.6.0: `q` and the gathered KV are widened to fp32
                     per tile and both products run in fp32. Byte-for-byte the shipped engine;
                     this module writes nothing and touches no torch flag.
  tf32            -- gated 4 strict / 9 finished / no hard miss, +5 %; the same fp32 tensors, but `torch.backends.cuda.matmul.allow_tf32` is turned
                     on around the tile loop and off again after it. TF32 rounds the GEMM *inputs*
                     to an 11-bit significand and still accumulates in fp32 on the tensor cores,
                     so it is the "keep fp32 accumulate" path, at ~2^-11 = 4.9e-4 relative on the
                     operands and of the order of 1e-3 on the product.
  bf16  (default since 2026-09-15: two full gates, 6·9·1 and 8·10·0, plus 13 more runs with no hard miss; +21 %) -- the operands are never widened at all (`q` and the KV are bf16 in the cache),
                     the products run bf16-in/fp32-accumulate on the tensor cores, and the score
                     product is widened back to fp32 before the softmax, which therefore stays
                     exactly as accurate as it was. 8-bit significand, ~2^-8 = 3.9e-3 relative.
                     It is also the only mode that removes the per-tile fp32 widening of the
                     gathered KV -- [64, 640, 512] fp32 is 84 MB per tile, 2.7 GB per layer.

The risk, stated plainly. This engine's whole precision argument is chunk invariance: a row of a
GEMM must not depend on how many other rows were in the call, because a 1e-4 difference in the
residual stream flips a borderline router top-k and the MoE output then depends on the chunk
length. `Model._softmax_attn`'s own comment is about exactly this GEMM pair -- "rounding the
probabilities to bf16 first is a cliff that turns 1e-7 fp32 GEMM jitter into 1e-4 output jitter".
Both non-default modes are far above 1e-7, so both are expected to move routing somewhere in a
long prompt. That is not an argument against them; it is the reason they are off by default, the
reason the switch is prefill-only, and the reason `tools/verify_prefill_hc.sh` ends on
`tools/gate_profile.py` rather than on a tok/s number. A speed number alone cannot accept either
mode.

Prefill-only. `mode_for(prefill)` returns the default for every call that is not a prefill call,
so the decode path -- the captured graphs, the DSpark drafter, `engine/fastdecode.py` -- keeps
strict fp32 whatever the environment says. `math_scope` restores both torch settings in a
`finally`, so an exception inside a chunk cannot leave TF32 armed for the decode steps after it.

`DSV41_PREFILL_HC_GEMM` is accepted as an alias. The investigation started on the assumption that
these GEMMs belonged to the Hyper-Connection residual (`tools/v41_ref.py::hc_mixes`); the call
counts said attention instead, and the alias is kept so that name still reaches the right switch.
The HC mix GEMM is a real fp32 cost too, but a different one: M=16, N=24, K=20480 under the
`MM_TILE` row tiling, which is cuBLAS's `gemmSN_TN_kernel`, not a cutlass simt sgemm, and its cure
is the split-K Triton kernel decode already uses (`tools/fp32_skinny.py`), not a math type.

Values: fp32 (or unset / "off" / "default") | tf32 | bf16.
"""

from __future__ import annotations

import contextlib
import os

ENV_VAR = "DSV41_PREFILL_ATTN_GEMM"
ALIAS = "DSV41_PREFILL_HC_GEMM"
DEFAULT = "bf16"
DECODE = "fp32"   # decode never leaves fp32, whatever the prefill default becomes
MODES = ("fp32", "tf32", "bf16")


def parse(raw) -> str:
    """The env value -> one of MODES. Pure and torch-free, so the unit test covers every branch."""
    v = (raw or "").strip().lower()
    if v in ("", "off", "default", "none"):
        return DEFAULT
    if v not in MODES:
        raise ValueError(f"{ENV_VAR}: {v!r}; use {' / '.join(MODES)}")
    return v


def from_env(env=None) -> str:
    """The mode the environment asks for, reading ENV_VAR first and the alias only if it is unset."""
    env = os.environ if env is None else env
    raw = env.get(ENV_VAR)
    if raw is None or not raw.strip():
        raw = env.get(ALIAS)
    return parse(raw)


#: Read once at import. A harness that wants to A/B the modes inside ONE process -- which is the
#: only way to A/B anything on this box, because a second process lays the arena down at a
#: different offset and that alone moves a streaming kernel's time -- assigns this attribute
#: directly between runs. Nothing in the engine ever writes it.
MODE = from_env()


def mode_for(prefill: bool) -> str:
    """The math type an attention call should use. Never anything but fp32 off the prefill path."""
    return MODE if prefill else DECODE


@contextlib.contextmanager
def math_scope(mode: str):
    """Arm TF32 for the duration of the block, and put both settings back afterwards.

    For fp32 and bf16 this is an empty context manager: it imports nothing, reads nothing and
    writes nothing, so the default configuration issues exactly the calls it always did.

    `torch.set_float32_matmul_precision` and `torch.backends.cuda.matmul.allow_tf32` are two views
    of the same knob in current torch, so both are saved and both are restored -- the precision
    string first, because setting it rewrites `allow_tf32`, and `allow_tf32` has to have the last
    word.
    """
    if mode != "tf32":
        yield
        return
    import torch

    mm = torch.backends.cuda.matmul
    prev_allow = mm.allow_tf32
    prev_prec = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        mm.allow_tf32 = True
        yield
    finally:
        torch.set_float32_matmul_precision(prev_prec)
        mm.allow_tf32 = prev_allow
