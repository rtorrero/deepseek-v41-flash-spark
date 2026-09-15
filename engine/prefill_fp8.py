"""DSV41_PREFILL_FP8_DEQUANT -- how an fp8 dense weight becomes bf16 at prefill row counts.

The problem, from docs/gemm-dispatch.md (rows 5 and 7 of the inventory). `tools/v41_ref.dense`
runs `_fp8_linear_kernel` only at `M <= 16`; above it the weight is dequantised to bf16 and handed
to cuBLAS, and the dequant is `FP8Weight.dequant()`:

    s = torch.exp2(self.s.float() - 127.0)
    s = s.repeat_interleave(32, 0)[: self.N].repeat_interleave(32, 1)[:, : self.K]
    return (self.w.float() * s).to(torch.bfloat16)

which materialises a full `[N, K]` fp32 scale table, a full `[N, K]` fp32 weight and the bf16
result -- ten transient bytes per weight -- allocated and thrown away on every call, per weight,
per layer, per chunk. The weights still on that path in the shipped configuration
(`DSV41_DENSE_FP4=attn,wo_a`) are `ffn.shared_experts.w1/w2/w3` on every layer, the indexer `wq_b`
on every index-source layer and the engram `wkv` on every engram layer.

    encoder pass (layers 0..20, DSV41_SWA_REPLAY=1), per 2,048-token chunk
        21 x shared w1/w3 [2304, 5120] + w2 [5120, 2304]        743,178,240 weights
         4 x indexer wq_b [4096, 1280]                           20,971,520
         2 x engram  wkv  [25600, 6144]                         314,572,800
                                                       total  1,078,722,560 weights
        x 10 transient bytes                                        10.79 GB per chunk

The measured rate of that shape is in the audit: 5.14 ms to dequantise a [4096, 1280] weight at
M = 2048, i.e. 0.98 ns per weight, because it is six allocator-bound elementwise kernels and not
one streaming copy. 1.079 G weights x 0.98 ns is **~1.06 s per chunk**, against ~5.55 s for a
2,048-token chunk at the measured 369 tok/s -- about a fifth of prefill spent turning a weight
into a copy of itself.

The three modes, and why `fused` is the one to use.

`fused` -- one Triton kernel (`tools/fp8_linear.dequant_fused`) reads the fp8 codes and the UE8M0
    scale table and writes bf16 directly: two transient bytes per weight instead of ten, three
    bytes of DRAM traffic instead of about twenty-seven, one launch instead of six, and no fp32
    intermediate at all. Bit-identical to `dequant()` by construction, not approximately: e4m3 ->
    fp32 is exact, the scale is a power of two, the fp32 product is exact, and the final
    round-to-nearest-even to bf16 is the same rounding the torch path does. Costs no memory.

`cached` -- `fused`, plus the bf16 result is kept for the rest of the request's prefill, so a
    weight is dequantised on the first chunk and read from the cache on every chunk after it. It
    buys the residual ~16 ms per chunk that `fused` still pays, and only from the second chunk on:
    a single-chunk prompt gets nothing and pays the memory anyway. The memory is `cache_bytes()`
    below -- 2.16 GB for the encoder pass, 3.70 GB once the decoder replay and the DSpark seed
    have touched their own weights -- which has to come out of the expert arena (~150 slots at
    2.16 GB) and out of `tools/budget.py`'s prefill reserve. That is why `fused` is the
    recommendation and `cached` is the arm to measure, not the default to assume.

`scaled_mm` -- **refused, with a reason.** The obvious "real fp8 GEMM" answer does not fit this
    checkpoint's scale layout. `torch._scaled_mm` takes per-tensor, per-row, 1x128/128x128
    block-wise or 1x32 (MX) scales. This checkpoint stores **one UE8M0 scale per 32x32 block** --
    see `FP8Weight.__init__`'s assert, `s.shape == ((N+31)//32, (K+31)//32)`. 32x32 is not one of
    the supported granularities, so reaching `_scaled_mm` at all would mean re-quantising the
    weight into a layout it accepts, which changes the numbers the recipe was measured on. The
    audit's own warning applies exactly here: a layout mismatch does not raise, it dispatches to
    something slow and correct. The mode name is kept so the refusal is discoverable rather than
    silent.

Fused is the default since 2026-09-15 (bit-identical to `w.dequant()` on the device,
`tools/test_fp8_dequant.py`, +3.3 % prefill); `off` is the path the engine used until then, and
off is byte-identical: `v41_ref.dense` calls `w.dequant()` exactly as it always did. `tools/test_prefill_fp8.py` pins
both halves of that. Torch-free, like `engine/prefill_topk.py`, so the parser and the memory
arithmetic are under test on a CPU runner.
"""

from __future__ import annotations

import os

ENV_VAR = "DSV41_PREFILL_FP8_DEQUANT"

#: Every spelling the variable accepts. "fused" is the shipped behaviour; "off" is the pre-0.6 path.
MODES = ("off", "fused", "cached", "scaled_mm")
DEFAULT = "fused"

_OFF = ("off", "0", "none")
_DEFAULT = ("", "default")

_SCALED_MM_REFUSAL = (
    "the checkpoint's scale table is one UE8M0 value per 32x32 block "
    "(FP8Weight: s.shape == ((N+31)//32, (K+31)//32)), and torch._scaled_mm takes per-tensor, "
    "per-row, 1x128/128x128 or 1x32 scales -- none of which that is. Reaching it would mean "
    "re-quantising the weight into a layout it accepts, which moves the numbers this recipe was "
    "measured on. Use 'fused'"
)


def parse(raw):
    """The env value -> one of MODES.

    Pure and torch-free so the unit test can exercise every branch on a CPU runner.
    """
    v = (raw or "").strip().lower()
    if v in _DEFAULT:
        return DEFAULT
    if v in _OFF:
        return "off"
    if v in ("fused", "cached"):
        return v
    if v == "scaled_mm":
        raise ValueError(f"{ENV_VAR}=scaled_mm is not implemented: {_SCALED_MM_REFUSAL}")
    raise ValueError(f"{ENV_VAR}: {v!r}; use {' / '.join(MODES)}, or leave it unset")


#: Read once, at import. Nothing in the engine ever writes it; a harness that wants to A/B the
#: modes inside ONE process (the only way to A/B anything on this box -- a second process lays the
#: arena down at a different offset and that alone moves a streaming kernel's time) assigns this
#: attribute directly between runs.
MODE = parse(os.environ.get(ENV_VAR))


def enabled() -> bool:
    """True when `dense` should take the new path at all."""
    return MODE != "off"


def cached() -> bool:
    """True when a dequantised weight should be kept for the rest of the prefill."""
    return MODE == "cached"


# --------------------------------------------------------------------------- the memory of `cached`
# Shapes are the checkpoint's (inference/config.json): dim 5120, moe_inter_dim 2304,
# q_lora_rank 1280, index_n_heads 32 x index_head_dim 128, engram rows [24, 256] -> hc*dim + dim.
# (name, N, K) of every weight that is still FP8Weight under DSV41_DENSE_FP4=attn,wo_a.
FP8_PREFILL_WEIGHTS = (
    ("ffn.shared_experts.w1", 2304, 5120),
    ("ffn.shared_experts.w3", 2304, 5120),
    ("ffn.shared_experts.w2", 5120, 2304),
)
INDEXER_WQ_B = ("attn.indexer.wq_b", 4096, 1280)
ENGRAM_WKV = ("engram.wkv", 25600, 6144)
MTP_MAIN_PROJ = ("mtp.0.main_proj", 5120, 15360)

BF16 = 2


def _weights(n_layers: int, index_layers, engram_layers, mtp: bool) -> int:
    """How many fp8 weight elements a pass over `n_layers` backbone layers would cache."""
    n = n_layers * sum(N * K for _, N, K in FP8_PREFILL_WEIGHTS)
    n += sum(1 for L in index_layers if L < n_layers) * INDEXER_WQ_B[1] * INDEXER_WQ_B[2]
    n += sum(1 for L in engram_layers if L < n_layers) * ENGRAM_WKV[1] * ENGRAM_WKV[2]
    if mtp:
        n += MTP_MAIN_PROJ[1] * MTP_MAIN_PROJ[2]
    return n


# The shipped topology: 40 backbone layers, candidate source (the last encoder layer) 20,
# index_source_layer_ids and engram_layer_ids from config.json.
N_LAYERS = 40
ENCODER_LAYERS = 21                      # 0..candidate_source_layer with DSV41_SWA_REPLAY=1
INDEX_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
ENGRAM_LAYERS = (1, 14)


def cache_bytes(whole_prompt: bool = True) -> float:
    """Resident bytes `DSV41_PREFILL_FP8_DEQUANT=cached` adds while a prompt is in prefill.

    `whole_prompt=False` is the encoder pass alone (layers 0..20, which is where the chunk loop
    spends its time); True adds what the decoder replay (layers 21..39 at M = 128) and the DSpark
    seed touch on their single pass each, which is the high-water mark the budget has to clear.
    """
    if not whole_prompt:
        return _weights(ENCODER_LAYERS, INDEX_LAYERS, ENGRAM_LAYERS, mtp=False) * BF16
    return _weights(N_LAYERS, INDEX_LAYERS, ENGRAM_LAYERS, mtp=True) * BF16


#: 3.70 GB -- what tools/budget.py adds to the prefill reserve when the mode is `cached`.
CACHE_BYTES = cache_bytes(whole_prompt=True)
