# GEMM dispatch audit

Every quantised matmul in this engine picks its kernel in Python, at call time, from two things:
the **type of the weight object** (`FP8Weight` / `FP4Weight` / `FP8GroupedWeight` /
`FP4GroupedWeight` / a plain tensor) and the **row count of the call**. Both branches of every one
of those dispatch points are numerically correct. Only one of them is fast. That is the failure
mode this page is about: on this same checkpoint, the single largest public win (35 -> 118 tok/s in
SGLang) came from noticing that a block/scale layout mismatch was sending GEMMs down a slow
fallback, and nothing about the output said so.

Two questions, and the two scripts that answer them:

* **Q1 — do our quantised GEMMs take the path we think?** The static half is below, marked
  *verified in code* where it was read out of the source and *needs the profiler run* where only a
  kernel list can settle it. `tools/audit_gemm_dispatch.py` produces that kernel list.
* **Q2 — is prefill compute-bound or bandwidth-bound?** `tools/prefill_bound_test.py`, and the
  decision rule at the bottom of this page.

The configuration audited is the shipped one: `EXPERT_FORMAT=cb3`, `DSV41_DENSE_FP4=attn,wo_a`,
`DSV41_HEAD_FMT=fp8`, `DSV41_WOA_FP8=1`, `DSV41_DENSE_FP8=1`, `DSV41_PREFILL_CHUNK=2048`,
`DSV41_SWA_REPLAY=1`, prune keep on with the device slot LUT. Target sm_121a, 48 SMs, one device.

---

## The inventory

| # | matmul | weight object | kernel it should reach | dispatch site |
|---|---|---|---|---|
| 1 | routed experts, decode | `CB3ArenaV2` | `_cb3v3_up_kernel` / `_cb3v3_down_kernel` | `cb3_moe.moe_forward_v3` |
| 2 | routed experts, prefill | `CB3ArenaV2` | `_cb3_unpack_kernel` **then** `_moe_up_kernel` / `_moe_down_kernel` | `cb3_moe.moe_forward_v3` -> `moe_forward_prefill` |
| 3 | `attn.wq_a/wq_b/wkv/wo_b` | `FP4Weight` | `_fp4_linear_kernel` | `v41_ref.dense` |
| 4 | `attn.wo_a` | `FP4GroupedWeight` | `_fp4_grouped_kernel` | `v41_ref.wo_a_proj` |
| 5 | `ffn.shared_experts.w1/w2/w3` | `FP8Weight` | `_fp8_linear_kernel` at M<=16, **cuBLAS on a bf16 dequant above it** | `v41_ref.dense` |
| 6 | LM head | `FP8Weight` | `_fp8_linear_kernel` at M<=16, **blocked dequant + cuBLAS above it** | `v41_ref.head_logits` |
| 7 | indexer `wq_b`, engram `wkv`, `mtp.0.main_proj` | `FP8Weight` | same as 5 — outside the `DSV41_DENSE_FP4` switch by design | `v41_ref.dense` |
| 8 | router gate, compressor `wkv`/`wgate`, indexer `wk`, HC mix | fp32/bf16 tensors | cuBLAS, in fixed 16-row tiles | `v41_ref.mm` |
| 9 | attention score / PV | fp32 activations | cuBLAS batched, in fixed 64-row tiles | `Model._softmax_attn` |

**There is no `torch._scaled_mm` anywhere in this repo, and no direct cuBLAS/cutlass call**
(*verified in code*: `grep -rn "_scaled_mm\|cublas\|cutlass" --include='*.py'` outside `corpus/`
returns only prose). Everything is either a Triton kernel of ours or `F.linear`/`torch.einsum`,
which is why cutlass names appear in a profile at all — they are cuBLAS's kernels, reached through
`F.linear`.

---

## 1–2. The CB3 routed-expert kernel

**Layout** (*verified in code*, `tools/cb3.py`, `tools/cb3_moe.py`). A CB3 expert is the packed-FP4
expert re-encoded as a per-row 8-entry codebook plus a 3-bit index per weight, split into two
planes plus the **unchanged** UE8M0 scales:

```
w1, w3 :  lo [2304, 1280] u8   hi [2304, 640] u8   cb [2304, 8] u8   s [2304, 160] u8
w2     :  lo [5120,  576] u8   hi [5120, 288] u8   cb [5120, 8] u8   s [5120,  72] u8
                                                        = 14.45 MB/slot (3.07 bpw) vs FP4's 18.80
```

The scale table is byte-identical to the FP4 one — one UE8M0 byte per 32 consecutive K weights of a
row — so **a scale-shape mismatch is not possible between the two formats**; the prefill unpack
copies the scale rows verbatim (`_unpack_into`: `s_dst[:B].copy_(s_src[src_slots.long()])`).

**The decode kernel** is `_cb3v3_up_kernel` / `_cb3v3_down_kernel` (the v3, inline-PTX variant; v1
and v2 are kept in the file as the measured ancestors and are not launched by the engine). One
program = one expert × `BM` (token, k) pairs × `BN` output rows. Launch config
(*verified in code*, `CB3_UP_CFG` / `CB3_DOWN_CFG`): `BN=32, num_warps=4, num_stages=3` at every
`BM`. `BM` comes from `_pick_bm(P)` with `P = T*k`: 16 for `P<=64`, 32 for `P<=1024`, 64 above.

Three dispatch hazards here, all real, none of them silent-fallback-shaped:

* **`num_warps=8` miscompiles this kernel.** The source says so in as many words: at `BN=32` it is
  both slower and *wrong* (rel err ~4 instead of 4.3e-3) — ptxas/Triton mishandles the inline asm at
  that warp count. Nothing in the shipped config uses 8 for CB3, but any future sweep must
  re-validate rather than assume. *Verified in code.*
* **The block tiles are hard-wired to 512/256 logical K** (`_cb3v3_block_dot`, `CB3.block_plan`),
  because a 128-byte `lo` row tile reaches 218 GB/s and a 32-byte `hi` tile caps at 101 GB/s on this
  box. `K=5120` and `K=2304` both decompose cleanly, so no tail path is taken. *Verified in code.*
* **The routing builder switches on size**: `build_routing_small` (pure torch, graph-capturable) at
  `P<=64`, the Triton `_route_kernel` above it. The Triton router costs one program per **arena
  slot** — with a 6,000-slot arena that was ~29 ms per decode step, which is why the small path
  exists. A prefill chunk always takes the Triton path. *Verified in code.*

**The prefill path is a deliberate fallback, and it is the biggest single thing in a prefill
chunk's MoE.** `moe_forward_v3` short-circuits when `P >= 65` and `DSV41_CB3_PREFILL == "fp4"`
(the default) into `moe_forward_prefill`, which:

1. takes the distinct arena slots the chunk touches, in batches of `DSV41_CB3_UNPACK_BATCH` (32),
2. runs `_cb3_unpack_kernel` to rebuild packed FP4 into a scratch `ExpertArena` (0.6 GB at the
   default batch),
3. runs the ordinary `_moe_up_kernel` / `_moe_down_kernel` over the scratch.

The reason is in the source: at decode there is one pair-block per expert and CB3 wins (0.79× the
FP4 time); at prefill block counts the per-block decode work dominates and CB3 measured 2.3–6.7×
FP4. So the fallback is correct as a choice. What it costs is not small and it is **not amortised**:
one unpack reads 14.45 MB and writes 18.80 MB per expert, a 2,048-token chunk touches ~370 of a
layer's 384 experts, and the unpack repeats for **every chunk and every layer**. That is ~1.2 GB of
read + 1.6 GB of write per layer per chunk — call it 25 GB of pure format conversion per encoder
pass of one 2,048-token chunk, on a box whose practical copy bandwidth is ~210 GB/s. The existing
measurement that prefill throughput scales with *chunk count* rather than token count (291 tok/s at
chunk 512, 326 at 1024, 369 at 2048) is exactly this signature. *Verified in code; the share of GPU
time needs the profiler run.*

Two launch-config notes worth a cheap re-sweep, both *verified in code*:

* the prefill path reuses `fp4_moe._UP_CFG` / `_DOWN_CFG`, and at `BM=64` the **up** kernel runs
  with `num_stages=1` — no software pipelining, loads not overlapped with compute — against
  `num_stages=3` for the down kernel. Those tables came from a sweep; the up entry at prefill `BM`
  is the one to re-check first.
* `_unpack_into` launches with `BN=64, num_warps=4, num_stages=2` and was never swept at all.

---

## 3–4. The FP4 dense projections and the `wo_a` grouped kernel

**Layout** (*verified in code*, `tools/fp4_linear.py`). `FP4Weight` = E2M1 codes `[N, K/2]` u8 (low
nibble = even K element) + UE8M0 scales `[N, K/32]` u8, one scale per 32 consecutive K weights of a
row. `FP4GroupedWeight` is the same two buffers addressed as `G=8` row ranges of `[R=1024, K=4096]`.
Both constructors **assert the exact shapes and dtypes**, and `K % 128 == 0`:

```python
assert codes.dtype == torch.uint8 and scales.dtype == torch.uint8
assert tuple(codes.shape) == (N, K // 2)
assert tuple(scales.shape) == (N, K // 32)
assert K % 128 == 0
```

so a **scale-shape mismatch cannot reach the kernel** — it raises at load time. Both objects call
`.contiguous()` on construction. *Verified in code.*

**Activation layout.** `fp4_linear` reshapes to `[-1, K]`, casts to bf16 if needed, and calls
`.contiguous()`; `fp4_grouped_linear` asserts `[T, G, K]` and calls `.contiguous()`. Every caller in
`engine/model.py` feeds a freshly produced contiguous tensor (`rmsnorm` output, `act_qdq_fp8`
output, `o.reshape(T, o_groups, -1)`), so **the contiguous() calls are no-ops and never silently
copy**. *Verified in code; a copy would show up in the profiler as an `elementwise`/`copy_` row, so
the run confirms it.*

**Tile selection** (*verified in code*): `BLOCK_M = 16 if M <= 16 else 64`; `BLOCK_N = 32` at decode
M and `128` at prefill M for the dense kernel, `64`/`128` for the grouped one — both measured, with
the reasoning in the docstrings (at M=6 a 128-wide N tile leaves only 4–40 programs for 48 SMs).
`USE_F16` defaults to 1 (`DSV41_FP4_DENSE_F16`): activations enter `tl.dot` as fp16, weights decode
to fp16 through the hardware `cvt.rn.f16x2.e2m1x2`.

**The one real fallback here is opt-in and off.** `v41_ref.dense` and `v41_ref.wo_a_proj` both carry
a branch that dequantises to bf16 and calls `F.linear`/`einsum` for `M > 16`, gated by
`DSV41_FP4_DENSE_PREFILL`; the default is `kernel`, i.e. **the FP4 kernels run at every M including
prefill**. If that variable is ever set to `dequant`, the profile grows a cutlass row and an
elementwise dequant row and loses `_fp4_linear_kernel`. *Verified in code.*

**What cannot be proven by reading**: whether `tl.dot` on these tiles actually issues tensor-core
MMAs. The K step inside `_chunk_dot` is 16 (`xe` is `[BM, 16]`), which is the minimum for an fp16
MMA — legal, but at `BM=16, BN=32` it is a very small instruction and Triton's choice between MMA
and FMA lowering is not visible from the Python source. `torch.profiler` cannot answer this either:
the kernel name is `_fp4_linear_kernel` whichever it picked. **This one needs `ncu`
(`sm__inst_executed_pipe_tensor`), not the profiler run** — flagging it rather than pretending the
audit covers it.

---

## 5–7. What is still fp8, and what that costs at prefill M

`DSV41_DENSE_FP4=attn,wo_a` covers `attn.wq_a/wq_b/wkv/wo_b` and `attn.wo_a`. It does **not** cover
the `shared` group, and by design never covers the indexer `wq_b`, the engram `wkv` or
`mtp.0.main_proj`. Those stay `FP8Weight`, and `v41_ref.dense` for an `FP8Weight` is:

```python
if FP8Weight is not None and isinstance(w, FP8Weight):
    if x.numel() // x.shape[-1] <= 16:
        return fp8_linear(x, w)          # the Triton kernel
    return F.linear(x.to(torch.bfloat16), w.dequant())   # <-- prefill takes THIS
```

and `FP8Weight.dequant()` is:

```python
s = torch.exp2(self.s.float() - 127.0)
s = s.repeat_interleave(32, 0)[: self.N].repeat_interleave(32, 1)[:, : self.K]
return (self.w.float() * s).to(torch.bfloat16)
```

That is a **full `[N, K]` fp32 scale tensor materialised by two `repeat_interleave`s, a full `[N,K]`
fp32 weight, and a bf16 cast — allocated and thrown away on every prefill call, per weight, per
layer.** For one layer's three shared-expert matrices (`w1`/`w3` `[2304, 5120]`, `w2` `[5120, 2304]`)
that is ~35.4 M weights × (4 B scale + 4 B fp32 + 2 B bf16) ≈ 354 MB of transient traffic **per
layer per chunk**, i.e. ~7.4 GB per 21-layer encoder pass of a single 2,048-token chunk, before the
GEMM itself reads the bf16 copy back. The existing measurement that this dequant is the dominant
cost of the fp8 prefill path (5.14 ms of 7.14 ms for `wq_b` at M=2048, against 4.33 ms for the FP4
kernel) is the same effect on a weight that has since moved to FP4. *Verified in code; the share of
GPU time needs the profiler run.*

**This is the cheapest lever on the list.** `DSV41_DENSE_FP4=attn,wo_a,shared` already exists, is
already implemented, and moves the shared experts onto `_fp4_linear_kernel` at every M — the switch
was left at `attn,wo_a` for **quality** reasons (per-group NLL deltas), not because the shared group
lacks a kernel. If the profiler shows the dequant elementwise rows are large, the options in order
are: (a) add `shared` to the switch and re-run the quality gate, or (b) keep fp8 and use the
existing `_fp8_linear_kernel` at prefill M too, which costs only a `BLOCK_M=64` sweep — the kernel
already has the prefill tile shape and the branch is a single comparison.

The **LM head** (`DSV41_HEAD_FMT=fp8`) takes `_head_blocked` for `M > 16` unless
`DSV41_HEAD_PREFILL=kernel`: it dequantises 16,384 vocabulary rows at a time (166 MB transient) and
hands each block to cuBLAS, because both quantised kernels are bandwidth-shaped and measured
24–27 TFLOPs at M=512 against cuBLAS's 80. With `DSV41_SWA_REPLAY=1` prefill chunks run
`need_logits=False`, so **the head is not a per-chunk prefill cost** — it runs once per prompt, in
`decoder_replay`, at M=128. Correct as it stands; listed so the profile's cutlass rows are not
mistaken for a leak. *Verified in code.*

---

## 8–9. The unquantised GEMMs, and the 16-row tiling

`v41_ref.mm` runs **every** bf16/fp32 activation GEMM in fixed `MM_TILE = 16` row tiles, and
`v41_ref.tiled_rows` does the same for every last-dim reduction. This is not an optimisation, it is
the chunk-invariance guarantee: cuBLAS picks tiling *and* split-K from M, and for several of these
shapes even a row's offset inside the tile changes its last bits. It is already costed in the
gotchas at "about +10%: a 512-token prefill chunk issues 32 GEMM launches per projection instead of
one". At a 2,048-token chunk it is **128 launches per projection**. *Verified in code.*

Quantised weights escape this (`mm` returns `dense(x, w)` before the tiling, because both Triton
kernels are row-count- and row-offset-invariant by construction), so the tiling now applies only to
row 8 and 9 of the inventory. One of those is much larger than it looks:

**`hc_mixes` runs the 20-iteration Sinkhorn under `tiled_rows` at prefill.** The torch
implementation (`v41_ref.hc_split_sinkhorn`) is ~130 tiny elementwise/reduction kernels per call on
a `[m, 4, 4]` tensor. `tiled_rows` calls it once per 16 rows. At `T = 2048` that is 128 calls
× ~130 kernels ≈ **17,000 launches per `hc_mixes`**, and `Model.block` calls `hc_mixes` twice per
layer over 21 encoder layers, i.e. of the order of **700,000 kernel launches per 2,048-token
chunk**, every one of them on a 16×4×4 tensor.

A fused replacement already exists and is already used — by decode only. `engine/hc_sinkhorn.py`
is a Triton kernel, one program per token row, that "replaces ~160 tiny torch ops per layer with one
launch"; `engine/fastdecode.py` imports it, `engine/model.py` does not. Because it is one program
per row it is row-count- *and* row-offset-invariant by construction, so using it at prefill would
remove the need for the tiling here as well as the launches. The existing note that prefill
`hc_mixes` "still goes through `R.mm`, and at prefill M cuBLAS is in its element" is about the GEMM
inside `hc_mixes` and is correct — it is the **Sinkhorn after** the GEMM that was never moved.
*The launch count is verified in code; whether it dominates the chunk needs the profiler run*, and
it is the specific reason `tools/audit_gemm_dispatch.py` reports the GPU-busy fraction and the total
launch count alongside the kernel table: a launch-bound phase is invisible in a top-20-by-GPU-time
list, because each of those 700,000 kernels is individually microscopic.

The other tiled sites — the router gate (`N=384`), the compressor `wkv`/`wgate`, the indexer `wk`,
the attention score/PV at `ATTN_TILE=64` — are GEMMs, where a 16- or 64-row tile is merely
inefficient rather than pathological.

---

## How to prove any of this: `tools/audit_gemm_dispatch.py`

```
python3 tools/audit_gemm_dispatch.py --json results/gemm_dispatch.json
python3 tools/audit_gemm_dispatch.py --phase prefill --chunk 2048
python3 tools/audit_gemm_dispatch.py --expect _fp4_linear_kernel,_moe_up_kernel,cutlass
```

It runs one prefill chunk and one decode step under `torch.profiler` and prints, per phase: the
wall time measured **without** the profiler attached, the total GPU kernel time and the GPU-busy
fraction of that wall, the total launch count, the top-20 CUDA kernels by device time with call
counts, everything above 5 % of GPU time that matches nothing in `--expect`, and every kernel whose
name matches a known fallback shape (`fallback`, `unpack`, `elementwise`, `memcpy`, `dequant`,
`repeat_interleave`, …) at any share.

**What each answer means.**

| what the run shows | reading |
|---|---|
| `_cb3v3_up_kernel` / `_cb3v3_down_kernel` present in **decode**, absent in prefill | correct: prefill takes the unpack path by design |
| `_cb3_unpack_kernel` a large share of prefill | expected, and it is the thing to attack (P2) |
| `_moe_up_kernel` / `_moe_down_kernel` in prefill | correct: the unpacked scratch is FP4 |
| `_fp4_linear_kernel` + `_fp4_grouped_kernel` in **both** phases | correct: FP4 dense runs at every M |
| `cutlass_*` / `*gemm*` large in prefill | the fp8 dequant path (shared experts, indexer, engram) and the 16-row-tiled bf16 GEMMs |
| `elementwise` / `repeat_interleave` / `copy_` large in prefill | the `FP8Weight.dequant()` transients — row 5 above |
| GPU-busy fraction well under 1, launch count in the hundreds of thousands | launch-bound: the Sinkhorn tiling, not any kernel |
| any `*_fallback*` name at all | the Triton import failed and `engine/moe_fallback.py` is running — check the startup log line |

A decode step is the control: the last full decode profile put self-CUDA at 116.6 ms against 118.9 ms
of wall (~98 % GPU-busy) with `_cb3v3_up_kernel` 41.0 ms, `_cb3v3_down_kernel` 22.5 ms,
`_fp4_linear_kernel` 16.2 ms, `_fp8_linear_kernel` 9.8–12.6 ms, `_fp4_grouped_kernel` 4.6 ms,
`unrolled_elementwise_kernel` 2.1 ms, `_skinny_kernel` 1.9 ms. **If the decode half of the audit
does not reproduce roughly that, the audit is measuring something other than the served
configuration** and nothing in the prefill half should be trusted either.

---

## Q2: is prefill compute-bound or bandwidth-bound?

`tools/prefill_bound_test.py` runs one real prefill chunk, taps ONE backbone layer's MoE input and
router output, and re-times that layer's MoE with the router's top-k forced to 6, 4 and 3 — three
repeats, CUDA-synchronised, median. It reports two numbers per k: `layer` (the whole `Model.moe`,
which includes the shared expert and therefore can only scale sub-linearly) and `routed` (`moe_fn`
alone — on this configuration the unpack plus the two FP4 launches). It also reports how many
**distinct** experts the chunk still touches at each k.

```
python3 tools/prefill_bound_test.py --chunk 2048 --layer 10 --k 6,4,3 --repeats 3
```

The override is `DSV41_PREFILL_TOPK_TEST` (`engine/prefill_topk.py`): prefill-only, off by default,
byte-identical off, refused in `DSV41_PRUNE_MODE=drop`, pinned by `tools/test_prefill_topk.py`. It
is deliberately **not** `DSV41_TOPK`, which rewrites `n_activated_experts` for the whole engine
including the decode graphs. The script drives it by assigning the module attribute rather than the
environment variable, because an A/B on this box has to happen inside one process: a second process
lays the ~89 GB arena down at a different offset, and that alone moves a streaming kernel's time.

### The decision rule

Let `r = routed(k=3) / routed(k=6)`. The pair count falls by exactly 0.50; if time were pure
per-pair compute, `r` would too.

* **`r <= 0.65` — compute-bound.** Time tracks the pairs. An ExFold-style top-k reduction in
  prefill (arXiv 2608.24938, validated on DeepSeek-V4-Flash) is worth implementing: it buys close to
  its own ratio, and the quality question becomes the only question.
* **`r >= 0.90` — bandwidth/unpack-bound.** Cutting k buys nothing, because the cost is proportional
  to the **distinct experts touched**, not to the pairs: at 2,048 tokens a layer touches ~370 of 384
  experts at k=6 and still ~350 at k=3, and each of those is one 14.45 MB read plus one 18.80 MB
  write through `_cb3_unpack_kernel` whatever k is. Then the levers are the ones that cut bytes per
  chunk — larger chunks (fewer unpacks), a persistent or cached unpack scratch, or a CB3 kernel fast
  enough at prefill block counts to skip the unpack entirely (`DSV41_CB3_PREFILL=direct` is the
  measurement arm that already exists) — plus the two non-MoE items above: the shared-expert fp8
  dequant and the Sinkhorn tiling.
* **`0.65 < r < 0.90` — mixed.** Report the number. The `distinct_experts` column says how much of
  the residue is unpack: if `distinct(3)/distinct(6)` is near 1 while `r` is near 0.8, the compute
  half is real but the unpack floor is what caps the win.

**The prior, stated so the run can contradict it:** the code says the prefill MoE pays a fixed
per-expert unpack whose size does not depend on k, and the existing chunk-size measurements
(291 / 326 / 369 tok/s at chunk 512 / 1024 / 2048) are the signature of a per-chunk, per-expert
cost rather than a per-token one. So `r` near 1 is the expected outcome and ExFold is expected to
be the *wrong* next thing on this engine. That is a prediction, not a finding — it is what the run
is for.
