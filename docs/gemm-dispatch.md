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

The override is `DSV41_PREFILL_TOPK` (`engine/prefill_topk.py`): prefill-only, off by default,
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

---

## Q2, answered — 2026-09-15

The run, on the shipped configuration (`EXPERT_PROFILE=frontend`, `PRUNE_KEEP=0.36`, chunk 2,048,
layer 10, three repeats, median):

| k | routed `moe_fn` | relative | pairs relative | what pure per-pair compute would give |
|---|---|---|---|---|
| 6 | 54.1 ms | 1.000 | 1.000 | 1.000 |
| 4 | 38.3 ms | 0.708 | 0.667 | 0.667 |
| 3 | 30.9 ms | 0.571 | 0.500 | 0.500 |

`r = routed(3) / routed(6) = 0.57`. The decision rule's compute-bound branch is `r <= 0.65`.

**The prediction written above this section was wrong, and it is left standing on purpose.** It
said `r` near 1: the per-expert unpack is paid once per expert per chunk whatever k is, a
2,048-token chunk still touches nearly every resident expert at k=3, and the chunk-size curve
(291 / 326 / 369 tok/s at 512 / 1024 / 2048) is the signature of a per-chunk cost. All of that is
true and none of it dominates. The unpack is a floor, not the bill: at k=3 the time falls almost
as fast as the pairs do, 0.57 against 0.50, so the FP4 grouped GEMMs are where a prefill chunk's
MoE time actually goes. Reading the code told us which kernels run; only the measurement told us
which of them costs.

### The decision, and how big it can be

**Fewer experts per prefill token is worth implementing** — and the honest size of it is small,
because the MoE is not the whole chunk. The FP4 MoE kernels are ~893 ms of a ~3.7 s chunk (24 %;
~42.5 ms per layer over the 21 layers a prompt runs with `DSV41_SWA_REPLAY=1`). Scaling only that
share:

| k | routed MoE | expected prefill speedup |
|---|---|---|
| 4 | 0.708x | **1.075x** |
| 3 | 0.571x | **1.115x** |

So the ceiling here is ~7–12 % on prefill, not the 1.32x the ExFold paper reports on an H800 — on
that box the MoE is a much larger share of a chunk, and its 8K TTFT figure also carries queueing
amplification (their own Appendix says so). Anyone quoting 1.32x for this engine is quoting a
different machine's bottleneck. `tools/verify_prefill_topk.sh` measures what this one does, and
prints those predictions next to the measurement so the two cannot drift apart.

Which leaves quality as the only real question, and that is why the reduction shipped as **ExFold**
(arXiv 2608.24938) rather than as a smaller `topk`: on DeepSeek-V4-Flash, at three routed experts
per prefill token, direct Top-3 reduction keeps 97.05 % of the Top-6 baseline average and folding
keeps 98.42 %. See [architecture](architecture.md#fewer-experts-in-prefill).

### What this does not license

The two non-MoE items flagged above have since been given switches of their own -- see "What
changed after the audit" below -- but neither has been run on the box: the shared expert's
`FP8Weight.dequant()` at prefill M (`DSV41_PREFILL_FP8_DEQUANT`), and the 20-iteration Sinkhorn
under the 16-row tiling, ~700,000 launches per chunk, with a fused kernel already written and used
by decode only (`DSV41_PREFILL_FUSED_SINKHORN`). A
compute-bound MoE makes the second of those *more* interesting, not less — a launch-bound phase is
invisible in a profile sorted by GPU time, which is exactly what `tools/audit_gemm_dispatch.py`'s
GPU-busy fraction is for.

---

## What changed after the audit — the two fixes, both off by default

Two of the items above were fixed rather than only measured. Both are behind an environment
variable, both default to the behaviour that shipped, and off is byte-identical — the audit's own
figures in the sections above are still figures of the default engine.
`tools/verify_prefill_fixes.sh` is the run that says whether either is worth turning on: four runs
on the shipped `.env` over one identical ~7,000-token prompt (both off, each alone, both), TTFT and
`prefill_tok_s` from the server, plus this page's own script for the launch count and the GPU-busy
fraction of the first and the last, then one generation gate on the last. It records under
`results/prefill/`.

### Fix A — `DSV41_PREFILL_FP8_DEQUANT`, for row 5 and row 7

`FP8Weight.dequant()` is untouched; what changed is that `v41_ref.dense` no longer has to call it.
The M > 16 branch now goes through `v41_ref.prefill_dequant`, which is `w.dequant()` unless the
variable says otherwise.

`fused` (`tools/fp8_linear.dequant_fused`, a new Triton kernel next to the two that were already
there) reads the fp8 codes and the UE8M0 table and writes bf16 in one launch. **Two transient bytes
per weight instead of ten, about three bytes of DRAM traffic instead of about twenty-seven, one
launch instead of six, and no fp32 intermediate at all.** It is bit-identical, not approximately:
e4m3 → bf16 loses nothing (four significand bits into eight), the block scale is a power of two so
the fp32 product is exact wherever `dequant()`'s is, and the closing round-to-nearest-even to bf16
is the same rounding. `tools/test_fp8_dequant.py` compares raw bits, on the CPU for the torch
fallback and on the device for the kernel, and self-skips where neither is available.

The size of the prize, from the numbers already on this page. The weights still on that branch
under `DSV41_DENSE_FP4=attn,wo_a` are `ffn.shared_experts.w1/w2/w3` on all 21 encoder layers, the
indexer `wq_b` on layers {2, 8, 14, 20} and the engram `wkv` on layers {1, 14}:

```
21 x  shared w1/w3 [2304, 5120] + w2 [5120, 2304]      743,178,240 weights
 4 x  indexer wq_b [4096, 1280]                         20,971,520
 2 x  engram  wkv  [25600, 6144]                       314,572,800
                                             total   1,078,722,560 weights per chunk
```

The measured rate of this shape is the datum in the row-5 section: 5.14 ms to dequantise a
[4096, 1280] weight at M = 2048, i.e. **0.98 ns per weight** — six allocator-bound elementwise
kernels, not one streaming copy, which is why it is nowhere near this box's ~210 GB/s. 1.079 G
weights at that rate is **~1.06 s per 2,048-token chunk**, against ~5.55 s for a chunk at the
measured 369 tok/s. **About a fifth of prefill is spent turning a weight into a copy of itself**,
and a kernel that streams 3 bytes per weight at device bandwidth would spend ~16 ms doing it.

`cached` is `fused` plus memoisation for the rest of a request's prefill: a weight is dequantised
on the first chunk and read back on every chunk after it, which buys the residual ~16 ms a chunk
and only from the second chunk on. It costs **2.16 GB** while the encoder pass is running and
**3.70 GB** once the decoder replay and the DSpark seed have touched their own weights — the
engram `wkv` alone is 0.63 GB of the first figure — and that comes out of the arena. It is in the
pre-flight model: `tools/budget.py` reads the variable itself and adds the whole-prompt figure to
the prefill reserve, so `PRUNE_KEEP=auto` and `./tune.sh` already answer for it
(`docs/memory-budget.md`, gate 2). **`fused` is the recommendation**: it helps the first chunk as
well as the rest, it helps the replay, and it costs nothing, where `cached` pays 3.70 GB for the
last 1.5 % of the same lever and gives a single-chunk prompt nothing at all.

`scaled_mm` is **refused with a reason rather than implemented**, and the name is kept so the
reason is discoverable. `torch._scaled_mm` takes per-tensor, per-row, 1x128/128x128 block-wise or
1x32 (MX) scales. This checkpoint stores **one UE8M0 scale per 32x32 block** — `FP8Weight.__init__`
asserts exactly that. 32x32 is not one of those granularities, so reaching `_scaled_mm` would mean
re-quantising the weight into a layout it accepts, which moves the numbers the recipe was measured
on. This page's own warning is the point: a layout mismatch does not raise, it dispatches to
something slow and correct, and that is precisely the failure the 35 → 118 tok/s public win was.

### Fix B — `DSV41_PREFILL_FUSED_SINKHORN=1`, for the 16-row tiling of rows 8–9

No new kernel. `engine/hc_sinkhorn.py` already replaces the whole 20-iteration Sinkhorn with one
Triton launch, one program per token row, and `engine/fastdecode.py` has used it since the fast
decode path existed; `engine/model.py` simply never picked it up. It needed no extension for
prefill shapes either — the grid is `(max(n, 1),)` with an `if row >= n_rows: return` guard, `HC`
is a `tl.constexpr` of 4, and there is no batch or row limit in it. `v41_ref.hc_mixes` takes a
`fused` keyword, `engine/model.py` passes `PS.fused_prefill(prefill)`, and the GEMM and the rsqrt
above the Sinkhorn stay tiled in both modes.

The count, per 2,048-token chunk of the encoder pass:

| | launches |
|---|---|
| today: 128 tiles × ~130 torch kernels × 2 calls per layer × 21 layers | **698,880** |
| fused: 1 launch × 2 calls per layer × 21 layers | **42** |

One program per row is row-count- *and* row-offset-invariant by construction, which is the property
`tiled_rows` exists to manufacture, so the tiling is not merely skipped on this path — it is
replaced by something stronger. The two paths are not expected to be bit-identical (130 torch
kernels and one Triton program do the same normalisations in a different order, and `tl.exp` is not
`torch.exp`), so `tools/test_hc_sinkhorn_prefill.py` measures the disagreement at every row count
from 1 to 2,048, checks that it does not grow with the row count, and checks the invariance
directly; it self-skips off a CUDA box. It is **prefill-only** — a fused Sinkhorn leaking into the
un-graphed decode forward would change the tokens the DSpark verify step is asked to accept and
nothing would crash, the acceptance rate would just quietly move — and `tools/test_prefill_sinkhorn.py`
pins that guard mechanically.

**This is the fix the kernel table cannot score.** Every one of those 698,880 launches is
microscopic, so none of them appears in a top-20-by-GPU-time list at any share; the only evidence
either way is the launch count and the GPU-busy fraction that `tools/audit_gemm_dispatch.py`
reports beside the table, which is why the verify script runs it on the baseline and on both-on
and on nothing in between.

---

## What the run found: two fp32 attention GEMMs, 855 ms of a 2,048-token chunk

*Added 2026-09-15, after the first `tools/audit_gemm_dispatch.py --phase prefill` run on the box:
one 2,048-token chunk, the shipped configuration, GPU 99 % busy over 24,816 launches.*

The prediction above was about the MoE. The largest single line in the table is not in the MoE and
is not a quantised kernel at all:

| kernel | ms | calls |
|---|---|---|
| `cutlass_80_simt_sgemm_128x32_8x5_tn_align1` | 559.5 | 672 |
| `cutlass_80_simt_sgemm_128x32_8x5_nn_align1` | 295.7 | 672 |
| **both together** | **855.2** | **1,344** |
| `_moe_up_kernel` | 483 | — |
| `_moe_down_kernel` | 410 | — |

`simt_sgemm` says fp32 through the SIMT pipeline: FFMA, not a tensor core. On this box that costs
roughly an order of magnitude against the tensor-core path before any tiling question is asked.
The two rows are more expensive than both FP4 MoE kernels the chunk exists to run.

### Which matmuls they are, from the call counts

The count settles it without a single ambiguity. `engine/model.py::Model._softmax_attn` runs the
attention softmax in fixed query tiles of `ATTN_TILE = 64` rows, and each tile issues exactly two
products:

```python
scores = torch.einsum("thd,tnd->thn", qt, kvt) * scale   # the score product
...
return torch.einsum("thn,tnd->thd", p / denom, kvt)      # the PV product
```

with `qt = q[i:j].float()` and `kvt = kv_all[i:j].float()` — both widened from the bf16 they are
stored in, so both einsums are fp32, and `torch.einsum` on a `[t, ·, ·]` pair is a strided-batched
cuBLAS GEMM, which reaches the same cutlass kernels `F.linear` does.

```
      2048 tokens / ATTN_TILE 64        =  32 tiles per layer
      x 1 score product per tile        =  32 score GEMMs per layer
      x 21 encoder layers (0..20, the candidate-source layer; DSV41_SWA_REPLAY=1)
                                        = 672 calls   <- the _tn_ row
      and the same arithmetic for the PV product
                                        = 672 calls   <- the _nn_ row
                                  total = 1,344
```

32 per layer, 672 per product, 1,344 together: exactly the profile. Nothing else in the prefill
path has that count — the `MM_TILE = 16` sites issue 128 calls per layer per projection at this
chunk length (the HC mix GEMM twice, the router gate once, the compressor twice on the KV-source
layers), and the indexer's score einsum is bf16 and nested two deep. *Verified in code.*

The `_tn_` / `_nn_` split matches the two contractions: `thd,tnd->thn` contracts the last axis of
both operands (one operand transposed, `tn`, and the more expensive of the two because N = the KV
width), `thn,tnd->thd` is an ordinary product (`nn`).

Shapes, per tile, batch 64 (the query rows):

| | M | N | K | operands |
|---|---|---|---|---|
| score | 64 (`n_heads`) | 640 (`window_size` 128 + `index_topk` 512) | 512 (`head_dim`) | fp32, contiguous |
| PV | 64 | 512 | 640 | fp32, contiguous |

Two of the 21 encoder layers have no compressed stream (`w.ratio == 0`), and their KV width is 128
rather than 640; the arithmetic above is unchanged because the tile count is a property of the
query axis, not of the KV axis.

That is 2.68 GFLOP per product per tile, so 3.6 TFLOP per chunk for the pair — 855 ms of it is
~4.2 TFLOP/s, which is about a quarter of this box's fp32 SIMT ceiling. Half of the shortfall is
visible in the kernel name: a `128x32` threadblock tile against M = 64 leaves half the tile idle.

### About `align1`

`align1` is the alignment-1 (scalar-load, non-vectorised) variant, and it is worth saying plainly
that **there is no alignment fix to make here**. `q` is a fresh contiguous tensor out of
`torch.cat`, `kv_all` a fresh contiguous tensor out of `torch.cat`, `p / denom` a fresh contiguous
tensor, and every leading dimension in play — 512, 640, 128 — is a multiple of four floats, i.e.
16-byte aligned, on top of base pointers the caching allocator hands out 512-byte aligned. Both
operands already satisfy `align4`. The alignment-1 pick follows from cuBLAS's heuristics for the
batched fp32 path, not from anything the caller can restride, so the lever is the **math type**,
not the layout. (This is the one place the audit's own advice — read the kernel name — can be
read too eagerly: a name that describes the kernel cuBLAS chose is not always a description of a
mistake the caller made.)

Nor is the tiling the lever. Raising `ATTN_TILE` above 64 would halve or quarter the launch count,
but at 855 ms over 1,344 calls each call is 0.64 ms — launch overhead is well under a percent of
it — and it would not improve a single GEMM, because the batch axis is the query rows and M stays
`n_heads = 64` whatever the tile is. It would only cost transient memory, linearly.

One layout cost *is* real and is not in the sgemm rows: `kv_all[i:j].float()` materialises a
`[64, 640, 512]` fp32 tile, 83.9 MB, once per tile — 2.7 GB written per layer, ~56 GB written and
~28 GB read per 2,048-token chunk, purely to widen bf16 that the GEMM is about to reduce anyway.
It shows up in the profile's `elementwise` / `copy_` rows, not in the GEMM rows. Only the `bf16`
mode below removes it; `tf32` keeps every byte of it.

### The switch: `DSV41_PREFILL_ATTN_GEMM=fp32|tf32|bf16`

`engine/prefill_attn_gemm.py`, prefill-only, `fp32` by default, and `fp32` is byte-for-byte the
engine that shipped — the module writes nothing and imports no torch on that path, and the two
`.to(dt)` calls in `_softmax_attn` are identities at `dt=float32`.

| mode | what changes | error against fp32 | memory |
|---|---|---|---|
| `fp32` | nothing | 0 (bit-identical) | as shipped |
| `tf32` | `torch.backends.cuda.matmul.allow_tf32` and `float32_matmul_precision` armed around the tile loop and restored after it | inputs rounded to an 11-bit significand, fp32 accumulate: ~2^-11 = 4.9e-4 per operand, of the order of 1e-3 on the product | as shipped |
| `bf16` | the operands are never widened; bf16 in, fp32 accumulate on the tensor cores; the score product is widened back before the softmax | 8-bit significand: ~2^-8 = 3.9e-3 | **less** — the 84 MB-per-tile fp32 widening disappears |

`tf32` is the "keep fp32 accumulate" path: TF32 truncates the operands and still accumulates in
fp32, which is why it sits an order of magnitude closer to the reference than `bf16` does.

`tools/budget.py::prefill_bytes` is unaffected: `fp32` and `tf32` allocate exactly what the engine
always allocated, and `bf16` allocates strictly less, so the ceiling the budget enforces still
holds in every mode and no gate moves.

### The risk, and why a tok/s number cannot accept either mode

This engine's precision argument is chunk invariance — `v41_ref.mm`'s 16-row tiling, `ATTN_TILE`
itself, `allow_bf16_reduced_precision_reduction = False` — and the reason is written directly above
this GEMM pair in `engine/model.py`: *"rounding the probabilities to bf16 first is a cliff that
turns 1e-7 fp32 GEMM jitter into 1e-4 output jitter, which is enough to flip a borderline router
top-k and make the MoE output depend on the chunk length."* Both non-default modes are far above
1e-7. `tf32`'s 5e-4 on the operands is three orders of magnitude above the jitter the design
budgets for, and `bf16`'s 4e-3 is the cliff itself, named.

What goes wrong is therefore not a wrong number but a different expert, deep in a long prompt, and
the only instrument that sees it is generation. `tools/verify_prefill_hc.sh` ends on
`tools/gate_profile.py --profile Frontend --thinking on` for that reason, and its record says so
in as many words when the gate was skipped. The prefill path also writes the global KV and the
window rings that every later decode step reads, so a prefill-only precision change is not
decode-neutral in its *effects*, only in its *kernels*.

Both modes remain ungated as this is written: the arithmetic and the bounds are settled, no
generation has been taken on the box with either armed.

### What to run, and what each answer means

```
./tools/verify_prefill_hc.sh                 # three speed runs, two audits, one gate
SKIP_GATE=1 ./tools/verify_prefill_hc.sh     # speed + audit only, and accepts nothing
```

| what the run shows | reading |
|---|---|
| the two 672-call `simt_sgemm` rows present in the `fp32` audit | the finding reproduces |
| the `tf32` audit still showing them, unchanged | the flag never reached cuBLAS — every speed number in that run is fp32 measured three times, and nothing else in the record is worth reading |
| `tf32` showing a `tensorop` / `s1688` / `s16816` name at roughly 672 calls, sgemm ms down | the GEMMs moved to the tensor cores; the prefill tok/s row says what that bought |
| prefill tok/s flat while the sgemm ms fell | the chunk was never bound by those kernels — look at the GPU-busy fraction and the launch count instead (the Sinkhorn tiling, §8–9) |
| `bf16` much faster than `tf32` | that is the 56 GB of widening, not the GEMM; it is the strongest speed case and the weakest precision case |
| the gate on `tf32` matching `results/keepsets/frontend/GATE.md` | the only evidence that would accept the mode |

### The other fp32 GEMM, which this switch deliberately does not touch

The Hyper-Connection mix projection (`tools/v41_ref.py::hc_mixes` → `mm`) is fp32 too, at M=16,
N=24, K=20480 under the `MM_TILE` tiling — 5,376 calls per chunk. It is cuBLAS's
`gemmSN_TN_kernel`, not a cutlass simt sgemm, it is latency-bound rather than FLOP-bound, and its
cure is not a math type: the split-K Triton kernel in `tools/fp32_skinny.py` already does it in
fp32 throughout (`input_precision="ieee"`) at 3× cuBLAS's speed, and `engine/fastdecode.py` has
used it since it was written. Prefill does not. That is a separate, precision-free change and it
belongs with the Sinkhorn item in §8–9, not here.

`DSV41_PREFILL_HC_GEMM` is accepted as an alias for the switch: the investigation began on the
assumption that the 1,344 calls were the HC residual's, the call counts said attention instead, and
the alias is kept so that name still reaches the right knob.
