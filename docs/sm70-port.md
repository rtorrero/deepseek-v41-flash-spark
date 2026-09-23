# Running DeepSeek-V4.1-Flash on 4x Tesla V100 32 GB (SM70) + 256 GB DDR4

Branch `sm70-v100-port`. This document is the map: what blocks the port, what this branch
already does about it, what is measured, and what still has to be run on the V100 host.

Everything here that could be checked without a GPU **was** checked, on an AMD Ryzen 5900X with
AVX2 and no CUDA driver at all. Everything that needs a GPU is labelled as such and comes with
the one command that tests it.

## Why this is worth doing

The routed experts are 15,360 slots of 18,800,640 B = **288.8 GB** at FP4. The single-box
reference implementation (`0xBakeer/deepseek-v41-flash-spark`, from which this tree comes) runs on
a 121 GiB DGX Spark, so it keeps a 74-87 GB hot set resident and **streams 0.92 GB of expert
weights off NVMe per generated token**, which is why it decodes at 2.6 tok/s with the SSD
responsible for over 92% of the step.

A 4x V100 host with 256 GB of DDR4 has 384 GB of memory in two tiers whose bandwidths differ by
roughly 30x:

| tier | capacity | bandwidth | what fits |
|---|---|---|---|
| 4x V100 32 GB HBM2 | 128 GB | ~900 GB/s per card | dense weights, KV, drafter, the hottest experts |
| DDR4 | 256 GB | measured 29 GB/s here (see below) | **the rest of the experts** |
| local NVMe | - | ~2.5-5.5 GB/s | the Engram tables (198 GB, gathered ~12 KB per token) |

288.8 + 18.1 + 7.2 = **314 GB** of weights and caches: it fits the hierarchy with room to spare,
**nothing has to be requantised, and no expert byte has to be streamed**. The measured single-box
recipe cannot do that on the Spark; this box can. What is missing is not capacity, it is the
software to compute the RAM-resident experts -- upstream sglang stages offloaded weights back to
the GPU before computing (`OffloaderV1.forward` does `v.to(device)`), which would put ~1 GB per
token over PCIe 3.0.

## The five blockers, with line references

### B1. The decode is a Blackwell instruction (`cvt.rn.f16x2.e2m1x2`)

`tools/fp4_moe.py:128-154` (and the same idea in `tools/fp4_linear.py`). The instruction is
sm_100a/sm_120a/sm_121a only; on SM70 it does not assemble, and there is no FP4 hardware path.
The whole decode has to be integer arithmetic.

**Done in this branch.** `tools/fp4_decode.py` is the specification -- fp16 exponent/mantissa
fields built with shifts and selects -- and `tools/test_fp4_decode.py` holds it bit-exact against
the reference and against `v41_ref.dequant_fp4_packed` for all 16 codes. `tools/fp4_decode_triton.py`
carries the same arithmetic into Triton and ships `python3 tools/fp4_decode_triton.py --check`,
which compares it on the GPU, code by code, before any of the real kernels are touched.

Two traps are pinned by tests because both produce a model that runs and is quietly wrong:

* the E2M1 grid is not linear in the code, and the mantissa bit is `mag & 1` **except for code 1**
  (0.5 is exactly `2**-1`, not `1.5 * 2**-1`): the naive form decodes it as 0.75;
* the block scale goes on the **accumulator**, not on the weight. `_chunk_dot` computes the fp32
  group partial and scales that; folding a scale of `2**14` into a weight of 6.0 leaves fp16's
  range. `dequant_fp4(scaled=False)` is the kernel-shaped variant.

### B2. bf16 (V100 has no bf16 hardware path)

12 files mention `bfloat16`/`bf16`, 239 occurrences together:

| file | sites | what changes |
|---|---:|---|
| `tools/v41_ref.py` | 46 | reference port; `DSV41_DTYPE` |
| `engine/model.py` | 26 | attention/CSA2/CED and the dense projections |
| `tools/fp8_linear.py` | 24 | dense FP8 linears |
| `tools/cb3_moe.py` | 21 | the 3-bit codebook MoE path |
| `engine/fastdecode.py` | 21 | the decode fast path |
| `tools/fp4_linear.py`, `tools/decode_attn.py` | 18 each | kernels |
| `tools/fp4_moe.py` | 12 | `assert x.dtype == torch.bfloat16`, `h.to(tl.bfloat16)`, the final cast |

This is the bulk of the remaining work and it is mechanical but not blind: fp16 has a smaller
exponent range than bf16, and the places that carry large magnitudes are exactly the ones that
need review. Precedent from this hardware family: 1Cat-vLLM runs V100 with `dtype="half"` and
needed a specific fix to "keep mHC finite under float16" (their PR #658), and sglang's V100 port
forces FP16 process-wide (`SGLANG_SM70_FORCE_FP16`). The relevant sites here are the mHC/HC
mixing and the Engram gate.

### B3. `tl.dot` and the launch configs

`tools/fp4_moe.py:405-407` carries GB10-tuned `(BN, num_warps, num_stages)` triples
(`_UP_CFG`/`_DOWN_CFG`) chosen against 48 SMs, 228 KB of shared memory per SM and ~273 GB/s of
LPDDR5X. A V100 SM has 64 KB of shared memory, 4 warp schedulers and 900 GB/s of HBM2, so those
numbers have to be re-swept rather than inherited. fp16 `tl.dot` on sm70 is expected to work
(the same stack runs Triton MoE/decode kernels on V100 elsewhere) but it is unverified here, which
is exactly why `fp4_decode_triton.py`'s check kernel uses `tl.sum` and not `tl.dot`: it isolates
the decode from that question.

### B4. The host stack is pinned to a GB10

`README.md:203` (`torch==2.13.0+cu130`), `README.md:195-196` (GB10 / sm_121a / CUDA 13 driver),
`README.md:239` (the image is arm64 only), `env.example:2,19`. The V100 host needs CUDA 12.x wheels
(the sglang V100 fork pins `torch==2.9.1+cu128`, 1Cat-vLLM validates 2.10.0+cu128) on x86_64.

### B5. The attention path is unverified on SM70

`engine/model.py` and `engine/fastdecode.py` implement CSA2/CED: the compressor, the C4 indexer,
the sinked softmax and the mHC mixing. Nothing in them is obviously Blackwell-only, but they are
Triton kernels written against a different SM, and no part of them has run on Volta. They are the
part of the port that most resembles a project rather than a sweep.

## What this branch already delivers

| artifact | state | validation |
|---|---|---|
| `tools/fp4_decode.py` | the portable decode, bit-exact (blocker B1) | `python3 tools/test_fp4_decode.py` -- 15 checks, **passing here** |
| `tools/fp4_cpu.c` + `build_fp4_cpu.sh` | fused decode+GEMV, AVX2/FMA, OpenMP | `tools/test_fp4_gemv_cpu.py` -- 21 checks, **passing here** |
| `tools/fp4_expert_cpu.py` | the RAM tier: arena, expert, routed MoE | same suite; agrees with `moe_forward_reference` |
| `engine/caps.py` | blocker B2's decision point: fp16 on sm_70, and the capability gates | `tools/test_caps.py` -- 28 checks, **passing here** |
| `tools/tier_plan.py` | the three-tier planner (VRAM/DDR4/NVMe) | `tools/test_tier_plan.py` -- 43 checks, **passing here** |
| `tools/bench_fp4_cpu.py` | the measured cost of the tier | numbers below |
| `tools/fp4_decode_triton.py` | the decode in Triton + a GPU check | **needs the V100**: `--check` |

107 checks across the four suites, all passing on a machine with no GPU.

Two findings from writing it that are worth more than the code:

* **PyTorch calls `omp_set_num_threads(1)` when it initialises** (its intra-op pool is its own, so
  it disables OpenMP parallelism to stop the two fighting). That is process-global: a tier using
  `#pragma omp parallel for` inherits **one thread** whenever torch has been imported, which in
  this engine is always. Measured: 6.5 ms per expert instead of 0.65 ms, a 10x loss that looks
  like nothing in a profiler. `fp4_cpu_set_threads` exists for that and `kernels()` calls it.
* **Duplicate top-k routes matter.** Top-k can hand the same expert two of a token's six slots;
  the T=1 fast path originally used the first routing weight only and silently dropped the other.
  `h` is linear in that weight, so folding the duplicates before the call is both correct and
  cheaper, and a test asserts the semantics rather than the implementation.

## Measured: the CPU expert tier

`bash tools/build_fp4_cpu.sh && python3 tools/bench_fp4_cpu.py`

Host: AMD Ryzen 9 5900X, 12 cores / 24 threads, AVX2+FMA (no AVX-512), 62 GB DDR4, torch 2.14 CPU.
Copy bandwidth (512 MiB, read+write): 43.7 GB/s.

| threads | ms per expert | GB/s | vs copy ceiling |
|---:|---:|---:|---:|
| 1 | 6.467 | 2.9 | 7% |
| 4 | 3.188 | 5.9 | 13% |
| 8 | 1.633 | 11.5 | 26% |
| **12** | **0.647** | **29.1** | **66%** |
| 24 | 0.846 | 22.2 | 51% |

Breakdown at 24 threads: w1 gate 0.253 ms, w3 up 0.250 ms, clamped SwiGLU 0.013 ms, w2 down
0.253 ms.

29.1 GB/s is 66% of what a `memcpy` achieves on this box, and the kernel is doing the decode on
the way past -- the fused form reads each stored byte once, where materialising to fp32 first
would touch roughly six times the bytes.

## The tier plan, from the planner

`python3 tools/tier_plan.py` builds this out of the repo's own memory terms (`tools/budget.py`),
the routing trace already in `results/`, and exactly one measured input: the CPU tier's rate. With
the Ryzen's 29.1 GB/s and a deliberately conservative 400 GB/s for the GPU expert path:

| tier | routed mass | GB/token | ms/token | tok/s alone |
|---|---:|---:|---:|---:|
| VRAM, 133 slots/layer | 0.758 | 3.42 | 8.5 | 117 |
| DDR4, 251 slots/layer | 0.242 | 1.09 | 37.5 | 26.6 |
| NVMe | 0.000 | 0.00 | 0.0 | - |
| **total** | 1.000 | 4.51 | 46.1 | **21.7** |

All 15,360 slots are resident, nothing is streamed, and the weakest layer covers 1.000 of its
routing. Serialised that is 21.7 tok/s; with the CPU expert work overlapped under the GPU's
attention, 26.6. If the host's CPU side reaches 60 GB/s -- a 16-core server board has more memory
channels than a desktop -- it becomes 37 serialised and 55 overlapped.

The planner's only calibration is the box this engine ships for. Run it with `--profile spark`
against the same trace and it predicts **2.05 tok/s with 1.18 GB of NVMe per generated token**,
where that box was measured at 2.64-2.71 tok/s and 0.92 GB; and **0.739 of the routed mass in the
arena**, where the repo reports a static coverage of 0.748 at 4,000 resident slots. Same regime,
same trace, nothing fitted. That is what makes the V100 numbers worth reading.

**Correction to an earlier estimate in this document.** A naive "35% of the routed work in VRAM,
65% on the CPU" split gives 10.7 tok/s. The trace says the hot set is far hotter than that: 133
slots per layer is 34.6% of the *experts* but **75.8% of the routing**. Ranking is what makes the
CPU tier cheap, which is also why `tools/budget.py`'s ranking rules matter here and not just on
the Spark.

Three levers, in the order they are worth pulling:

1. **The ranking that decides which experts sit in VRAM.** HBM2 at ~900 GB/s against DDR4 at
   29-60 GB/s is a 15-30x difference, and the trace machinery for it already exists
   (`engine/experts.py::rank_from_trace`, `tools/budget.py`). The 0.758 above is that lever
   already pulled with the repo's own trace.
2. **The `cb3` codebook layout for the RAM tier** (14.45 MB per slot instead of 18.80): measured
   here, it moves 0.834 of the routed mass into VRAM and takes the plan to 36.9 serialised / 50.4
   overlapped, at the quality cost the repo already documents.
3. **AVX-512 or AMX** if the host has it. The current kernel is AVX2 because that is what the
   machine that wrote it has; at one thread it is compute-bound (2.9 GB/s) and at twelve it is at
   66% of the copy ceiling, so the headroom is real but on the compute side.

## What to run on the V100 host, in order

```bash
# 0. the checkpoint is 510 GB; the Engram tables alone are the two 101 GB shards
MODEL_DIR=/path/to/DeepSeek-V4.1-Flash

# 1. does the portable decode survive the GPU? (this is the blocker B1 answer)
python3 tools/fp4_decode_triton.py --check

# 2. the CPU tier, no GPU needed, but run it here for this host's real number
bash tools/build_fp4_cpu.sh
python3 tools/test_fp4_gemv_cpu.py
python3 tools/bench_fp4_cpu.py --reps 9        # <- the tier planner's input

# 3. the plan, with this host's measured CPU rate and the GPU rate measured in step 4
python3 tools/test_caps.py
python3 tools/test_tier_plan.py
python3 tools/tier_plan.py --cpu-gbs <from step 2>

# 4. the GPU expert kernels once the portable decode is in: this prints the effective GB/s
MODEL_DIR=/path/to/DeepSeek-V4.1-Flash python3 tools/test_fp4_moe.py
python3 tools/tier_plan.py --cpu-gbs <step 2> --gpu-gbs <step 4>
```

If step 1 fails, the failure is a specific code or a specific K position, not a vague numerical
complaint, and it tells us whether the correction to make is in `_decode_codes` or in the
activation's stride-2 indexing.

## Honest status

* The decode, the CPU tier, the capability/dtype layer and the planner are **validated**: 107
  checks across four suites, all passing on a machine with no GPU. The arithmetic is bit-exact
  against the reference and against fp64, and the planner reproduces the measured single-box
  regime without fitting anything.
* The Triton decode is **written and unrun**. No driver, no `nvcc`, and no 510 GB of disk on the
  box this was written on.
* The bf16 -> fp16 sweep (B2) and the attention path (B5) are **not started**. They are the two
  pieces of real work left. `engine/caps.py` gives the sweep its single decision point
  (`DSV41_DTYPE`, resolving to fp16 on sm_70) and `tools/test_caps.py` guards the rule, but the
  239 call sites still have to follow it, and the ones carrying large magnitudes need review
  rather than a find-and-replace.
* The tier plan is arithmetic from a measured CPU rate plus the repo's own memory constants. It is
  a plan, not a benchmark of the assembled system: the overlap between the CPU expert work and the
  GPU attention work is a scheduling property the runtime still has to deliver.
