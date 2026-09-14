# The memory budget

Every term in `./tune.sh`'s budget panel, where each number comes from, and the two gates a
configuration has to pass. The design this arithmetic serves is in
[`docs/architecture.md`](architecture.md); the screen is in [`docs/tune.md`](tune.md) and
[`docs/tune-reference.md`](tune-reference.md); the sharp edges are in
[`docs/gotchas.md`](gotchas.md).

The model is `tools/budget.py`. It holds no torch, loads nothing and answers in a millisecond, which
is the point: the engine answers the same question by loading for three minutes first.

## What is being budgeted

Memory on a GB10 is unified. The GPU allocator and the page cache draw on one pool, and
`torch.cuda.mem_get_info()` counts the page cache as used — right after a checkpoint download it
reported 32.0 GB free on a box with 99.9 GiB of `MemAvailable`. `nvidia-smi` answers `[N/A]` for
every memory field. So the honest number, for the tool and for the engine alike, is `MemAvailable`
from `/proc/meminfo`, and the engine takes the larger of that and `mem_get_info()`.

There is no OOM-kill moment to catch: push `MemAvailable` towards zero on this hardware and the
machine stops being reachable. Everything below is arranged around not doing that.

## The terms

| term | value | where it comes from |
|---|---|---|
| routed experts | 40 layers x 384 = 15,360 | `config.json` `n_routed_experts`, 40 layers |
| activated per token | 6 per layer | `config.json` `n_activated_experts` |
| expert slot, `cb3` | 14,454,784 B (14.45 MB) | `CB3_BYTES_PER_SLOT` in `tools/cb3_moe.py` |
| expert slot, `fp4` | 18,800,640 B (18.80 MB) | `EXPERT_BYTES` in `engine/experts.py`, `3 x (2304x2560 + 2304x160)` |
| kept experts | `ceil(keep x 384) x 40`, never fewer than 6 per layer | `build_keep_masks` in `engine/v41_engine.py` |
| transient ring | 8 slots as the tool writes it; 400 is the engine's default | `TRANSIENT_SLOTS`, `engine/experts.py` |
| expert arena | `(kept + ring) x slot` | |
| dense weights | 7.61 GB | measured at load, `7.09 GiB allocated after weights`, 2026-09-12 |
| drafter experts | `3 x 128 x 18,800,640` = 7.22 GB | the DSpark MTP blocks, `engine/v41_engine.py` |
| KV + indexer cache | `max_seq x 3,200 B` | `Caches` in `engine/model.py` |
| sliding-window rings | `43 x RING x 512 x 2` = 180,355,072 B at `RING` 4,096 | same, independent of `max_seq`, but `RING` follows the prefill chunk |
| warm-start pack scratch | 3 GB for `cb3`, 1 GB for `fp4` | the 3-bit packer's GPU buffers |
| keep-free floor | 6 GB as the tool writes it; 20 GB is the engine's default | `KEEP_FREE_GB` |
| prefill chunk | 7.2 GB at the default 2,048-token chunk, plus 15.1 KB per token of context — **both** scale with the chunk | measured; itemised in [Where a prefill chunk's memory goes](#where-a-prefill-chunks-memory-goes) |
| prefill gather format | `DSV41_PREFILL_KV_FP8=1` takes 960 KB a token off the row above | `PREFILL_KV_FP8_SAVED_PER_TOKEN`, same section |
| prefill unpack cache | 0 by default; whatever `DSV41_PREFILL_UNPACK_CACHE_GB` asks for, in whole 18.80 MB experts | see below |

Everything except the last four rows stays resident for the whole run.

### Kept experts

`PRUNE_KEEP` is a fraction of each layer's 384 experts, and the engine rounds it up:
`ceil(keep x 384)` experts in every layer, never fewer than six. At `PRUNE_KEEP=0.39` that is 150 per
layer and 6,000 in all, 39.1 % of the 15,360.

The tool uses the same rounding, so the count on screen is the count the engine will build.

### The transient ring

Prefill touches nearly every expert of a layer — 370 to 381 of 384, measured at layer 0 — so a
prefill miss must not go through the LRU or every prompt would evict the hot set. Misses go to a
small ring of slots outside it instead, and the arena has to hold **the kept set plus that ring**:
`engine/experts.py` sets `lru_slots = n_slots - transient_slots`, and `warm_start` fills only
`lru_slots`. A kept set larger than that has its tail left cold, streaming from NVMe on every step,
with nothing but the tok/s to say so.

The size of the ring depends on which mode the engine is in, and the two answers are far apart:

* **Streaming mode**, where experts are read on demand, needs at least 384 slots, because below that
  the ring wraps inside a single layer and the engine computes that layer with the wrong experts —
  no exception and no warning, just an 0.88 relative error in the output. The default is 400 and it
  must not be lowered.
* **Pruned all-resident mode**, which is what `./tune.sh` configures, never misses during prefill,
  because every routable expert is already in the LRU. Eight slots is enough, and `engine/experts.py`
  asserts that as the hard minimum.

The difference is not small: 392 extra slots is 5.7 GB of arena, which is more than a full step of
the keep slider. Sizing an arena against 8 and running with 400 leaves 392 kept experts outside the
LRU. That is why `./tune.sh` writes `TRANSIENT_SLOTS` into `.env` alongside `ARENA_GB` — the two
numbers are only correct together.

### The arena

`(kept + ring) x slot_bytes`, written into `.env` as `ARENA_GB` rounded up to a whole GB. The engine
divides it back into slots by flooring, and `tools/test_budget.py` checks at every keep step and
every ring size that the round trip never leaves a kept expert outside the arena.

Format matters more than anything else on the screen. At `PRUNE_KEEP=0.39` the same 6,008 slots are
86.8 GB in `cb3` and 113.0 GB in `fp4` — the difference between serving on this box and not fitting
on it at all.

### Dense weights

Everything that is not a routed expert: attention, the shared experts, the embeddings, the LM head
and the Engram projections. This is measured at load rather than derived, and it moves with two
switches:

| `DSV41_DENSE_FP4` | `DSV41_HEAD_FMT` | dense weights |
|---|---|---|
| `attn,wo_a` | `fp8` | 7.61 GB |
| `attn,wo_a` | `bf16` | 8.94 GB |
| unset | `fp8` | 18.1 GB |
| unset | `bf16` | 19.4 GB |

`env.example` ships the first row, and the budget panel always assumes it. If those two switches are
changed in `.env`, the panel understates resident memory by up to 11.8 GB and the verdict cannot be
trusted.

### Drafter experts

The DSpark drafter has its own 384 experts — 3 MTP blocks of 128 — and they are loaded into their
own arena, fully resident, always in the FP4 layout whatever `EXPERT_FORMAT` says. 7.22 GB, fixed,
for every configuration.

They are also the largest term the engine's own pre-flight does not know about, because that check
runs before they are allocated.

### KV cache

Exact, and much smaller than people expect. `engine/model.py` allocates the caches for `MAX_SEQ`
up front. Per token, for each of the four `kv_source_layers` — 2, 8, 14 and 20 — one compressed-KV
row of `head_dim` 512 and one index row of `index_head_dim` 128, both bf16, at that layer's
compression ratio (2, 2, 2 and 1):

```
(3 x 1/2 + 1) x (512 + 128) x 2 = 3,200 bytes per token
```

The sliding-window rings are a separate, constant term: `RING` 4096 positions of `head_dim` 512 in
bf16 for 40 layers plus 3 MTP blocks, 180,355,072 B, the same at any context length. The panel's
`KV cache` row is the two added.

| `MAX_SEQ` | rows | + window rings | panel |
|---|---|---|---|
| 4,096 | 0.013 GB | 0.193 GB | 193 MB |
| 32,768 | 0.105 GB | 0.285 GB | 285 MB |
| 131,072 | 0.419 GB | 0.600 GB | 600 MB |
| 262,144 | 0.839 GB | 1.019 GB | 1.0 GB |
| 1,048,576 | 3.355 GB | 3.536 GB | — |

The cache is never what limits the context window on this box. Prefill is. The whole range the tool
offers, 4k to 256k, is 0.83 GB — less than a fifth of one 2-point step of the keep slider. The tool
therefore marks the longest context that has actually been loaded and prefilled from, rather than
predicting a ceiling from the cache arithmetic. On 2026-09-12 that length is **131,072**
(`tools/budget.py` `VALIDATED_MAX_SEQ`).

It read 32,768 before that, on the strength of a 64k attempt that tripped the memory watchdog at an
arena with room for the cache many times over. **That anecdote is superseded**: the watchdog was
firing on the prefill chunk's own growth with context — 15.1 KB a token, which is why it is a term
of the gate now — not on anything the cache arithmetic missed, and 131,072 has since been prefilled
and measured. The indexer's score tiles do still grow with the compressed cache and that term is
still not characterised on its own, which is why the marked length is a run that happened rather
than a ceiling derived from the formula.

### The prefill unpack cache

`DSV41_PREFILL_UNPACK_CACHE_GB`, off by default, and only meaningful with `EXPERT_FORMAT=cb3`.

A prefill-sized MoE call cannot read a 3-bit expert. It unpacks the experts it needs back into
packed FP4 — bit-exact, the CB3 codes are a subset of the FP4 grid — a batch at a time into a
0.6 GB scratch arena, runs the ordinary FP4 kernel, and throws the unpack away. That is what
prefill actually costs on this recipe (`NOTES.md` 2026-09-12), and the waste is structural:
**chunk 1 of a prompt unpacks exactly the experts chunk 0 already unpacked.** A 7,000-token prompt
is four chunks and pays the same unpack four times.

This budget buys a second region of that same FP4 arena which is *not* overwritten between the
chunks of one prompt. What it can buy is fixed arithmetic:

```
one expert          18,800,640 B                      (the FP4 slot, unchanged)
one layer           ceil(PRUNE_KEEP x 384) experts    2.61 GB at keep 0.36
                                                      2.90 GB at keep 0.40
                                                      7.22 GB with nothing pruned
one chunk           layers 0..20                      21 layers, 54.9 GB at keep 0.36
```

Nothing on this box holds 54.9 GB of cache, so the cache has to choose which layers it keeps — and
**an LRU is the worst possible rule here**. Prefill is a sequential scan: by the time chunk 1 asks
for layer 0 again, LRU has just evicted it as the oldest entry, and *every* reference misses.
`tools/unpack_cache.py` therefore admits on first touch and never evicts inside a request. Because
the chunk order is deterministic the cache fills with layers 0..k-1 plus a prefix of layer k, those
layers hit on every later chunk, and the layers past the window miss without disturbing them. The
hit rate is then simply

```
cached layers / layers per chunk
```

so at a 6 GB budget (319 experts, 2.3 layers of 21) about 8 % of a four-chunk prompt's expert
lookups — chunk 0 cannot hit, and admission is switched off for the last chunk and the decoder
replay because nothing admitted there is ever read back. `tools/test_unpack_cache.py` replays that
trace and checks each of these properties, LRU's zero included, with no GPU.

**Where it sits in the gates.** It is not transient — it is allocated at start-up and lives as long
as the process — but it is allocated *on top of* everything the resident total names, so
`tools/budget.py` charges it to `prefill_bytes` and it lands in both gates at once. The engine adds
it to its own pre-flight in the same place, and **clamps** rather than refuses: an arena that
serves must not stop serving because a cache budget was set too high, and the log says what was
asked for and what was taken.

At the shipped `cb3` sizes on a 118.6 GB box that is the difference between a configuration and no
configuration:

| keep | arena | free after load | + 6 GB cache | verdict with the cache |
|---|---|---|---|---|
| 0.36 | 80.5 GB | 19.4 GB | needs 16.2 GB | fits |
| 0.39 | 86.8 GB | 13.0 GB | needs 16.2 GB | **will not load** |

So the cache is not free headroom: at the shipped keep of 0.39 it has to be paid for with a step of
the keep slider, and whether that trade is worth taking is a measurement, not an argument —
`tools/verify_prefill_cache.sh` is the A/B that settles it, and the records go under
`results/prefill/`.

**What it cannot do.** In streaming mode (no keep-set, a transient ring recycling arena slots) every
entry is invalidated before it can be reused and the cache correctly degenerates to a no-op. With
`EXPERT_FORMAT=fp4` there is no unpack at all and the variable is ignored with a log line.

### What the panel leaves out

* The **Engram row cache**, capped at 200,000 rows of 264 bytes per table across two tables, so
  105.6 MB at most. It is also the only thing still read from NVMe once a keep-set is fully
  resident: 24 rows per token per table, about 13 KB a token, against zero bytes of expert weights.
* The **pack scratch** and the **prefill chunk**, which are transient rather than resident. They are
  not in the resident total but they are in the gates, which is where they belong.

## Where a prefill chunk's memory goes

The 7.2 GB row above is a fit to a measurement, not a sum of tensors, and until 2026-09-15 nobody
had written down what it is made of. This section does that: every allocation the prefill path makes
for one chunk, with its dtype and its size at the shipped `DSV41_PREFILL_CHUNK=2048`, taken by
reading `engine/model.py`, `tools/v41_ref.py` and the MoE kernels rather than by measuring.

Shapes come from the checkpoint: `dim` 5,120, `hc_mult` 4, `n_heads` 64, `head_dim` 512,
`q_lora_rank` 1,280, `o_groups` 8 × `o_lora_rank` 1,024, `window_size` 128, `index_topk` 512,
`index_n_heads` 32 × `index_head_dim` 128, `moe_inter_dim` 2,304, six activated experts.

### Per token of the chunk

One layer at a time — the chunk loop holds one layer's working set, not forty. `T` is the chunk,
2,048 in the right-hand column.

| where | tensor | shape | dtype | B / token | at T=2,048 |
|---|---|---|---|---:|---:|
| carried | `h` (the residual stream) | `[T, 4, 5120]` | bf16 | 40,960 | 83.9 MB |
| carried | `sh.topk` | `[T, 512]` | int64 | 4,096 | 8.4 MB |
| attention | `qr` after `wq_a` | `[T, 1280]` | bf16 | 2,560 | 5.2 MB |
| attention | `q` after `wq_b` + RoPE | `[T, 64, 512]` | bf16 | 65,536 | 134.2 MB |
| attention | `kv`, the new window row | `[T, 512]` | bf16 | 1,024 | 2.1 MB |
| attention | `wpos` | `[T, 128]` | int64 | 1,024 | 2.1 MB |
| attention | **`wkv`, the gathered window** | `[T, 128, 512]` | bf16 | **131,072** | **268.4 MB** |
| attention | **`ckv_rows`, the gathered compressed KV** | `[T, 512, 512]` | bf16 | **524,288** | **1,073.7 MB** |
| attention | **`kv_all`, the concatenation of both** | `[T, 640, 512]` | bf16 | **655,360** | **1,342.2 MB** |
| attention | the three masks | `[T, 128/512/640]` | bool | 1,280 | 2.6 MB |
| attention | `outs`, the softmax rows, and their `cat` | `[T, 64, 512]` ×2 | fp32 | 262,144 | 536.9 MB |
| attention | `o` after the inverse RoPE | `[T, 64, 512]` | bf16 | 65,536 | 134.2 MB |
| attention | `wo_a` output | `[T, 8, 1024]` | bf16 | 16,384 | 33.6 MB |
| attention | the block's output | `[T, 5120]` | bf16 | 10,240 | 21.0 MB |
| indexer (8 layers) | `q` | `[T, 32, 128]` | bf16 | 8,192 | 16.8 MB |
| indexer (8 layers) | the top-k indices | `[T, 512]` | int64 | 4,096 | 8.4 MB |
| hyper-connections | `hc_mixes` flattened input | `[T, 20480]` | fp32 | 81,920 | 167.8 MB |
| hyper-connections | `hc_pre` product | `[T, 4, 5120]` | fp32 | 81,920 | 167.8 MB |
| hyper-connections | `hc_post` `mixed`, `residual.float()`, `y` | `[T, 4, 5120]` ×3 | fp32 | 245,760 | 503.3 MB |
| MoE | `h`, the SwiGLU intermediate | `[6T, 2304]` | bf16 | 27,648 | 56.6 MB |
| MoE | `parts`, one row per `(token, k)` | `[6T, 5120]` | fp32 | 122,880 | 251.7 MB |
| MoE | routed + shared + their sum | `[T, 5120]` ×3 | fp32 | 61,440 | 125.8 MB |
| MoE | router scores and logits | `[T, 384]` ×2 | fp32 | 3,072 | 6.3 MB |

The softmax's own fp32 tiles are missing from the table on purpose: `_softmax_attn` runs in fixed
64-query tiles, so its scratch — 83.9 MB for `kvt`, 8.4 MB for `qt`, three score tiles of 10.5 MB —
is about 133 MB **whatever the chunk is**, and does not belong in a per-token column.

### Per token of the context

These are the indexer's, and they are shaped `[chunk × compressed positions]` — so they grow with
the context **and** with the chunk. The column below is the rate at T=2,048; at T=4,096 every one of
them doubles. Worst case is a ratio-1 layer (20 and up), where the compressed cache has one row per
position.

| where | tensor | shape | dtype | B / context token | at 32k |
|---|---|---|---|---:|---:|
| `_indexer` | `score` | `[T, n_c]` | fp32 | 8,192 | 268.4 MB |
| `_select_candidates` (layer 20) | the `-inf`-padded copy of it | `[T, n_c]` | fp32 | 8,192 | 268.4 MB |
| `_select_candidates` | the per-block maxima | `[T, n_c/8]` | fp32 | 1,024 | 33.6 MB |
| carried, layers 20-39 | `sh.candidates` | `[T, n_c]` | bool | 2,048 | 67.1 MB |
| layers 24, 28, 32, 36 | `~candidates` and the masked copy of `score` | `[T, n_c]` | bool + fp32 | 10,240 | 335.5 MB |

`sh.candidates` is the one that is live for the rest of the chunk; the rest are transient inside one
layer. Added up the way the engine meets them, the fitted 15.1 KB per context token is the right
order and slightly conservative, which is the direction it was chosen in.

### So where is the peak, and where is the 5 MB a token

Two moments compete, and at the shipped configuration they are not close:

| moment | live bytes at T=2,048, 32k context |
|---|---:|
| **`kv_all = cat([wkv, ckv_rows])` in `attention`** | **3.01 GB** |
| `_select_candidates` at layer 20 | ~0.80 GB |
| `hc_post` | ~0.67 GB |
| the MoE kernel | ~0.61 GB |

The attention gather is the peak by a factor of four, and it is the peak because the same values are
held three times: the window gather, the compressed gather, and the concatenation of the two, all
bf16, all alive at once — 1,280 KB a token of the 1,470 KB the whole moment costs.

That also settles a question the 7.2 GB row could not answer. **The line items do not add up to
5 MB a token. They add up to 1.47 MB a token** — the same order as the ~1.5 MB a token reported for
this model on SGLang, which is the figure this path was being measured against. The rest of the
fitted 3.5 MB a token, and of the ~5 MB a token the engine's pre-flight comment quotes, is **not a
tensor**. It is the caching allocator's high-water mark: a chunk cycles
through a dozen differently-shaped blocks per layer (the fp32 softmax tiles, `parts`, the three fp32
copies inside `hc_post`), the allocator keeps every size class it has ever served, and on this box
`MemAvailable` sees the reservation and not the live set.

Two consequences follow, and they are the whole reason this section exists:

* Cutting the largest live tensor also cuts the largest block class the allocator has to retain, so
  it pays twice.
* Whatever is left over after that is an allocator problem, not a tensor problem, and the instrument
  for it is `torch.cuda.memory_allocated()` against `memory_reserved()` during a chunk — not another
  pass over the shapes above. That measurement has not been taken.

### `DSV41_PREFILL_KV_FP8=1`

The three bold rows are what the flag addresses. With it on, `attention` stops building two bf16
gathers and concatenating them and fills **one** `[T, window_size + index_topk, head_dim]` fp8
buffer directly, 128 rows at a time (`Model._gather_kv_fp8`):

| | bf16 (default) | fp8 e4m3 |
|---|---:|---:|
| gathered window | 128 KB / token | — |
| gathered compressed KV | 512 KB / token | — |
| their concatenation | 640 KB / token | 320 KB / token |
| bf16 scratch, 128 rows, independent of T | — | 84 MB |
| **the attention peak at T=2,048, 32k** | **3.01 GB** | **0.75 GB** |

So the gather stops being the peak at all: at 32k the binding moment becomes `_select_candidates`
at ~0.8 GB, which is a context term rather than a chunk term. `tools/budget.py` claims only the
live-byte saving — 960 KB a token, `PREFILL_KV_FP8_SAVED_PER_TOKEN` — and not the allocator
multiplier, because being pessimistic here costs expert slots and being optimistic costs the
process.

**e4m3 and not e5m2.** Three mantissa bits against two, so round-to-nearest is within 2^-4 = 6.25 %
of a value instead of 2^-3 = 12.5 %, and the error lands in an attention score. The range e5m2
would buy is range these tensors do not use: both are the output of an rmsnorm (plus RoPE on the
last 64 dims), so O(1), and e4m3's ±448 is nine binades above that. Values are clamped to ±448
before the cast because `float8_e4m3fn` has no infinity and overflows to NaN, and one NaN in a
gathered row takes that query's whole softmax with it. It is also the format the reference
implementation uses for exactly these tensors — `engine/model.py`'s docstring lists "window KV and
compressed KV caches are kept in bf16 instead of fp8 / FP4-E4M3" as a deviation *towards* more
precision, and this flag gives that deviation back for the prefill working copy only.

**What it does not touch.** The window ring (`Caches.win`), the compressed KV cache (`Caches.ckv`)
and the index cache (`Caches.ik`) stay bf16, so every byte decode later reads is in the format it is
in today, and the decode path is not on this branch at all. The bounded replay is excluded as well
(`T > window_size` guards it): 128 queries are 42 MB of gathered KV, there is nothing to save, and
it is the pass that produces the prompt's final logits.

**The quality risk is real and is not bounded by the round trip.** The rounding changes an attention
output, the attention output changes the residual stream, and layers 21-39 write their window KV
from that residual stream during the prompt — so prefill precision does reach the KV decode reads,
by a longer path than the caches. The round trip itself is checked in
`engine/test_prefill_kv_fp8.py` (at most 6.25 % relative on a value; well under 1 % on an attention
output, once the softmax has reduced over 512 dimensions). The only instrument for the rest is a
profile gate, which is why `tools/verify_prefill_fp8.sh` runs one.

### A 4,096-token chunk

Fewer chunks is the point: a chunk unpacks every resident expert of every layer once, so halving
the number of chunks halves the unpack passes — and, on a streaming configuration, halves the NVMe
traffic as well (`NOTES.md` S.2 measured 512 → 2,048 as a 52 % cut in prefill time). Three things
have to be true for 4,096:

1. **The ring has to grow.** The window gather runs *after* the whole chunk is written into the
   ring, so `DSV41_RING` must exceed `window_size + chunk`; at chunk 4,096 the shipped 4,096-slot
   ring wraps inside one chunk and the first queries silently read what the last ones wrote. The
   default now follows the chunk — `max(4096, chunk + 512)` — and `Caches` refuses a ring that is
   too short instead of computing the wrong answer. 4,608 slots is 202.9 MB against 180.4, i.e.
   22.5 MB and about two expert slots.
2. **The budget has to price it as a 4,096-token chunk in both terms.** `prefill_bytes` used to add
   a context term that did not move with the chunk; it does now, in `tools/budget.py` and in
   `engine/v41_engine.py` alike. At 32k the reserve goes 7.7 GB → 15.4 GB in bf16, and → 11.4 GB
   with fp8.
3. **It has to stay on a tile boundary.** `MM_TILE` is 16 rows and `ATTN_TILE` 64, and the last tile
   of every GEMM is padded, so a chunk that overruns one pays about 30 % more iteration time for
   rows that are only padding. 4,096 is a multiple of 128; so is 2,048; so is 512.

Point 2 is why a 4,096-token chunk is not free even with fp8: 11.4 GB against the 7.7 GB the shipped
configuration reserves is still 3.7 GB less free memory — about 250 expert slots at `cb3` — which is
affordable at keep 0.36 and is not at keep 0.40.

None of the three numbers in this subsection has been measured on the box yet.
`tools/verify_prefill_fp8.sh` is the run that does it: baseline, fp8, fp8 at chunk 4,096, one
identical ~7,000-token prompt each, with the low-water mark of `MemAvailable` sampled throughout and
a profile gate on the last one.

## The two gates

A configuration has to pass two separate checks, and they are not the same check.

### Gate 1 — the engine's own pre-flight

`engine/v41_engine.py` refuses to start when

```
arena + pack scratch + prefill unpack cache + floor  >  MemAvailable
```

measured **after the dense weights are already resident**, with `floor = max(KEEP_FREE_GB, one
prefill chunk)`. The tool reproduces it against `MemAvailable` as it is now, before the weights are
loaded, so the dense term is explicit:

```
room to launch = MemAvailable − (arena + pack scratch + dense weights + prefill unpack cache
                                 + max(keep-free floor, one prefill chunk + the 2.5 GB watchdog floor))
```

That is the `room to launch` row (`tools/budget.py`, `Plan.launch_need`). It takes the same `max()`
the engine takes, so the row is the engine's own margin and not a more generous reading of it.

**Corrected 2026-09-12.** This paragraph used to say the row took the keep-free floor alone and was
therefore 4.2 GB more generous than the engine at `KEEP_FREE_GB=6`. That is no longer the code: the
`max()` is mirrored exactly, and the 4.2 GB discrepancy it described does not exist. What still
holds is the reason it mattered — pin an arena by hand and leave the engine's 20 GB default floor in
place, and gate 1 rather than gate 2 becomes the binding check.

### Gate 2 — a prefill chunk still fits

Everything resident, subtracted from what the box has:

```
resident        = arena + dense weights + drafter experts + KV cache
free after load = MemAvailable − resident
```

and that has to leave room for one prefill chunk. At the default 2,048-token chunk that is **7.2 GB**,
plus **15.1 KB for every token of context**: 7.5 GB at 32k, 9.2 GB at 128k, 11.2 GB at 256k. Measured, not
derived, and the measurement is the reason the gate exists. Both terms are proportional to
`DSV41_PREFILL_CHUNK` and the first one moves with `DSV41_PREFILL_KV_FP8`; what is inside them is
itemised in [Where a prefill chunk's memory goes](#where-a-prefill-chunks-memory-goes).

| arena | free after load | what happened |
|---|---|---|
| 98 GB | 5.5 GB | loaded, logged `ready`, and the memory watchdog killed it on the **first** request |
| 87 GB | 16.5 GB | served two 2,400-token generations |

Same box, same `.env` except the keep fraction, `MemAvailable` 111.0 GB with the dense weights
resident, 2026-09-12:

```
FATAL: host MemAvailable 0.4 GB stayed below the 2.5 GB floor for 3.0 s
```

**Gate 1 accepts both.** It runs before the drafter experts, before the cache and before anything
has prefilled, so it cannot see the 7.2 GB of drafter and the chunk cost that turn 98 GB from a
configuration that starts into one that dies. That asymmetry is the single most useful thing the
tool does: it applies gate 2 and refuses configurations the engine would have accepted.

### The verdict

```
will not load   room to launch < 0  or  free after load < one prefill chunk
tight           room to launch < 3 GB  or  free after load < that chunk + 3 GB
fits            otherwise
```

### A third, coarser gate

Before either of them, `start.sh` and the container entrypoint refuse to start at all below
`MIN_FREE_GIB` (90 GiB of `MemAvailable`), and name the processes and containers holding the pool.
That one is about whether the box is free, not about whether the configuration is sized right.

## The ceiling

The largest arena that both starts and survives a prefill chunk is the smaller of the two gates:

```
max_arena = min( MemAvailable − pack scratch − unpack cache − dense − keep-free floor,
                 MemAvailable − dense − drafter − KV − one prefill chunk − unpack cache )
max_keep  = (max_arena / slot_bytes − 8) / 15,360
```

With `cb3` experts and `MemAvailable` at 118.6 GB — the box before the dense weights, the same
reading the 111.0 GB above was taken after — that is a 93.2 GB arena and **about 42 % of the routed
experts**. It is not a constant: it moves with `MemAvailable` while the screen is open, and it is
lower in `fp4` (32 %) because each slot is 30 % larger.

The shipped default sits at 39 %, below the ceiling on purpose. `RESULTS.md` §4.3 was taken at 44 %,
and that configuration does not reliably serve.

## The ladder

`cb3` experts, an 8-slot ring, `MAX_SEQ=32768`, `MemAvailable` 118.6 GB:

| keep | per layer | kept | arena | resident | free after load | verdict |
|---|---|---|---|---|---|---|
| 25 % | 96 | 3,840 | 55.6 GB | 70.7 GB | 47.9 GB | fits |
| 32 % | 123 | 4,920 | 71.2 GB | 86.3 GB | 32.3 GB | fits |
| 39 % | 150 | 6,000 | 86.8 GB | 102.0 GB | 16.6 GB | fits |
| 40 % | 154 | 6,160 | 89.2 GB | 104.3 GB | 14.3 GB | fits |
| 41 % | 158 | 6,320 | 91.5 GB | 106.6 GB | 12.0 GB | tight |
| 42 % | 162 | 6,480 | 93.8 GB | 108.9 GB | 9.7 GB | will not load |
| 44 % | 169 | 6,760 | 97.8 GB | 112.9 GB | 5.7 GB | will not load |
| 51 % | 196 | 7,840 | 113.4 GB | 128.6 GB | −10.0 GB | will not load |

The 44 % row is the configuration that was killed on its first request, and the 5.7 GB here is the
5.5 GB that was measured. The 39 % row is the shipped default, and its 16.6 GB is the 16.5 GB that
served.

Reproduce any row without a terminal:

```bash
./tune.sh --keep 0.42 --print
```

## Asking the ladder instead of reading it

The ceiling above is a function of `MAX_SEQ`, because the KV cache and the prefill term both are.
`tools/keep_for_context.py` is that question asked directly — the largest **step** of the ladder
whose plan clears both gates, which is not the continuous `max_keep` above: the engine rounds the
per-layer count up (`ceil(keep x 384)`), so a plan at the continuous ceiling is already over it.

```bash
$ python3 tools/keep_for_context.py --max-seq 262144
PRUNE_KEEP=auto -> 0.36 for MAX_SEQ=262144 (fits with 3.3 GB spare)

$ python3 tools/keep_for_context.py --max-seq 262144 --bare
0.36
$ python3 tools/keep_for_context.py --max-seq 262144 --arena
81
```

On a 121 GiB box, reading `MemAvailable` as the 117.0 GB stand-in this document uses elsewhere:

| `MAX_SEQ` | KV | prefill reserve | largest step that fits | arena |
|---|---|---|---|---|
| 32k | 0.3 GB | 7.7 GB | 38 % | 85 GB |
| 64k | 0.4 GB | 8.2 GB | 38 % | 85 GB |
| 128k | 0.6 GB | 9.2 GB | 38 % | 85 GB |
| 256k | 1.0 GB | 11.3 GB | 36 % | 81 GB |

The 256k row is the one the record measured: 0.36 with an 81 GB arena served a filled 256k
context, and 0.40 with 89.2 GB served short prompts and was killed 582 s into a 195k-token
prefill. The spare it reports is the tighter of the two gates, so a row that says `fits with 3.3
GB spare` has 3.3 GB on the gate that binds and more on the other.

`PRUNE_KEEP=auto` in `.env` is this call, made by `./start.sh` before it launches anything; it
then sizes `ARENA_GB` from the answer when none is pinned, because the engine's own automatic
sizing takes 82 % of what is free and that is more arena than a filled context can afford. A
pinned `ARENA_GB` caps the answer instead — the kept set has to fit inside it, less the transient
ring — and when the pinned arena is itself too big for the context, the tool says so rather than
walking the ladder down, since the keep fraction is not what is over budget. Below 36 % it still
answers and warns: no keep-set that small has ever been through a generation gate here.

## Where this is checked

```bash
python3 tools/test_budget.py
python3 tools/test_keep_for_context.py
python3 tools/test_unpack_cache.py
```

`test_unpack_cache.py` covers the unpack cache: the policy replayed against the access pattern
prefill really has (and the zero an LRU would score on it) with no torch at all, then — on a box
that has CUDA and the checkpoint — that the cache changes no number, bit for bit, against the same
call with it off.

`test_budget.py` cross-checks the slot sizes against the kernel's own constant, the KV formula against two measured
lengths, the launch gate against an arena the box accepted and one it did not, the `ARENA_GB`
rounding at every keep step and ring size, and — by reading `engine/v41_engine.py` with a regular
expression — that the engine still reserves a prefill chunk at the same rate this model assumes, by
the same chunk factor, with the same fp8 saving. If the two ever drift, the tool would start
advising configurations the engine refuses, or worse, ones it accepts and the watchdog then kills.
It also checks that a 4,096-token chunk is priced in both terms, that the chunk stays on a tile
boundary, and that `engine/model.py` still derives the ring default from the chunk and refuses a
ring that is too short for it.

```bash
python3 engine/test_prefill_kv_fp8.py
```

The fp8 gather itself: the e4m3 round trip against its 2^-4 bound, that `_gather_kv_fp8` is
bit-identical to quantising what the bf16 path gathers at every chunk length and gather tile, what
the rounding does to an attention output, and that an out-of-range value saturates instead of
becoming a NaN. Needs CUDA; skips itself with exit 0 anywhere else.

```bash
./tools/verify_prefill_fp8.sh
```

On the serving box: baseline, fp8, and fp8 at a 4,096-token chunk, one identical ~7,000-token
prompt each, with prefill tok/s, TTFT and the low-water mark of `MemAvailable` per run, then a
profile gate on the last one. This is the run that turns the arithmetic above into a measurement.
