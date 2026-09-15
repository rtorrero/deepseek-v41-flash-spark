# Changelog

What a version means here: this repository is not a library and nothing imports it. What
you depend on is **the defaults the recipe ships and the measurements taken on them**, so
a release is a measurement epoch — the configuration as it stood, and the figures that
belong to it.

- **MAJOR** — the measurement basis changes (different hardware, model or checkpoint).
- **MINOR** — a shipped default changes, or the recipe gains a capability. Your numbers move.
- **PATCH** — documentation, corrections, tooling. Your numbers do not move.

`./run.sh` and the container print the version they were launched from, and the image is
tagged with it: `ghcr.io/<owner>/deepseek-v41-flash-spark:<version>`. A `-wip` version means
exactly what it says: the defaults are not settled and the measurements are incomplete.

## 0.1.0-wip — 2026-09-10

**Work in progress, not a release.** The recipe serves, and the numbers it served at are
in [`RESULTS.md`](RESULTS.md), but the benchmark sweep is one row long, the container has
never been run, and nothing here has been repeated on a second day. This entry records
what exists on the day the engine first served and the container/documentation
scaffolding was added, so that the first real release has something to be a delta from.

Box: one DGX Spark class machine — NVIDIA GB10, `sm_121a`, 128 GB unified memory
(~121 GiB visible), 20 cores, one local NVMe — Ubuntu 24.04 / DGX OS, CUDA 13.

### What works

- **Phase 0 is complete.** Checkpoint layout, architecture notes and the public landscape
  survey in [`NOTES.md`](NOTES.md); the expert-routing tracer (`tools/expert_trace.py`,
  layer-streaming and resumable), the Engram row fetcher (`tools/engram_rows.py`,
  multipart HTTP ranges against the two 101 GB shards), the corpus builder and the
  coverage/LRU analysis (`tools/expert_stats.py`).
- **The full 40-layer routing histogram** over 10,760 teacher-forced tokens, in
  `results/trace-full-20260910/`. At ~4,500 resident experts (84.6 GB): 0.780 static
  coverage, 0.875 LRU hit per token, 0.796 per 6-token block.
- **The teacher-forced check of the pure-torch port**, all 40 layers plus the head: coding
  NLL 2.15 / top-1 63.8%, general NLL 3.41 / top-1 47.4%. A broken port would sit near 10%.
- **The engine runs.** `engine/` loads the real checkpoint and produces coherent greedy
  text: the arena + LRU + transient ring over `O_DIRECT` NVMe streaming
  (`engine/experts.py`), the chunked-prefill/decode-block model with caches
  (`engine/model.py`), Engram rows at serve time (`engine/engram.py`), and the generation
  loop with DSpark drafting and verification (`engine/v41_engine.py`).
- **The Triton FP4 grouped-MoE kernel** (`tools/fp4_moe.py`): 193–197 GB/s effective at
  decode sizes on GB10, relative error 4.4e-3 against the dequantised reference.
- **Chunk invariance.** `engine/model.py` is bit-exact under every chunking tested for
  sequences ≤ 512 tokens, including cache rollback after a 6-token speculative block.
- **The OpenAI-compatible server** (`server/app.py`, standard library only) with the
  thinking/effort mapping, `reasoning_content` streaming, DSML tool-call parsing and
  `x_engine_stats` on every response; 15 end-to-end tests against the mock engine.
- **The launcher and the harness**: `start.sh` / `stop.sh` with port and memory guards,
  `bench/bench.py` with the `x_engine_stats` medians.

### Added in this entry

- `Dockerfile` — arm64, `nvidia/cuda:13.0.2-devel-ubuntu24.04`, torch 2.13.0+cu130 from
  the PyTorch cu130 aarch64 index (which is also where the matching `triton` comes from),
  plus transformers / tokenizers / safetensors / numpy / sympy / huggingface_hub. No
  compile step: the only kernel is JIT-compiled on the box. The devel base rather than
  `-runtime` because Triton needs a `ptxas` that knows `sm_121a`, and
  `TRITON_PTXAS_PATH` points at the toolkit's.
- `compose.yaml` — loopback-only `127.0.0.1:8000`, `./models:/models` and
  `./results:/app/results`, `.env` pass-through, `ipc: host`, `memlock` unlimited, all
  GPUs, and `restart: on-failure:1` so a failing load can never loop the box.
- `run.sh` — `setup` / `serve` / `logs` / `stop` / `shell` / `bench` / `config`, reading
  the same `.env` as `start.sh`.
- `scripts/entrypoint.sh` — the container's `start.sh`: the same env knobs as
  `env.example`, the `MemAvailable` guard, and auto-discovery of the newest
  `results/trace-*/stats/coverage.json` to rank the warm start.
- `scripts/download-model.sh` — resumable `snapshot_download` of
  `deepseek-ai/DeepSeek-V4.1-Flash`, with the 510 GB warning and a free-space check.
- `.github/workflows/image.yml` — build and push to GHCR on `v*` tags and on demand,
  `ubuntu-24.04-arm`, `docker/build-push-action`, GHA cache.
- `.dockerignore`, `VERSION`, `docs/` (install, architecture, openai-api, benchmarking,
  gotchas), this file and `CREDITS.md`.

### Known not to work

Everything in [`LIMITATIONS.md`](LIMITATIONS.md). The short version: the container image
has never been built or run; decode is NVMe-bound at 2.6–2.7 tok/s and nothing overlaps the
expert reads with compute; bit-exactness stops at 512 tokens; long context, concurrency and
model quality beyond teacher forcing are all unmeasured; and one benchmark row is not a
benchmark.

### Measured on it

Everything in [`RESULTS.md`](RESULTS.md), all of it on 2026-09-10 on one GB10 box with the
pool to itself. The headline row, at a 73.8 GB arena (3,926 slots, 25.6 % of the routed
experts) on the `code` workload with DSpark on and thinking off:

| load to `/health` | TTFT (62-token prompt) | decode | acceptance | expert hit rate | NVMe per token |
|---|---|---|---|---|---|
| ~90 s | 11.05 s | 2.68 tok/s | 3.03 | 0.830 | 0.92 GB |

Plus the load breakdown (§1), the teacher-forced agreement with the pure-torch port (§2,
within 0.03 nats), the DSpark spec-on/spec-off A/B (§3, greedy output token-for-token
identical), and the two performance bugs the A/Bs caught (§5).

**Benchmarks are work in progress.** One workload row (`code`) exists. `prose`, both
one-shot generations and every thinking-on run were stopped before they produced a number,
so there is no measured long generation and no thinking-mode figure in this repo at all.
The earlier bring-up figures in `NOTES.md` taken on a 20 GB debug arena (6.9 % of the
routed experts) are a measurement of that arena, not of the recipe — do not quote them.

## 0.6.0-wip — 2026-09-14

**Two configuration settings stop being numbers to memorise.** The keep fraction is an answer to
the context length, and the thinking default is a property of the keep-set — so both are now
derived where they are decided, instead of being copied out of a results file by hand.

**Nothing here changes a shipped default, and nothing here has been measured on the box yet.** The
prefill chunk has been the single largest transient this engine holds since 2,048-token chunks
landed in 0.2.0, and the 7.2 GB the budget model reserves for it was a fit to a measurement with no
itemisation behind it. This entry writes down what is in it and adds an opt-in way to make it
smaller.

**The drafter becomes something you can train.** The verify step is ~145 ms and flat across
workloads, so served tok/s is `accept_len_mean / 0.145` and nothing else — and acceptance is
workload-shaped: about 5 accepted tokens a step on markup against 2.5 on prose (RESULTS.md §4.3).
Draft trees, an EAGLE-style head and adaptive draft lengths were measured and did not pay. What is
left is the shipped MTP head itself, and this tag adds the whole path to fine-tuning it on the
target's own outputs. **Nothing measured yet**: the default is the shipped head, unchanged to the
bit, and no acceptance number in this repository moves until a fine-tuned head passes
`tools/verify_mtp.sh`.

**And prefill turns out to be compute-bound, which the page predicting otherwise now says out
loud.** A page of careful reasoning said a prefill chunk's MoE time would not move with the routed
k, because the per-expert unpack is paid once per chunk whatever k is. It moves: 54.1 ms at k=6,
38.3 at k=4, 30.9 at k=3. Fewer routed experts per prefill token is therefore worth having, and it
lands here as folding rather than dropping.

### Added
- **`DSV41_ESCAPE_K` / `DSV41_ESCAPE_MARGIN` — the escape hatch.** The keep-set is a hard mask, and
  a pick it takes away stays taken away for the whole request; the two ends of that trade are the
  measured ones (no mask at all passes the prompts every keep-set fails, at 1,008–1,549 s each —
  `results/keepsets/null-unmasked/GATE.md`; the mask is 15–25 tok/s and fails them) and nothing in
  between had been tried. With `DSV41_ESCAPE_K=1` a layer-step may stream **one** non-resident
  expert into the transient ring when the router's raw, pre-mask scores say the mask displaced a
  pick worth at least `DSV41_ESCAPE_MARGIN` of the token's routed weight more than the resident
  expert standing in for it. The expert's bit is set in the router's mask in place — the captured
  decode graphs pick it up without re-capture — and withdrawn when the ring recycles its slot or the
  request ends. Bounded at 8 × K fetches per decode step, decode only, `substitute` only, and it
  requires the all-resident device slot LUT. **Ungated**: no prompt has been generated with it armed;
  `tools/verify_escape.sh` is the run that would say. Costs, all on the critical path: an 18.80 MB
  `O_DIRECT` read, the CB3 repack of three matrices, and the merged graph segments (41 graph replays
  a step instead of 3, measured at +0.6 ms of a 146.6 ms step). Off by default, and off is
  byte-for-byte the engine that shipped — `tools/test_escape_rule.py` and `tools/test_route_modes.py`
  hold both mask lines and every escape site to that. See
  [`docs/keep-sets.md`](docs/keep-sets.md#escaping-the-mask).
- **Reasoning-span decode controls**, both **off by default**, so no shipped number moves.
  `DSV41_THINK_BUDGET` / `reasoning_budget` caps the think block: once the completion has
  produced that many tokens without closing it, the decode loop forces `</think>` on the next
  step — on the speculative path too, by masking row 0 so the drafts are rejected and the
  block's bonus token is the close — and generation continues as the answer.
  `DSV41_THINK_REPEAT_BREAK` / `think_repeat_break` is a loop breaker scoped to the same span:
  when the last n tokens repeat a window already seen twice in this reasoning span, the tokens
  that continued the earlier copies are masked, so a third copy cannot be extended. The answer
  is never touched by either — an n-gram ban over an answer was refuted here, because CSS
  repeats `px` and `0` legitimately.
  The motivation is measured (`RESULTS.md` §5.4 and the 2026-09-14 addenda): with thinking on,
  a gate run that passes deliberates for about 4,000 reasoning tokens and one that fails for
  about 19,000; the Backend keep-set at 0.36 answered all ten of its prompts and restated
  itself 3–8× on seven of them, at 18–38k reasoning tokens.
  `usage.completion_tokens_details.reasoning_budget_hit` says when the budget fired and
  `x_engine_stats.think_controls` says what both controls did.
  New `server/think_controls.py` (standard library only), `server/test_think_controls.py` and
  `engine/test_think_budget.py` run without torch, a GPU or the checkpoint;
  `tools/verify_think_controls.sh` takes the gate runs on the box.
  [`docs/openai-api.md`](docs/openai-api.md#reasoning-span-controls).
- **CI runs the torch-free tests.** `.github/workflows/tests.yml` runs the sixteen `tools/test_*.py` and `server/test_*.py` scripts that need only a CPU on every push and pull request (Python 3.12, numpy); the torch/CUDA tests and `server/test_server.py`, which needs the checkpoint tokenizer, are skipped by name and reported as such.
- **`PRUNE_KEEP=auto`**, and it is the recommendation `env.example` ships. The KV and indexer
  caches are allocated for `MAX_SEQ` up front and one prefill chunk costs more behind a longer
  context, so a keep fraction that serves 32k is over budget at 256k — and it goes over by being
  killed by the memory watchdog on the first long request, not by refusing to start. `./start.sh`
  now resolves `auto` before it launches anything and prints the one line it decided on:

  ```
  PRUNE_KEEP=auto -> 0.36 for MAX_SEQ=262144 (fits with 3.3 GB spare)
  ```

  On a 121 GiB box that is **0.38 up to 128k and 0.36 at 256k** — 0.36 being the fraction the
  filled-256k run was measured at, and 0.40 the one that was killed 582 s into a 195k-token
  prefill ([`RESULTS.md`](RESULTS.md), 2026-09-13 22:50 and the 2026-09-14 addenda). A number
  still works and is passed through untouched.
- **`tools/keep_for_context.py`** — the resolver, and the same question anybody can ask by hand.
  The largest *step* of the keep ladder whose plan clears both gates (the launcher's own
  pre-flight, and a prefill chunk plus the watchdog floor once it is up), computed from
  `tools/budget.py`'s model — no torch, no CUDA, no model load, milliseconds. It honours a pinned
  `ARENA_GB` (the kept set has to fit inside it, less the transient ring) and says which arena did
  the capping; when the pinned arena is itself too big for the context it says that, rather than
  walking the ladder down after something a keep fraction cannot fix. Below 0.36 it answers and
  warns: no keep-set that small has ever been through a generation gate here. `--bare` prints the
  number for a shell, `--arena` the arena that keep needs.
- **`./start.sh --print-env`** — resolve everything `.env` and the environment imply, print it
  with the command that would have run, and start nothing. It skips the model, port and memory
  guards, so it answers away from the box; `start.sh` also no longer uses a bash-4 associative
  array, so it runs under the 3.2 that ships with macOS.
- **A thinking default per profile.** A profile in `tools/tune.py` may name one as a sixth field,
  and `./tune.sh --print/--write` emits it as `DEFAULT_THINKING` — the variable `./start.sh`
  already reads. Both language bundles ship `off`: with thinking off every language in them came
  out clean on their gate, and with it on French, German, Chinese and Japanese corrupted a word
  and looped (`results/keepsets/european_languages/GATE.md`,
  `results/keepsets/world_languages/GATE.md`, 2026-09-14; whole-file generation behaves the same,
  `RESULTS.md`). The profile screen says `thinking off by default` on the gate line, and a profile
  from a file may carry `"thinking": "on"|"off"` too. It is a default for requests that say
  nothing — any request can still ask for the other.
- **Top-1 and co-routing, out of the trace that was already taken.** The expert atlas had two
  empty fields — `top1_share` (which expert *led* a token, not merely which six it took) and
  `coroute` (which experts a token takes together) — because the reduction never wrote them, not
  because the trace never saw them. `tools/expert_stats.py` now folds both out of arrays
  `tools/expert_trace.py` has stored since the first run: `top1_<topic>`, a histogram of the
  expert carrying the largest **gate weight** per token, and `pairs_<topic>`, the top pairs per
  layer by count and by lift (`count / (count_a × count_b / tokens)`). The tracer is unchanged.
  First pick is the largest weight and not column 0 of `indices`: the router selects with the
  aux-loss-free bias and weights without it, so on the reference trace only 44 % of tokens have
  their six weights in descending order and the two definitions disagree on one token in twenty.
  `top1_<topic>` goes in `coverage.json` (+1.3 MB on 13 MB at 39 topics); the pair tables go in a
  sibling `pairs.json` (~2 MB) because the engine parses `coverage.json` on every start and reads
  neither. `tools/atlas_export.py` maps them to `top1_share`, `dynamics.all[].top1_gini` and
  `coroute`, and omits all three on a stats file that predates them, as it did before.
  `tools/expert_stats.py --merge SHIPPED --runs DIR ...` rebuilds a shipped stats file out of
  fresh reductions, and `tools/trace_extras.sh` drives it over every per-layer trace on the box
  before re-taking the export. Two things that file makes easy to get wrong are handled where
  they belong. A shipped topic can be the **union of several traces** — `reasoning_code`'s corpus
  was collected in two passes, and its shipped histogram is 118,272 routed slots a layer against
  21,258 for the smaller pass alone — so the trace set per topic is *derived*, by subset-summing
  the candidate reductions until one reproduces the shipped histogram, never guessed from a
  directory name. And the rebuilt file replaces nothing unless every `counts_<topic>` and
  `saliency_<topic>` comes back **element for element identical** to the shipped one; on any
  mismatch the shipped file is left byte-for-byte alone and the rebuild goes beside it as
  `coverage.new.json` (exit 5). Key presence was not enough: a merge that dropped a trace carries
  every key and the wrong numbers in one of them.
  Checked by `tools/test_trace_extras.py` (brute-forced pair counts and lift, both `--pairs`
  layouts, the exporter with and without the two fields, and a two-trace union rebuilt from both
  shards and refused when given one) — see [`docs/keep-sets.md`](docs/keep-sets.md)
  "Top-1 and co-routing".
- **`DSV41_PREFILL_KV_FP8=1`** (default `0`) — the prefill path's gathered window and compressed KV
  as one fp8 e4m3 buffer instead of two bf16 gathers and a concatenation of both. At a 2,048-token
  chunk that is 1,280 KB a token of live bytes down to 320 KB, and the attention gather — 3.01 GB
  at 32k context, four times the next largest moment — stops being the peak of a chunk at all. The
  caches are untouched: the window ring and the compressed/index caches stay bf16, so everything
  decode reads is in the format it is in today, and the decode path never takes this branch. The
  bounded replay does not either — 128 queries have nothing to save and produce the prompt's final
  logits. e4m3 rather than e5m2 for the mantissa bit (2^-4 against 2^-3 worst-case relative error)
  and because range is not what is scarce in an rmsnorm output; it is also the format the reference
  implementation uses for these very tensors. With the flag off, not one byte of the path changes.
- **`DSV41_PREFILL_CHUNK=4096`** now works. It did not before: `DSV41_RING` was pinned at 4,096
  slots and the window gather runs *after* the whole chunk is written into the ring, so a
  4,096-token chunk wrapped the ring inside itself and the first queries of a chunk silently read
  what the last ones wrote — the same failure class as the 64-slot ring in `docs/architecture.md`,
  with the same absence of an exception. The ring default follows the chunk now
  (`max(4096, chunk + 512)`, 22.5 MB more at 4,096), `Caches` refuses a ring that is too short by
  name, and `start.sh` passes `DSV41_PREFILL_CHUNK`, `DSV41_PREFILL_KV_FP8` and `DSV41_RING` from
  the environment the way it passes every other engine variable.
- **`tools/verify_prefill_fp8.sh`** — the run that turns all of this into numbers: baseline, fp8,
  and fp8 at a 4,096-token chunk, one identical ~7,000-token prompt each, with prefill tok/s, TTFT
  measured at the client, and the low-water mark of `MemAvailable` sampled five times a second
  throughout; then `tools/gate_profile.py --profile Frontend --thinking on` on the last of them,
  because the rounding reaches an attention score and a memory number cannot see that. Results
  under `results/prefill/`.
- **`engine/test_prefill_kv_fp8.py`** — the e4m3 round trip against its 2^-4 bound, bit-identity
  between the tiled fp8 gather and the bf16 gather it replaces at every chunk length and tile size,
  the effect on an attention output, and that an out-of-range value saturates rather than becoming
  a NaN that takes a whole query's softmax with it. Self-skips without CUDA.
- **`DSV41_PREFILL_UNPACK_CACHE_GB`** — keep unpacked FP4 experts across the chunks of one prompt.
  With `EXPERT_FORMAT=cb3` a prefill-sized MoE call cannot read a 3-bit expert: it unpacks what it
  needs back into packed FP4 (bit-exact), runs the FP4 kernel, and throws the unpack away — so
  chunk 1 of a prompt unpacks exactly the experts chunk 0 already unpacked, and at keep 0.36 a
  four-chunk prompt pays 4 × 21 × 139 unpacks where 21 × 139 would do. This buys a region of the
  same FP4 scratch arena that survives between chunks. **The policy is a fixed layer window, not an
  LRU**: prefill is a sequential scan over layers 0..20, so an LRU smaller than the 54.9 GB working
  set evicts layer 0 exactly before chunk 1 asks for it again and scores zero hits at every budget
  (`tools/test_unpack_cache.py` checks that against a real LRU). Admitting on first touch and never
  evicting inside a request gives a hit rate of `cached layers / layers per chunk` instead. Cleared
  at the end of every request and whenever an arena slot is rewritten, so it can never serve a stale
  expert; with the cache off the prefill path is the previous code, launch for launch.
  `unpack_hits` / `unpack_misses` / `unpack_bytes_saved` / `unpack_ms_saved` appear in
  `x_engine_stats`. **Default 0 = off**: at a 6 GB budget the arithmetic predicts ~8 % of a
  four-chunk prompt's expert lookups, which is under a percent of prefill wall time, and whether
  that is worth a step of the keep slider is a measurement nobody has taken on a box yet.
  [`docs/memory-budget.md`](docs/memory-budget.md#the-prefill-unpack-cache).
- **`tools/verify_prefill_cache.sh`** — the A/B that settles it: one engine load with the cache off
  and one with it on, the same prompt both times, then the generation gate against the cache-on load
  (a cache that changes what the model writes is a bug, not a speedup). Records go under
  `results/prefill/`, never into a profile's `GATE.md`.
- **`tools/longprefill.py`** — one long, deterministic prompt built from the checkout's own corpus
  in a fixed file order, sent streaming, reporting client time-to-first-token beside the server's
  `prefill_s` / `prefill_tok_s` and the unpack counters. Stdlib only.
- **`DSV41_RECORD_DRAFT_DATA=<dir>`** — record what a fine-tune needs while the server decodes
  normally. Per settled position: the target's last hidden state (`main_hidden`, bf16 [15360] — the
  drafter's entire view of the target, kept *before* `main_proj` so that projection stays
  trainable), the token the target settled on, and its top-32 next-token logits. 30,988 bytes a
  position, so 300,000 positions is 9.3 GB; the last 128 prompt positions of every request are
  recorded too, because the drafter's window is 128 and without them only the tail of a 175-token
  generation would be usable. One shard per request, with a manifest, in `engine/draft_record.py`.
  Unset — the default — and the module is never imported and the decode loop runs one `is not None`
  test per verified block; the arithmetic of a step is byte-identical either way.
- **`tools/draft_data_gen.py`** — the data run. Streams passages out of `corpus/` against the
  running server and asks for 150–200 token continuations with thinking off until `--tokens`
  (300,000) have been settled. 75 % prose and reasoning — the registers where the drafter is weak —
  and 25 % code and markup so the head does not forget the register where it already accepts five
  tokens a step. `corpus/heldout_sources/` is never sampled: it is what every teacher-forced number
  in RESULTS.md is measured on. Prints the wall clock before the first request (~5 h for 300k
  tokens at the served 17 tok/s), logs every request, and resumes.
- **`tools/train_mtp.py`** — FastMTP (Red Hat / vLLM, 2026-09) on one GPU with the target never
  loaded. Starts from the shipped `mtp.*`, shares the full unreduced LM head, and trains the loop
  serving actually runs — the block forward over a window of recorded target hidden states, with
  the rank-256 Markov chain — under an exponentially decayed per-step loss (`beta = 0.6`) of
  forward KL against the target's top-32 plus cross-entropy on the token it settled on. The decay
  is not a taste: acceptance is a leading-prefix quantity. The drafter's own 13.6 G routed-expert
  parameters are frozen (27.2 GB resident as bf16; their gradient and AdamW state would be another
  190 GB) and the 636 M dense ones are trained. `--plan` prints the whole ~43 GB budget from the
  checkpoint's headers before anything is allocated, and the run refuses to start if the box has
  less free — this and the engine cannot share the box. Reports the acceptance proxy (top-1
  agreement per draft step, held out by shard) before, after, and after the fp8 round trip the
  engine does on load.
- **`DSV41_MTP_WEIGHTS=<path>`** — serve a fine-tuned head instead of the checkpoint's. Per tensor:
  what the file does not name comes from the checkpoint. Each tensor is re-quantized on load into
  the format the shipped path serves in (fp8, then FP4 for the `DSV41_DENSE_FP4` groups), so a
  fine-tuned head reads exactly as many bytes per draft as the shipped one and the speed of the
  draft graph does not move. Unset = the shipped head.
- **`tools/verify_mtp.sh`** — the number that decides anything. Serves the head on the Frontend
  keep-set at `PRUNE_KEEP=0.36`, reports `accept_len_mean` and tok/s for one prose and one markup
  prompt (the second is the one that must not regress), then runs the Frontend and Writing
  generation gates with thinking on. Records everything under `results/mtp/<stamp>/`,
  log in `/tmp/mtp.log`.
- **Three more checks.** `tools/test_draft_data_gen.py` needs no torch: the corpus mix and its
  determinism, the held-out exclusion, the 30,988-byte record layout and the manifest, and which
  recorded positions can be trained on at all. `engine/test_draft_record.py` and
  `tools/test_train_mtp.py` need torch and skip themselves without it: the shard's bfloat16 round
  trip and the rows a verified block may record, and the FastMTP loop on a 32-dimensional toy head
  — every trainable tensor gets a gradient, twenty steps overfit one batch, and the acceptance
  proxy counts a leading prefix rather than a per-step mean.
- **`DSV41_PREFILL_FP8_DEQUANT` — the fp8 weights stop being dequantised into fp32 and back.**
  Above decode-sized M, `v41_ref.dense` hands an `FP8Weight` to cuBLAS as a bf16 dequant, and
  `FP8Weight.dequant()` builds a full `[N, K]` fp32 scale table with two `repeat_interleave`s plus
  a full fp32 weight to get there — **ten transient bytes per weight, per weight, per layer, per
  chunk**. Under the shipped `DSV41_DENSE_FP4=attn,wo_a` that is the shared experts on all 21
  encoder layers, the indexer `wq_b` on four of them and the engram `wkv` on two: 1.079 G weights a
  chunk, at the 0.98 ns per weight this shape was measured at, or **~1.06 s of a ~5.55 s
  2,048-token chunk**. `fused` does it in one new Triton kernel
  (`tools/fp8_linear.dequant_fused`) — fp8 codes in, bf16 out, two transient bytes per weight, one
  launch instead of six — and **bit-identically**: e4m3 → bf16 loses nothing, the block scale is a
  power of two, and the closing rounding is the same one torch does. `cached` adds memoisation for
  the rest of a request's prefill, which buys the residual ~16 ms a chunk from the second chunk on
  and costs 2.16 GB while the encoder pass runs and 3.70 GB once the replay and the DSpark seed
  have run; `tools/budget.py` reads the variable and charges that against the prefill reserve, so
  `PRUNE_KEEP=auto` answers for it. `scaled_mm` is refused with its reason: this checkpoint's
  scales are one UE8M0 value per **32x32** block and `torch._scaled_mm` takes no such granularity,
  so reaching it would mean re-quantising the weight. **Off by default, and off calls
  `FP8Weight.dequant()` — which is untouched — so it is the old code and not a re-derivation of
  it.** Ungated: no prompt has been generated with either mode armed;
  `tools/verify_prefill_fixes.sh` is the run that would say. See
  [`docs/gemm-dispatch.md`](docs/gemm-dispatch.md).
- **`DSV41_PREFILL_FUSED_SINKHORN` — prefill picks up the Sinkhorn kernel decode already had.**
  `v41_ref.hc_mixes` runs the 20-iteration Hyper-Connection Sinkhorn through `tiled_rows` in
  16-row tiles, and the torch port is ~130 tiny kernels per call on a `[m, 4, 4]` tensor: at a
  2,048-token chunk that is 128 tiles × ~130 kernels × two calls per layer × 21 layers =
  **698,880 launches a chunk**, each on a 16×4×4 tensor. `engine/hc_sinkhorn.py` has replaced all
  of that with one launch, one program per token row, since the fast decode path existed —
  `engine/fastdecode.py` imports it, `engine/model.py` never did. With the flag on the count is
  **42**. No new kernel and no fork of the old one: the grid is `(max(n, 1),)` with a tail guard
  and there was no decode-shaped limit to extend. One program per row is row-count- *and*
  row-offset-invariant by construction, so it replaces the chunk-invariance the tiling was there to
  manufacture rather than skipping it. **Prefill-only** — a fused Sinkhorn in the un-graphed decode
  forward would move the tokens the DSpark verify step accepts and nothing would crash — and off by
  default. Ungated, like the above.

- **`DSV41_PREFILL_ATTN_GEMM=fp32|tf32|bf16` — the math type of prefill's two attention GEMMs**,
  and the finding that made it worth having. The first `tools/audit_gemm_dispatch.py --phase
  prefill` run on the box put the largest single kernel cost of a 2,048-token chunk not in the MoE
  but in `cutlass_80_simt_sgemm_128x32_8x5_tn_align1` (559.5 ms, 672 calls) and its `_nn_` variant
  (295.7 ms, 672 calls) — **855 ms, more than both FP4 MoE kernels together** (`_moe_up` 483 +
  `_moe_down` 410). The call counts name the matmuls with no ambiguity: they are the score product
  and the PV product of `engine/model.py::Model._softmax_attn`, which runs in `ATTN_TILE = 64`
  query tiles, so 2,048 / 64 = 32 tiles × 1 product × 21 encoder layers = 672 of each, 1,344
  together, and nothing else in the prefill path has that count (the `MM_TILE = 16` sites issue 128
  per layer per projection). `simt_sgemm` says fp32 through the FFMA pipeline: those GEMMs never
  touch a tensor core. `tf32` arms `torch.backends.cuda.matmul.allow_tf32` and
  `float32_matmul_precision` around the tile loop and restores both in a `finally` — inputs rounded
  to an 11-bit significand, accumulation still fp32, ~1e-3 on the product; `bf16` skips the fp32
  widening altogether — 8-bit significand, ~4e-3, and 84 MB less transient per tile (the
  `[64, 640, 512]` fp32 copy of the gathered KV, ~56 GB written per chunk, which `tf32` keeps).
  Prefill only in both cases: `mode_for(prefill)` returns fp32 for every other call, so the captured
  decode graphs, the DSpark drafter and `engine/fastdecode.py` keep strict fp32 whatever the
  environment says. **`fp32` is the default and is byte-for-byte the engine that shipped** — the new
  `engine/prefill_attn_gemm.py` writes nothing and imports no torch on that path, and
  `tools/test_prefill_attn_gemm.py` (torch-free) holds the default, the alias, the prefill guard,
  the scoped restore and every call site to that mechanically;
  `engine/test_attn_gemm_modes.py` compares all three modes on the real shapes against those error
  bounds from both sides and self-skips where there is no GPU. `tools/budget.py::prefill_bytes` is
  unaffected — `tf32` allocates what the engine always did and `bf16` allocates less.
  **Ungated**: no prompt has been generated with either non-default mode armed, and a speed number
  cannot accept one. This engine's chunk-invariance argument is built on 1e-7 GEMM jitter — the
  comment directly above these two einsums says rounding the probabilities to bf16 is "a cliff that
  turns 1e-7 fp32 GEMM jitter into 1e-4 output jitter, which is enough to flip a borderline router
  top-k" — so what breaks is a different expert deep in a long prompt, which only generation shows.
  `tools/verify_prefill_hc.sh` is the run that would say: one identical ~7,000-token prompt in each
  of the three modes (prefill tok/s and TTFT from the server), then the kernel audit in `fp32` and
  `tf32` with the server stopped, then `tools/gate_profile.py --profile Frontend --thinking on` on
  the recommended mode; record under `results/prefill/`, transcript in `/tmp/prefill_hc.log`.
  `DSV41_PREFILL_HC_GEMM` is accepted as an alias — the name the investigation started under,
  before the call counts said attention rather than the Hyper-Connection residual. The audit is
  appended to [`docs/gemm-dispatch.md`](docs/gemm-dispatch.md), including why `align1` is **not** a
  layout bug to fix (both operands are already contiguous and 16-byte aligned) and why the other
  fp32 GEMM, the HC mix projection, needs `tools/fp32_skinny.py` rather than a math type.

- **`DSV41_PREFILL_TOPK` / `DSV41_PREFILL_FOLD` — fewer routed experts in prefill, folded onto the
  ones that remain.** `tools/prefill_bound_test.py` settled the question
  [gemm-dispatch](docs/gemm-dispatch.md) was written to ask: one 2,048-token chunk through one MoE
  layer costs 54.1 ms of `moe_fn` at k=6, 38.3 ms at k=4 and 30.9 ms at k=3 — 0.57x where the pairs
  are 0.50x, so the FP4 grouped GEMMs dominate and the unpack is a floor rather than the bill. The
  prediction on that page said 1.0x and is left standing next to the measurement that refuted it.
  The reduction is implemented as **ExFold** (arXiv 2608.24938), whose Table 6 is on this model
  family — DeepSeek-V4-Flash, 256 routed experts, Top-6 routing: at three routed experts per prefill
  token, direct Top-3 keeps 97.05 % of the Top-6 baseline average and folding keeps 98.42 %. Per
  token the router still picks its six; `K'` are kept by weight **times** a calibrated output-norm
  estimate (expert norms span >3x within a layer, so the largest weight is not the largest
  contribution), and each excluded expert is folded onto the retained expert with the smallest
  calibrated reconstruction loss by adding `w_s · c_s · S[s,t]` to its weight — one scalar per
  ordered pair per layer, plus the confidence term the paper adds specifically for this
  architecture, whose remainder falls back pro rata. Nothing is renormalised: the scalar is the
  magnitude correction. Decode is untouched (`engine/fastdecode.py` does not import any of it),
  no graph changes, and prefill has no captured graphs to invalidate. **Expect ~7 % at k=4 and
  ~12 % at k=3, not the paper's 1.32x** — the FP4 MoE kernels are ~893 ms of a ~3.7 s chunk here,
  and their H800 figures come from a box with a different balance and from a serving curve with
  queueing in it. Off by default; off loads no table and takes no branch.
- **`tools/exfold_prepare.py`** — the one-time calibration, with the server stopped. Observes ~64
  tokens in each of 64 sequences of this repository's own trace corpus (unsupervised, and unlike
  the paper's released matrix it sees no benchmark input at all), recovers each routed expert's
  output through the engine's own MoE kernel rather than a second reference implementation, and
  fits the scalars and losses from one 6×6 Gram matrix per token per layer. Writes ~47 MB of
  tables next to the checkpoint. It refuses to run with a top-k override already in force, and the
  engine refuses to start with `DSV41_PREFILL_FOLD=exfold` and no tables rather than quietly
  serving the drop arm.
- **`tools/verify_prefill_topk.sh`** — the run that would gate it, and has not been taken: one
  identical ~7,000-token prompt at k=6, k=3 and k=4 with prefill tok/s and TTFT read back from the
  server (including the engine's own report of what `k` it actually used, so a variable that never
  arrived cannot be reported as "this changes nothing"), then
  `tools/gate_profile.py --profile Frontend --thinking on` at k=4. The gate is the point: prefill's
  approximation does not stay in prefill, because the window KV the decoder reads is what the
  prompt wrote, and an NLL number cannot see the failure mode that produces.

### Changed

- **Two prefill defaults move (2026-09-15).** `DSV41_PREFILL_ATTN_GEMM` defaults to `tf32`
  (+5 % prefill; Frontend gates 4 · 9 · 0 and, with the fused dequant, 8 · 10 · 0) and
  `DSV41_PREFILL_FP8_DEQUANT` to `fused` (bit-identical to the old dequant on the device,
  `tools/test_fp8_dequant.py`, +3.3 %). Decode stays fp32 by a named constant, under test.
  `bf16` is +21 % prefill (459 against 380 tok/s warm on a 6,678-token prompt) and passed two
  full gates, and it was the default for four hours of 2026-09-15 until the drafter verify
  measured prose decode acceptance after a bf16 prefill at 1.99 a step against 2.69 (10.4
  against 14.3 tok/s) on the same prompt and head, markup unchanged. A sampled A/B (fifteen runs a side, three prompts) then put bf16 3 % below tf32 in draft
  acceptance on prose and markup alike, which with thinking on costs more decode seconds than
  the prefill gives back. It is opt-in. RESULTS.md 2026-09-15 09:30, 12:50, 17:20 and 18:25 have
  the cards, the rows and the A/B.
- `env.example` ships `PRUNE_KEEP=auto` with `ARENA_GB` **empty**. When the keep fraction is
  resolved and no arena is pinned, `./start.sh` sizes the arena for the kept set: the engine's own
  automatic sizing takes 82 % of what is free, about 89.6 GB on this box, which is more arena than
  a filled context can afford. Pin both together to reproduce a measurement exactly.
- `./tune.sh --keep auto` follows the context slider, writes `auto` back as `auto` with the value
  it resolves to today on a comment line, and leaves `ARENA_GB` empty for the launcher to size —
  so a `.env` written at 32k is still the right configuration at 256k.
- `DEFAULT_THINKING` joins the keys `./tune.sh` manages in `.env` (eleven now). It is written only
  when a profile named one or the environment already carried one, never invented.
- `tools/budget.py` charges the unpack cache to `prefill_bytes`, so it lands in both memory gates
  and in `./tune.sh`'s verdict; `engine/v41_engine.py` adds it to its own pre-flight and **clamps**
  it to what is left rather than refusing to start. On a 118.6 GB box a 6 GB cache still fits keep
  0.36 and does not fit keep 0.39 — the trade is explicit rather than discovered by the watchdog.

### Fixed

- `tools/gate_profile.py` no longer reports `a===b` as a corrupted run: three-character operators
  welded between operands are JavaScript, and the check skips them (test added).

- `./start.sh` no longer exits silently, before printing anything, on a checkout with no
  `results/trace-*/stats/coverage.json`: the fallback lookup ends in a failed test, and under
  `set -e` an assignment took that exit status with it.
- **The generation gate's length columns are characters, and now say so.**
  `tools/gate_profile.py` recorded `len(reasoning_content)` / `len(content)` under the headings
  `reason` / `answer`, and those figures were quoted as token counts in `RESULTS.md`, `README.md`,
  `env.example`, `docs/openai-api.md` and `server/think_controls.py` — a token is about four
  characters in this register (2,001 forced reasoning tokens = 7,953 characters), so a reasoning
  budget chosen from them was about 4× too large to fire and the reasoning-span controls were read
  as broken when they were not. Rows now carry the server's own counts
  (`usage.completion_tokens_details`): stdout columns `reas tok` / `ans tok`, with a trailing `c`
  on any cell that could only be given in characters, GATE.md columns
  `reasoning chars | reasoning tokens | answer chars | answer tokens`, and
  `[reasoning budget hit at N reasoning tokens]` in the `why` of any row whose reasoning span the
  server ended. `tools/language_gap.py` reads both table shapes and no longer calls the characters
  it reads tokens. The corrected statements are listed in `RESULTS.md` (correction of
  2026-09-14 21:10); no measurement changed, only its unit.
- **The budget model priced a longer chunk wrong in one of its two terms.** The context term
  (15.1 KB per token of context) comes from the indexer's score tiles, which are shaped
  `[chunk × compressed positions]` — so it scales with the chunk as much as the chunk term does. It
  was modelled as a constant, which at a 4,096-token chunk understated the reserve by 0.5 GB at 32k
  and 2.0 GB at 128k. Corrected in `tools/budget.py` and `engine/v41_engine.py` together, and
  `tools/test_budget.py` now fails if the two drift on the chunk factor, the fp8 saving, or either
  rate.
- `tools/budget.py` reads `DSV41_PREFILL_CHUNK`, `DSV41_PREFILL_KV_FP8` and `DSV41_RING` from the
  environment, the way it already read `DSV41_DENSE_FP4` and `DSV41_HEAD_FMT`. A panel pricing a
  2,048-token chunk on a box whose `.env` says 4,096 understates the reserve by 7 GB and its verdict
  cannot be trusted.
- `/health` reports `prefill_kv_fp8` and `ring` beside `prefill_chunk`, so a run can be told what
  the engine actually read rather than what it was meant to be given.

### Documentation

- [`docs/tune.md`](docs/tune.md) — "The keep fraction can choose itself", and why thinking is part
  of a gate result rather than a preference.
- [`docs/memory-budget.md`](docs/memory-budget.md) — "Asking the ladder instead of reading it",
  with what each context length can afford on this box.
- [`docs/tune-reference.md`](docs/tune-reference.md) — `--keep auto`, `--thinking`, the
  `thinking` field in a profiles file, and the new gate-line element.
- [`docs/memory-budget.md`](docs/memory-budget.md) gains **Where a prefill chunk's memory goes**:
  every per-chunk allocation with its dtype and size at chunk 2,048, split into the terms that scale
  with the chunk and the terms that scale with chunk × context, and the four moments that compete
  for the peak. The headline of it is that the line items add up to **1.47 MB a token, not the ~5 MB
  a token the pre-flight quotes** — the rest is the caching allocator's high-water mark, not a
  tensor, and the instrument for that is `memory_allocated()` against `memory_reserved()`, which
  nobody has run.
- [`docs/architecture.md`](docs/architecture.md) gains "Fine-tuning the drafter": the data format
  and its size, the training loop as implemented, the memory budget, and what a fine-tune can cost
  — markup drift, the bf16→fp8 round trip, data shaped by the drafter that produced it, and a
  greedy proxy for a sampled path.
- [`docs/gemm-dispatch.md`](docs/gemm-dispatch.md) — "What changed after the audit", with the byte
  and launch arithmetic behind both switches, why `fused` is the recommendation over `cached`, and
  why `scaled_mm` is a refusal rather than a task.
- [`docs/memory-budget.md`](docs/memory-budget.md) — gate 2 gains the one setting that adds to the
  prefill reserve, with what `cached` holds after each phase of a prompt.

### Checks

- `tools/test_keep_for_context.py` — the resolution at every context length against a fake host,
  monotonicity in the context, `ARENA_GB` capping, the gate-floor warning, numeric passthrough,
  the CLI's exit codes, and `./start.sh --print-env` resolving it end to end.
- `tools/test_tune_profiles.py` and `tools/test_tune_draw.py` cover the thinking field, what
  `--print` writes for it, and the gate line at every width.
- `tools/test_prefill_fp8.py` and `tools/test_prefill_sinkhorn.py` — torch-free: both parsers, the
  refusal of `scaled_mm` and the reason it carries, the 2.16 / 3.70 GB of `cached` derived from the
  checkpoint's shapes, the 698,880 → 42 launch arithmetic derived from the op count, and regex pins
  holding both switches off by default, prefill-only, and `FP8Weight.dequant()` untouched.
- `tools/test_fp8_dequant.py` — the fused dequant against `FP8Weight.dequant()` compared as raw
  bits, on the CPU for the torch fallback and on the device for the Triton kernel, plus the cache's
  identity and lifetime. `tools/test_hc_sinkhorn_prefill.py` — the fused Sinkhorn against the tiled
  path at 1 to 2,048 rows, that the disagreement does not grow with the row count, and that a slice
  of a call is bit-identical to those rows of the whole call. Both self-skip without torch or CUDA,
  so they sit in the ordinary sweep.
- `tools/verify_prefill_fixes.sh` — the run on the box: four runs on the shipped `.env` over one
  identical ~7,000-token prompt (both off, each alone, both), TTFT and `prefill_tok_s` per run,
  `tools/audit_gemm_dispatch.py`'s launch count and GPU-busy fraction on the first and the last,
  and one Frontend generation gate on the last. Records under `results/prefill/`.

- `tools/test_exfold.py` — the routing arithmetic on a toy, torch-free: `k' >= k` is identity, the
  `w · h` ranking against `w` alone, the minimum-loss target chosen over the next-ranked route, the
  confidence split between fold and fallback, an unobserved pair folding nothing, a table with no
  observed pair at all degrading to a pro-rata reweighting rather than to a drop, a scalar above 1
  deliberately raising the routed mass and a negative one subtracting rather than being clamped.
  Where torch is importable it runs `engine/exfold.py` over the same toy and holds the tensor path
  to the reference elementwise; CUDA is never required.
- `tools/test_prefill_topk.py` — extended for the two new variables and the retirement of the
  measurement-only one: the fold defaulting to `exfold` whenever the top-k is reduced, an unknown
  mode refused, the table path resolution, and mechanical pins that the router still issues the
  *checkpoint's* k (so the fold knows what it excluded), that the reduction sits after the
  checkpoint's own renormalisation and before the slot lookup, and that the decode path's topk is
  untouched.

## 0.5.0 — 2026-09-14

**Choosing what the box is good at becomes a thing you can see.** About 40 % of the routed experts
fit in memory at once, and which 40 % decides both what the model is good at and whether it loads.
Until now that was two numbers in a file and a three-minute wait to find out.

### Added
- **`DSV41_PRUNE_SOURCE=saliency`** — rank a keep-set by how much each expert *contributes* instead
  of how often it is picked. Everything here has ranked experts by routing frequency; REAP (Lasby
  et al., Cerebras, ICLR 2026, [arXiv 2510.13999](https://arxiv.org/abs/2510.13999)) benchmarked
  that on Kimi-K2 — 384 routed experts, one shared, auxiliary-loss-free routing, this model's shape
  — and frequency-based pruning collapses where saliency holds: LiveCodeBench 0.434 → **0.082** at
  75 % of experts kept and **0.000** at 50 %, against 0.440 and 0.429 for saliency. Saliency is
  `gate_weight(t, e) · ‖expert_e(x_t)‖₂` over the calibration corpus, stored as a sum rather than
  REAP's mean (the rules normalise each layer's histogram before ranking it, so the two differ only
  by the count factor, and the sum is frequency × magnitude). `tools/expert_trace.py` now records
  `out_norms` beside `indices` and `weights`; `tools/expert_stats.py` writes `saliency_<topic>`
  beside `counts_<topic>`; the engine, `tools/budget.py` and `./tune.sh --source counts|saliency`
  read whichever family the variable names, with the three ranking rules unchanged. A trace taken
  before this carries no saliency histograms and the engine refuses by name rather than falling
  back. **Not yet gated on the generation harness here** — REAP's numbers are REAP's, on another
  model and another benchmark — so `counts` remains the default and nothing about a default run
  moves. `tools/test_saliency.py` checks all of it without torch, a GPU or the checkpoint.
  [`docs/keep-sets.md`](docs/keep-sets.md#frequency-is-not-contribution).
- **`./tune.sh`** — pick topics, watch what they cost against the memory the box has free right
  now, and start the server from the same screen. Coverage per topic (the fraction of its measured
  routing the budget keeps resident) with the number of tokens each was traced on; the memory
  the selection costs, line by line; the largest keep fraction this box will take.
  `--list`, `--print` and `--write` need no terminal. [`docs/tune.md`](docs/tune.md).
- **`tools/budget.py`** — the cost model behind it, with no torch dependency: slot sizes from the
  kernel's own constant, the KV cache from the checkpoint's shapes, the launch gate from
  `engine/v41_engine.py`.
- **`EXPERT_TOPICS`** composes a keep-set from named topics instead of one of three fixed profiles.
  Each topic is a per-layer expert histogram measured on a corpus of that topic alone, stored in
  `coverage.json`, so composing is arithmetic on numbers already in the checkout.
- **`corpus/fetch_topics.py`** gathers the sources a 35-topic catalogue needs — sixteen programming
  languages from a tree you name, eleven natural languages and eight domain registers from
  Wikipedia — and prints the flags `make_corpus.py` wants.
- **Profiles you write yourself.** The named bundles on the first screen are no longer only the ten
  built in: `results/keepsets/profiles.json` and
  `$XDG_CONFIG_HOME/deepseek-v41-flash-spark/profiles.json` are read as well, and one of them may
  replace a built-in profile by name. `s` on the topic screen and `--save-profile NAME` keep the
  current selection as one. A profile from a file is never gated and the screen says `untested`; a
  file that will not parse costs its own profiles and nothing else, and a topic name the keep-set
  does not carry is named rather than dropped in silence.
  [`docs/tune-reference.md`](docs/tune-reference.md#profiles-from-a-file).
- **`--brief`**, and `b` on the profile screen, write out the task of adding a topic to the
  catalogue, as Markdown, from the loaded keep-set: what it carries and how much text each topic was
  traced on, which catalogue groups are absent here, the commands with this checkout's paths, how
  many tokens a topic needs and the two correlations that say so, and what one more topic costs the
  ones already selected. Coverage can see a gap in the selection and never a gap in the catalogue,
  which is how a profile scoring 0.85 on all five of its topics still reasoned in circles.
- **Six checks that need no GPU, no checkpoint and no torch**: `tools/test_budget.py`,
  `tools/test_engine_kwargs.py`, `tools/test_tune_draw.py`, `tools/test_tune_profiles.py`,
  `tools/test_tune_brief.py` and the existing `server/test_server.py`.
- **`DSV41_PRUNE_MODE=drop`** — an experiment on what a routing pick does when its expert is not
  resident, rather than on which experts are kept. The engine has always hidden the evicted experts
  from the router, so a displaced token is computed with six experts it did not ask for at full
  renormalised weight; `drop` keeps the router's real six and weights the ones that did not survive
  with exactly 0. About 30 % of the routing mass is displaced at these keep fractions however the
  keep-set is chosen, and on the 2026-09-12 generation gate that showed up as rare tokens corrupted
  at subword boundaries — `clearTimeout` as `cleartimeout`, `OSError` as `oenerror` — which the
  model then loops trying to repair. Implemented in both the prefill and the CUDA-graph decode
  path; **not yet gated on the generation harness**, so the default is unchanged to the bit.
  `tools/test_route_modes.py` is the seventh torch-free check.
  [`docs/keep-sets.md`](docs/keep-sets.md#why-coverage-predicts-whether-long-generations-hold-together).
- **`a` in `./tune.sh` draws the keep-set** (2026-09-14). The screen budgets 15,360 experts and
  could never show one of them. **Weight Atlas by alesha-pro**
  ([github.com/alesha-pro/atlas](https://github.com/alesha-pro/atlas), MIT,
  [atlas.alesha.pro](https://atlas.alesha.pro)) draws a MoE model's expert field as one grid —
  a column per expert, a row per layer, coloured by how much of the output each expert carried,
  with any domain as a slice and any set as an outline over it — and that is exactly the shape
  of what `results/keepsets/` measures. The built site is **vendored** at commit
  `b57e75a583378fe073d106f122342718ec7f0887` under `tools/atlas/` (their `LICENSE` verbatim, the
  upstream URL, the pinned commit, the rebuild steps and the 22-replacement patch it was built
  from in [`tools/atlas/UPSTREAM.md`](tools/atlas/UPSTREAM.md)); it is somebody else's work and is
  credited as such in [`CREDITS.md`](CREDITS.md). Pressing `a` on either screen exports this
  checkout's routing trace into the three files that page reads and serves `tools/atlas/` from a
  `ThreadingHTTPServer` bound to `127.0.0.1` on a port the kernel picks — never `0.0.0.0`, and the
  socket dies with the screen — then puts the URL and an `ssh -L` line in a popup. `--atlas` does
  the same without a terminal and `--atlas-export` writes the files and serves nothing. Because
  `a` is now the atlas, **select-all on the topic screen moves to `A`**.
  [`docs/tune.md`](docs/tune.md), [`docs/tune-reference.md`](docs/tune-reference.md).
- **`tools/atlas_export.py`** — the trace in that page's schema, and nothing invented: the 40 × 384
  REAP-saliency grid, routing share and per-token contribution, each of the 39 traced topics as its
  own slice, and each of the ten shipped profiles both as a colour field and as an outline holding
  the experts `tools/budget.py` would hand the engine at the keep fraction that profile's
  generation gate was measured at. The weight inventory is architecture-derived with every measured
  statistic left at 0, so the wall above the grid renders hatched rather than claiming a scan that
  was never taken. Output is `tools/atlas/models/` (~5.6 MB, generated, gitignored) and is re-made
  whenever `coverage.json`, `gates.json` or `tools/tune.py` is newer than it.
  `tools/test_atlas_export.py` checks every outline against `TopicIndex.curves(...)` itself, and
  `tools/test_tune_atlas.py` checks the key legend, the popup at seven window sizes and a real
  fetch over a real loopback socket — two more checks that need no GPU, no checkpoint and no torch.

### Fixed
- **The gate tool's default output directory did not name the directories the records are in.**
  `tools/gate_profile.py:slug` turned `Chat and explanation` into `chat-and-explanation` where the
  record lives in `chat_and_explanation`, so the next run of that profile's gate would have started
  a second, empty directory beside a record it was meant to append to. Checked now, against the
  directories in the checkout.
- **`EXPERT_TOPICS` had never worked.** `expert_topics` was read inside `V41Engine.__init__` and
  passed by the engine's own CLI, but was never a parameter of it, so every launch through
  `start.sh` raised `TypeError` three minutes in, with the weights already on the GPU.
  `tools/test_engine_kwargs.py` now checks every launcher kwarg against the signature.
- **`VERSION` had said `0.1.0-wip` since the day it was written**, through four tags. The container
  is tagged from that file, so every image built from 0.2.0 onward carried the wrong version.

### Changed
- **`./tune.sh` shows the generation gate, and budgets a profile from it** (2026-09-14). Every one
  of the ten shipped profiles has been through `tools/gate_profile.py` since 2026-09-13, and the
  screen now reads each result back out of `results/keepsets/<record>/GATE.md` as it draws: the
  strict count at the right edge of the name row, and under the description the date of the run,
  the keep fraction it measured and — from 2026-09-14 on — how many of its prompts produced a
  correct answer despite the repeat rule. `Backend · 3 of 10 strict · 10 finished` says more than
  either number alone. A profile's record is the newest full run on exactly its topics; a filtered
  re-run and a run on a different bundle are never it. Where every profile used to read `untested`,
  only a profile from a file does now.
  Consequently a shipped profile is budgeted at **the keep fraction its gate ran at** rather than
  at the smallest one that reaches the coverage target. That target was calibrated on the `counts`
  histograms under `sum`; under the `saliency`/`maxmin` pair the box is run with, every topic in
  the shipped keep-set is above it at keep 0.12 — a third of the smallest keep fraction anything
  has ever been generated at. `--profile backend --print` now emits the configuration Backend was
  measured in — the keep fraction **and the ranking pair**, since a keep fraction reproduced
  without `DSV41_PRUNE_RANK` and `DSV41_PRUNE_SOURCE` holds a different set of experts. Applying a
  profile with a record switches the histogram family too and reloads the index, so the bars, the
  budget panel and the written `.env` all describe the keep-set that was gated; where the loaded
  keep-set has no histograms of that family, nothing is switched — that would be a configuration
  the engine refuses at load — and the mismatch is named on the gate line and on stderr. A profile
  from a file names no pair and changes neither. Coverage remains a true measurement of routing and
  is no longer read as a recommendation: where the coverage target asks for a keep fraction below
  anything gated, the screen says so.
- **The screen says which keep fraction holds a filled 256k**, on both views, because that is a
  fact about this box that nothing else on the screen implies — the KV cache is 1.0 GB at 256k and
  the prefill is what runs out.
- **`tools/gate_profile.py` records the keep-set it measured** in the card it appends —
  `PRUNE_KEEP`, `DSV41_PRUNE_RANK` and `DSV41_PRUNE_SOURCE` from the environment of the run — so a
  gate result carries its own configuration. For the runs written before that,
  `results/keepsets/gates.json` names each profile's current record and the configuration it used,
  with the `RESULTS.md` section that states it; `tools/test_tune_profiles.py` checks every entry
  against the record it points at.
- The recommended keep fraction on a 121 GiB box drops to **42 %**. One prefill chunk needs about
  10 GB on top of everything resident, and the engine's own pre-flight does not know that — it runs
  before the drafter experts, the KV cache and any prefill exist. A 98 GB arena passes it, reports
  ready, and is killed by the memory watchdog on the first request. See
  [`LIMITATIONS.md`](LIMITATIONS.md).

### Known, unfixed
- Long generations still degenerate past the gate's reach: clean at 1,200 tokens, collapsed into
  repeated corrupted CSS by 2,400, with penalties at 0. Not attributed to a layer of the stack yet.

## 0.4.1-wip — 2026-09-12

**Guard rails, after the box had to be power-cycled three times.**

### Added
- A pre-flight that refuses to start when the arena plus its warm-start scratch plus the free-memory
  floor exceeds `MemAvailable`, and a watchdog thread that kills this process rather than let the
  machine thrash when host memory runs out.
- A device-side slot table for the chunked prefill path.

### Fixed
- A published routing-timer reading was wrong and is corrected: `route_s` is nested inside `moe_s`,
  so removing it changed wall-clock time by nothing.

## 0.4.0-wip — 2026-09-12

**A configuration that writes whole files and whole stories.** v0.3.0-wip shipped a default that
scored better on teacher-forced loss and degenerated in free generation; this tag replaces the metric
that allowed it, rebuilds the expert keep-set on a corpus that contains the workloads, and fixes a
real routing bug in the graphed decode path.

### Fixed
- `engine/fastdecode.py`: the router gate runs in fp32, as `Model.moe` does. In bf16 it selected
  different experts for 11-31 % of tokens (RESULTS 4.4).
- `DSV41_FUSED_ATTN` now defaults to 0; the kernel measurably degrades agreement with the reference.
- The expert keep-set is ranked on `results/trace-union` (web, code, configuration, technical prose
  and narrative fiction; 190 sequences, 36,250 tokens) instead of a 50-document Python-only corpus.

### Added
- `corpus/trace_corpus_v2.jsonl`, `corpus/trace_corpus_v3.jsonl` and their sources under
  `corpus/sources/web` and `corpus/sources/prose`.
- Per-category histograms in `coverage.json`, so a keep-set can be built without the raw trace.
- `presence_penalty` / `frequency_penalty` per request and `DSV41_NO_REPEAT_NGRAM`, all defaulting
  to 0 — a frequency penalty fixes prose repetition and corrupts CSS, so none of them ships on.
- `engine/test_spec_lossless.py`; grammar-constrained DSML tool calls via xgrammar
  (`server/tool_grammar.py`, `DSV41_TOOL_GRAMMAR=1`, off by default).
- The engine refuses to start when host memory cannot hold the arena instead of squeezing.

### Changed
- Shipped defaults: `PRUNE_KEEP=0.44 EXPERT_FORMAT=cb3 ARENA_GB=98 TRANSIENT_SLOTS=8 KEEP_FREE_GB=6`
  with the union trace. 44.1 % of routed experts resident, hit rate 1.0.

### Measured on it
17-37 tok/s across nine workloads, prefill 337 tok/s on a 5,014-token prompt, every case passing a
900-2,000-token generation gate with structural checks (RESULTS.md v0.4.0-wip).

## 0.3.0-wip — 2026-09-11

**The shipped default changes: keep 40 % of the routed experts, all resident in the 3-bit CB3
format.** Decode 16.6 → 19.0 tok/s and held-out loss −0.03 (code) / −0.17 (prose) nats against
the 0.2.0-wip default, on the same box (RESULTS.md v0.3.0-wip).

### Added
- `EXPERT_FORMAT=cb3` / `--expert-format cb3`: the resident arena holds 3-bit per-row codebook
  experts (`tools/cb3.py`, 14.45 MB each), packed on the GPU at warm start from the FP4 shards;
  decode runs the CB3 v3 Triton kernel (`tools/cb3_moe.py`, 182 GB/s of expert bytes, 0.79x the
  FP4 kernel's time per expert); prefill unpacks to FP4 codes and runs the FP4 kernel. Unit test
  `tools/test_cb3_moe.py`.
- `tools/fp8_linear.py::fp8_grouped_linear`: the `wo_a` projection runs from its stored FP8
  (`DSV41_WOA_FP8=0` restores the bf16 einsum).
- `tools/decode_attn.py`: fused decode attention (bf16 keys, fp32 softmax with the sink, two KV
  segments without a copy; `DSV41_FUSED_ATTN=0` restores the fp32 torch path).
- `tools/fp32_skinny.py`: split-K fp32 kernel for the Hyper-Connection mixing GEMMs
  (`DSV41_HC_KERNEL=0` restores `F.linear`).
- `--prune-select global` / `PRUNE_SELECT`: cross-layer keep-set ranking; measured worse than
  uniform on the held-out corpus (NOTES.md 2026-09-11 10:00) and left as a documented option.
- CUDA graphs per step segmented at the Engram layers (`DSV41_GRAPH_SEGMENTS=0` restores per-layer
  graphs; no measurable gain either way), pinned Engram staging (`DSV41_ENGRAM_PINNED=1`, off).
- LM head and DSpark Markov head loaded in their stored bf16 (`DSV41_HEAD_FP32=1` restores fp32).

### Changed
- Default configuration in `env.example`/`docs/install.md`: `PRUNE_KEEP=0.40 EXPERT_FORMAT=cb3`
  for a 128 GB box. Warm start is 183 s in this format (19 s for FP4).
- Verify step with everything resident: 168 → 147 ms (FP4, keep 31 %); RESULTS.md addenda 2.9-2.11.

### Measured on it
RESULTS.md v0.3.0-wip: 18.98 tok/s greedy decode, TTFT 9.81 s on a 1,806-token prompt, held-out
1.5384 / 3.2087 nats. Not measured: thinking-on in this configuration, 8k+ prompts in this
configuration, sampled A/B, the image end to end.

## 0.2.0-wip — 2026-09-11

**The model math fix and the resident pruned configuration.** Everything in 0.1.0-wip ran on a port
with a transposed Hyper-Connection residual mix; this tag fixes it and rebuilds the decode path.

### Fixed
- `tools/v41_ref.py::hc_post`: sum over the first index of `comb` (combᵀ · residual), as in the
  reference `Block.hc_post`. Teacher-forced coding loss 2.16 -> 1.37 nats; greedy output no longer
  stutters; DSpark acceptance 2.4 -> 3.75 on code. (commit bd24743, 2026-09-11 00:50)
- Transient prefill ring must hold a whole layer (>= 384) unless every routable expert is resident.

### Added
- `engine/fastdecode.py`: CUDA-graph decode path (per-layer graphs, host slot resolve between them),
  fused HC Sinkhorn Triton kernel (`engine/hc_sinkhorn.py`), bf16 head, masked fixed-length indexer.
  Verify step 436 -> 173 ms with everything resident.
- `tools/fp8_linear.py`: dense projections in stored FP8 (Triton, 1.9x bf16 GEMM at decode size);
  `v41_ref.dense()` dispatch; `DSV41_DENSE_FP8=0` restores bf16 copies.
- Pruned all-resident serving: `--prune-keep F` (router restricted to the top-F experts per layer by
  trace frequency, exactly those warm-started), `--prune-sweep` teacher-forced ladder,
  `--transient-slots`, `--keep-free-gb`, `--arena-gb` pinned sizing.
- `engine/codebook_sim.py`, `tools/cb3.py`, `tools/cb3_moe.py`: 3-bit per-row codebook expert format
  (simulation, packer, and a correct-but-slow kernel).
- `engine/diag_decode.py` (decode == prefill consistency, per layer), `engine/test_fastdecode.py`,
  `engine/profile_decode.py`, `engine/profile_fast.py`.
- Held-out corpus `corpus/heldout_corpus.jsonl` (sources in `corpus/heldout_sources/`).

### Measured (RESULTS.md §v0.2.0-wip)
keep 31 % resident: 12.9 tok/s at +0.07 / +0.19 nats; unpruned streaming 3.5 tok/s; full ladder there.

### Process
Conventions applied from this tag on: no benchmark sweeps (single decode numbers only); docs
append-only with dates and per-tag sections; credits limited to the model vendor, the author's other
Spark recipes and the toolchain.
