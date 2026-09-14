# Keep-sets, topics and coverage

The three ideas `./tune.sh` works with. The screen that uses them is described in
[`docs/tune.md`](tune.md), field by field in [`docs/tune-reference.md`](tune-reference.md); the
memory those ideas cost is in [`docs/memory-budget.md`](memory-budget.md); the shipped profiles and
what each was measured to be good and bad at are in
[`results/keepsets/README.md`](../results/keepsets/README.md).

## A keep-set is a cache policy

The model has 40 layers of 384 routed experts, 15,360 in all, and every token activates 6 of them in
every layer. At the FP4 layout the checkpoint ships, all 15,360 are 288.8 GB; a GB10 has about
121 GiB of unified memory for everything. So most of them cannot be resident, and the engine has to
choose.

A **keep-set** is that choice made once, before the server starts: per layer, the top *N* experts
stay in the arena and the router may only pick among them, where *N* is `ceil(PRUNE_KEEP x 384)`.
Nothing is requantised and nothing is approximated — the experts that stay are the checkpoint's own.
The experts that do not stay are simply not routable.

That is what makes it a cache policy rather than a compression scheme, and it is why the choice
matters so much: a policy is only as good as the workload sample it was learned from.

*N* is decided by a measured routing trace. `tools/expert_trace.py` pushes a corpus through the
model one layer at a time and records which experts each token picked; `tools/expert_stats.py`
turns that into the per-layer histograms in a `coverage.json`. The engine ranks each layer's 384
experts by those counts and keeps the top *N* — or by how much each expert *contributed* rather than
how often it was picked, which is [`DSV41_PRUNE_SOURCE`](#frequency-is-not-contribution).

## A topic is one per-layer histogram

`corpus/make_corpus.py` tags every sequence in the trace corpus with a category, and
`tools/expert_stats.py` writes a separate 384-wide histogram per category per layer
(`counts_<topic>` inside `coverage.json`) alongside the mixed one. A **topic** is one of those: the
routing a corpus of that subject alone produced, 40 layers deep.

Two consequences follow, and they are the reason the tool exists.

* One trace yields every topic in it. The trace is one pass over the corpus per layer and costs
  about the same whether the corpus carries five topics or thirty-five, so it is worth tagging
  generously.
* Composing a keep-set out of a subset of those topics is arithmetic on numbers already in the
  checkout. No GPU, no new trace, no model load — which is what makes a live screen possible at all.

`EXPERT_TOPICS` names the subset. Unset means every topic in the file, which is what the shipped
profiles do.

## How a selection becomes a ranking

For each selected topic, each layer's 384 counts are divided by their own sum. The normalised
histograms are added, and the top *N* of that sum is the layer's keep-set
(`engine/v41_engine.py`, and the same arithmetic in `tools/budget.py` so the screen agrees with the
engine).

That agreement holds for all three rules since 2026-09-12: `tools/budget.py` reimplements `max` and
the `maxmin` rule below as well, defaults to whatever `DSV41_PRUNE_RANK` is set to, takes
`./tune.sh --rank sum|max|maxmin`, shows the rule beside the keep fraction on both screens and
writes it into `.env` with them. `tools/test_budget_rank.py` lifts the engine's own function out
with `ast` and holds the two implementations to the same keep-set, layer by layer, as sets, at
seven keep fractions — because a screen that ranks by one rule while the engine ranks by another
reports coverage nothing will deliver: over the seven topics below at keep 0.36 that gap is 0.15 on
`english` alone.

Normalising before summing is the whole point. Without it a topic that contributed 20,000 tokens
would outvote one that contributed 3,000 by a factor of seven, and "serve this workload too" would
quietly mean "serve it if it happens to be the bigger half of the corpus". With it, every selected
topic gets one vote per layer.

The alternative combination — keeping an expert that matters to *any* selected topic, rather than
summing — exists as `DSV41_PRUNE_RANK=max` and was measured worse, so `sum` remained the default
until `maxmin`, below.

### One vote each is not the same as one outcome each

`sum` maximises the total routing mass the resident set keeps. That is the wrong quantity when one
request spans several topics at once, which is what a coding request with thinking on does: it
writes English prose, deliberation, HTML, CSS and JavaScript in a single generation, and it
degenerates at whichever of them the resident set serves least. A rule that maximises the total is
free to let an already well-served topic go on taking slots while another starves.

Normalising per layer has a second consequence that is easy to miss: it divides corpus size out, so
a *broad* topic — one that spreads its mass over many experts — scores low on every one of them and
loses slot after slot to a peaky specialist. Over
{english, html, python, reasoning, css, javascript} at `PRUNE_KEEP=0.36` on a GB10, `sum` leaves
english at 0.556 while css and javascript sit at 0.803 and 0.814.

`DSV41_PRUNE_RANK=maxmin` hands each layer's slots out one at a time to whichever selected topic is
currently least covered. An expert admitted for one topic counts for every topic that also routes
to it, so overlap is paid for once instead of per topic, and the selected topics converge on a
common coverage rather than a spread. The same budget then holds 0.676–0.688 across all six.

It also changes what adding a topic costs. Under `sum` the worst-served topic falls from 0.657 at
four topics to 0.410 at eighteen; under `maxmin` the same span costs 0.06. Breadth is affordable
under `maxmin` and ruinous under `sum` — but neither rule creates capacity, and a wide enough
selection runs every topic down. Measured on this box at `PRUNE_KEEP=0.36`, 139 experts a layer
(2026-09-12): under `maxmin` about **sixteen** topics is where the worst-served one lands at 0.671,
below the 0.7 line; 35 of the catalogue's 36 topics together run 0.580–0.636, with english the
hardest. The same full selection under `sum` puts chinese hardest at 0.464. Sixteen is this box at
this keep fraction, not a property of the rule — the boundary moves with both.

## Frequency is not contribution

Everything above ranks experts by how **often** the router picked them. That is one measurement of
an expert's worth and it is not obviously the right one.

REAP (Lasby et al., Cerebras, ICLR 2026, [arXiv 2510.13999](https://arxiv.org/abs/2510.13999))
benchmarked the alternative on Kimi-K2 — 384 routed experts, one shared, auxiliary-loss-free
routing, the same shape as this model — and the gap is not subtle:

| criterion | LiveCodeBench, 75 % of experts kept | 50 % kept |
|---|---|---|
| routing frequency | 0.082 | 0.000 |
| saliency | 0.440 | 0.429 |

(dense baseline 0.434/0.440.) Frequency-based pruning does not degrade there; it collapses.
Our own generation gate shows that failure mode at 36 % kept: the keep-set scores well on every
selected topic's coverage and the output still degenerates.

**Saliency** is the magnitude an expert actually contributes. For expert *e* in layer *L*, over a
calibration corpus, REAP defines it as the mean over the tokens routed to *e* of

```
gate_weight(t, e) · ‖ expert_e(x_t) ‖₂
```

— the gate weight the router gave that pick, times the L2 norm of that expert's output for that
token, taken *before* the weight is applied. An expert the router reaches for constantly whose
output barely moves the residual stream is at the top of a frequency ranking and near the bottom of
a saliency one, and a keep-set built the first way spends its budget on it.

This repository stores the **sum** over those tokens rather than the mean. Every rule downstream
normalises a layer's histogram by its own total before ranking it, so mean and sum differ by exactly
the count factor; the mean asks *how much does this expert contribute when it fires*, the sum asks
*how much of this layer's output does it account for*, and the second is the question a cache policy
asks. The sum is also literally frequency × magnitude, which is what makes the two families the same
measurement with and without the magnitude factor — an expert that fired once at enormous magnitude
tops a mean ranking and is worth nothing to a keep-set.

`tools/expert_trace.py` records, per token and per pick, the gate weight (`weights`, post
renormalisation and post `route_scale` — the number actually multiplied into the expert's output)
and `out_norms`, the unweighted `‖expert_e(x_t)‖₂`. `tools/expert_stats.py` sums `weight × out_norm`
per expert into `saliency_<topic>` beside `counts_<topic>` in `coverage.json`.

**`DSV41_PRUNE_SOURCE=counts|saliency`** picks which family the engine ranks a keep-set by. The
ranking rules are untouched: `sum`, `max` and `maxmin` see 384 non-negative numbers per layer either
way and cannot tell which measurement produced them. `./tune.sh --source counts|saliency` reads the
same variable, shows the family beside the rank on both screens and writes it into `.env`;
`tools/test_saliency.py` holds the screen's saliency keep-set to the engine's, layer by layer, under
all three rules, the way `tools/test_budget_rank.py` does for counts.

A `coverage.json` built before 2026-09-13 carries every `counts_<topic>` and no `saliency_<topic>`,
and the engine **refuses by name** rather than falling back to frequency. Getting them needs a fresh
trace — the norms are taken inside the layer forward and cannot be recovered from the stored
indices:

```bash
python tools/expert_trace.py --model-dir ./models/DeepSeek-V4.1-Flash \
    --corpus corpus/trace_corpus.jsonl --engram-dir engram_rows \
    --out results/trace-YYYYMMDD
python tools/expert_stats.py --trace results/trace-YYYYMMDD \
    --out results/trace-YYYYMMDD/stats
./tune.sh --source saliency --stats results/trace-YYYYMMDD/stats/coverage.json --write
```

The trace is resumable as before (`--resume`), and the new arrays are the same size as `indices`, so
a trace costs what it always did. What has **not** been done here is the measurement that matters:
no generation gate has been run on a saliency keep-set on this box. REAP's numbers are REAP's, on
another model and another benchmark. Until that run exists, `counts` stays the default.

## Coverage

**Coverage** is the fraction of a topic's *measured* routing that lands on an expert the current
keep-set holds resident. It is read off a cumulative curve: index *n* of the curve is the routing
mass captured by keeping the top *n* experts of every layer, so the number on screen is
`curve[ceil(keep x 384)]`.

It is a measurement, not an estimate — the histogram it is computed from is the routing that
actually happened on that corpus. It is monotone in the keep fraction and reaches 1.0 at 100 %.

Coverage depends on the *selection*, not only on the keep fraction, because the selection decides
the ranking. On the shipped 2-topic keep-set at `PRUNE_KEEP=0.39`:

| selected | coverage of `coding` | coverage of `general` |
|---|---|---|
| `coding` | 0.90 | 0.51 |
| `general` | 0.44 | 0.85 |
| both | 0.83 | 0.75 |

Both topics are in the file either way. Selecting one does not remove the other from the world; it
spends the budget on the first, and the screen shows what that costs the second in the same glance.

## Why coverage predicts whether long generations hold together

An expert that is not resident is not routable. When the router's first choice for a token is
outside the keep-set, the token goes to its second choice, and then to the second choice again on
the next token, and the one after that. The errors do not cancel — they compound into repetition,
and then into structurally broken output.

That substitution is a choice, and `DSV41_PRUNE_MODE` makes it one. The engine masks the router's
logits to the resident experts and takes the top-*k* among those (`substitute`, the default and the
only behaviour before this switch existed), so a token whose real top-6 is not resident is computed
with six experts it did not ask for, each at full renormalised weight. `drop` instead keeps the
router's real top-6 and gives every pick that did not survive a weight of exactly 0, renormalising
over the rest; a token with no resident pick at all falls through to the layer's shared expert
alone. Because the survivors are renormalised to the routed scale — `norm_topk_prob` is on in this
checkpoint, and the technical report keeps the correction bias for selection only — `drop` is
top-*k′* routing with *k′* the picks that survived, about four of six at the keep fractions here;
it is not an attenuation toward the shared expert. The reason to try it is that a substituted
expert injects a signal the model was never trained to receive, and the mHC residual feeds that
error into the next layer's mixing coefficients as well. Measured on Frontend at keep 0.36 with thinking on (2026-09-13): six of six prompts failed under
`drop`, including a page that passes under `substitute`. Fewer correct experts do not beat six
partly wrong ones here; `drop` is kept as a documented negative result. The motivation is a profile: at the keep fractions here about 30 % of the routing
mass is displaced *however* the keep-set is chosen, and on the 2026-09-12 generation gate that
showed up as rare tokens corrupted at subword boundaries — `clearTimeout` as `cleartimeout`,
`OSError` as `oenerror`, `.some` as `.s.s` — which the model then loops trying to repair. Changing
the keep-set cannot fix that; changing what a displaced pick *does* might. `drop` is an experiment
and has not been through the generation harness, so it is off by default and `substitute` is
unchanged to the bit.

This is not a theory about the format or the kernels. It is what this repository spent days chasing:

| keep-set, all at keep 44 % and otherwise identical | story | Python | HTML |
|---|---|---|---|
| a corpus whose only content marker was Python | 0.27 | 0.51 | **0.03** |
| the same, plus a 12-gram repeat ban | 0.32 | 0.51 | **0.07** |
| a corpus of web, code, configuration and technical prose | 0.43 | 0.50 | 0.40 |
| a corpus of narrative fiction and dialogue | 0.54 | 0.58 | **0.04** |
| the union of the last two | **0.56** | **0.47** | **0.59** |

(distinct-token ratio of the generated text; 0.15 or below is degenerate. `RESULTS.md` §4.2.)

The first row wrote `<!DOCTYPE><!DOCTYPE><!DOCTYPE>` for as long as it was allowed, and measured
*better* on teacher-forced loss than the configuration that replaced it. Markup coverage in that
keep-set was 0.03. Raising it to 0.40 fixed the generation. A corpus of fiction fixes prose and
loses markup; a corpus of code does the reverse; at 44 % of the experts the two rankings compete
for the same slots.

So the useful reading of the screen is not "how many topics" but **"what is the weakest selected
topic's coverage"**. Below about 0.7 is where generation starts to degrade, which is why the bars
turn amber there and red below. The default target of 0.85 is not a universal constant: it is where
the shipped keep-sets sit for the domains they were built for.

Two limits on that reading, both recorded in [`LIMITATIONS.md`](../LIMITATIONS.md):

* The generation gate that produced those rows runs to 2,000 tokens. Past that, generations still
  degenerate at the shipped keep fraction, and the cause is not yet attributed.
* Coverage says nothing about speed. A step reads the experts the token activates either way, so
  choosing fewer topics buys a smaller keep fraction rather than a faster step; see
  [`docs/tune.md`](tune.md#fewer-topics-are-not-faster-under-sum-they-are-cheaper-under-maxmin-breadth-is-cheap-but-not-free) for the mechanism, the one
  measurement that supports it, and the A/B that has not been run.

## How many experts a step actually reads

Six per layer is the per-token figure, and it is not the figure a decode step pays. DSpark verifies
a block of six tokens at once, and the routing of six consecutive tokens overlaps heavily, so the
36 `(token, expert)` pairs of a layer collapse to far fewer distinct reads — but to noticeably more
than six.

| measurement | value | how it was taken |
|---|---|---|
| distinct experts per layer per verify block | **20.96** | counted on the real decode path over ~20 verify blocks at keep 40 %, `RESULTS.md` §3.5 |
| `block6_unique_mean` in `coverage.json` | 22.6 mean, 18.2 to 29.5 by layer | counted on the trace, with **no** keep mask applied |

The two are close but they answer different questions, and the difference is the reason the speed
question in [`docs/tune.md`](tune.md#fewer-topics-are-not-faster-under-sum-they-are-cheaper-under-maxmin-breadth-is-cheap-but-not-free) is still open:
`block6_unique_mean` is measured without a keep-set, so it cannot say whether a keep-set matched to
its workload concentrates routing and lowers the count. Only an A/B at a fixed keep fraction, one
topic against many, can, and it has not been run.

What the first row does establish is the read volume a step pays: about 21 experts per layer across
40 layers at 14.45 MB a slot is 12.12 GB of expert reads per step, which over the 65.0 ms the two
expert kernels take is 186 GB/s — the rate the kernel reaches in isolation. The floor is real, and
the only way under it is fewer reads.

## A thinly traced topic reports coverage that is too high

Each token contributes 6 picks per layer spread over 384 experts. A topic traced on 300 tokens
therefore leaves each layer with 1,800 picks across 384 experts — a mean of under 5 each, so most of
the ranking is the difference between one count and two. At 2,000 tokens it is 12,000 picks, a mean
of about 31, which is where the tool stops calling a topic thin.

The problem is worse than noise, because **coverage is measured on the very trace that chose the
experts**. A topic seen for 300 tokens routes to whatever fired during those 300 tokens, and those
are exactly the experts its histogram ranked highest, so it scores as though it were well served.
The bias is upward and it is systematic.

The tool therefore draws such a topic's bar hollow, prints its traced token count next to it in red,
and puts a warning on the line above the keys whenever one is selected. Treat a hollow bar's number
as an upper bound. A few thousand tokens a topic is what makes a bar worth believing;
`corpus/fetch_topics.py` collects about that much and flags any topic it could not.

## Profiles and topics

A **profile** is a whole `coverage.json` directory under `results/keepsets/`, selected with
`EXPERT_PROFILE`. A **topic** is one histogram inside such a file, selected with `EXPERT_TOPICS`.
They compose: the profile decides which measurements are on the table, the topic selection decides
how the budget is spent across them.

Not every shipped profile carries topics. `results/keepsets/general/` has two (`coding` and
`general`); `code` and `prose` carry only the mixed histogram, so `./tune.sh` shows the budget panel
for them but no topic list, and `EXPERT_TOPICS` cannot be used with them. Each profile ships a
`GATE.md` recording which domains were measured, which passed and which failed. Read it before
choosing one.

## A small topic under `maxmin` reshuffles the tail (2026-09-14)

`maxmin` water-fills to the least-covered selected topic, so a topic whose saliency histogram is
spread thin — a 40-record deliberation corpus split over four languages, for instance — keeps
asking for experts and keeps getting them. The coverage screen cannot see the cost: saliency mass
is so top-heavy (half of it sits on one expert a layer) that swapping the tail moves every bar by
about 0.005. The keep-set moves a great deal more. Adding `reasoning_lang` to the nine European
topics at keep 0.36 swapped 513 of the 5,560 resident experts, 3 to 28 a layer and most of them
in layers 36–39; together with the 600 that 0.40 → 0.36 drops, the run was missing 1,059 experts
of the 0.40 keep-set it was meant to improve on. The gate went from 6 of 10 to 3 of 10, and the
two rows that fail loudest (French and German deliberation) are exactly the ones that route into
that tail. Read a topic's traced token count and its `n80` (experts holding 80 % of its mass)
before adding it to a bundle: `reasoning_lang` needs 8 a layer where `reasoning_design` needs 1,
and the latter is the one that helped Frontend.

The same swap, gated on the World bundle the same night, went the other way: 3 of 10 at keep 0.40
without the topic, 6 of 10 at 0.36 with it (Chinese and Russian deliberation now finish; Arabic,
Japanese and Turkish still loop). Which direction a tail reshuffle takes is not readable off the
coverage screen; only a gate says. The topic ships in World and not in European.

Correction, later the same day: "most of them in layers 36–39" overstates it. Recomputed from the
shipped file, those four layers hold 98 of the 513 swapped experts — the most per layer, not a
majority. The 513 and the 3-to-28-a-layer range stand.

## What predicts the gate (2026-09-14)

The section above ends on "only a gate says", and that is an uncomfortable place to leave a tool
whose whole job is to answer the question without running one. So the coverage bar was measured
against the gate records, and so were four alternatives to it. `tools/tail_metric.py` is the
runner; `results/keepsets/tail_metric.json` is its output, with the `GATE.md` record each row came
from; `TopicIndex.tail_curves` in `tools/budget.py` computes the new numbers.

### The candidates

Coverage asks how much of a topic's *measured routing mass* stayed resident. What
`DSV41_PRUNE_MODE=substitute` actually does to a generation is narrower: it masks the router to
the resident experts and takes the top-6 of what is left, so a token whose six picks **in one
layer** are not all resident is computed with experts it did not ask for, at full renormalised
weight. That is a property of a token's tail, not of a corpus's mean, and four numbers get closer
to it:

| number | what it is | exact? |
|---|---|---|
| `cov (pick)` | the same keep-set measured on `counts_<topic>` instead of the family that ranked it | exact |
| `nonres` | expected non-resident picks per token per layer, 0 to 6 | exact |
| `all6` | the rate at which all six picks of a token-layer are resident | estimated |
| `worst layer` | the worst single layer's resident pick fraction | exact |

`nonres` is exact and needs no assumption at all, because `counts_<topic>` **is** the histogram of
picks and every token contributes exactly six of them per layer: the mean of a per-token count is
the missed mass times six. It follows that `nonres = 6 x (1 - cov(pick))` identically — the mean is
not new information, it is coverage in the units a token pays. The new information is `all6`, and
that one cannot be read off a histogram, because it needs the joint distribution of the six picks.
It is estimated as `mean_L r_L^6`, i.e. as though the six were independent draws. They are not —
experts co-fire — so the estimate is always **low**. Measured against the per-token `indices`
arrays of `results/trace-full-20260910` over twelve (topic, selection, keep) points from 0.20 to
0.40: 6 % to 75 % low, worst on a topic the keep-set was not spent on, rank correlation with the
truth 0.96. Read it as a ranking statistic, not as a rate. `tools/test_tail_metric.py` holds both
of those numbers.

One reading that sounds right and is not: *the fraction of tokens whose routing is clean*. Over
forty layers that is a product of forty terms, and on the real trace it is zero to five decimal
places at every keep fraction this repository has ever served at — 0.20, 0.36 and 0.40 alike.
Essentially every token is substituted somewhere. The quantity with any resolution is the
token-**layer**, which is what `all6` counts.

**What is shipped and what is not.** Only the histograms ship. `results/keepsets/topics/
coverage.json` carries `counts_<topic>` and `saliency_<topic>` for 39 topics and no per-token
arrays; the trace they were made from is not in the checkout. The one per-token trace that is —
`results/trace-full-20260910`, 40 layers, 10,760 tokens, two categories, taken before the saliency
tracer — is what calibrates the estimate above and nothing else.

### The ten shipped profiles, each at the keep its gate was run at

| profile | keep | strict | finished | hard | cov (rank) | cov (pick) | nonres | all6 | worst layer |
|---|---|---|---|---|---|---|---|---|---|
| Frontend | 0.36 | 7/10 | 9/10 | 1 | 0.9402 | 0.5976 | 2.414 | 0.0565 | 0.4551 |
| Backend | 0.36 | 3/10 | 10/10 | 0 | 0.9460 | 0.5990 | 2.406 | 0.0544 | 0.4679 |
| Systems programming | 0.40 | 6/11 | — | — | 0.9485 | 0.6398 | 2.161 | 0.0788 | 0.5023 |
| Chat and explanation | 0.40 | 6/8 | — | — | 0.9645 | 0.7058 | 1.765 | 0.1342 | 0.6059 |
| Medicine | 0.40 | 6/7 | — | — | 0.9641 | 0.7014 | 1.792 | 0.1289 | 0.6018 |
| Law and finance | 0.40 | 5/7 | — | — | 0.9649 | 0.7061 | 1.763 | 0.1342 | 0.6016 |
| Data and research | 0.36 | 8/11 | 10/11 | 1 | 0.9449 | 0.5947 | 2.432 | 0.0526 | 0.4630 |
| European languages | 0.40 | 6/10 | — | — | 0.9610 | 0.6894 | 1.864 | 0.1180 | 0.5865 |
| World languages | 0.36 | 6/10 | 7/10 | 3 | 0.9360 | 0.5784 | 2.530 | 0.0527 | 0.3860 |
| Writing | 0.40 | 5/8 | — | — | 0.9648 | 0.7074 | 1.756 | 0.1357 | 0.6040 |

Every column is the profile's **worst** topic: a request spans the whole bundle and comes apart at
whichever topic the keep-set serves least. `hard` is runs − finished, the misses that are not a
redraft; only the four cards written on 2026-09-14 carry the second verdict sentence, so the other
six have no such count.

### Against the gate

Spearman, with a p from a seeded permutation test — not a table, because n is ten and the
asymptotic approximation is poor there. `max/min` is the ratio of the largest to the smallest value
across the ten keep-sets: what the number can show a reader at all, before the question of whether
it predicts anything.

| predictor | want | max/min | vs strict (n=10) | vs finished (n=4) | vs hard misses (n=4) |
|---|---|---|---|---|---|
| coverage, ranking family, worst topic | + | 1.03x | 0.353 (p=0.32) | 1.000 (p=0.08) | −0.949 (p=0.16) |
| coverage, ranking family, mean | + | 1.02x | 0.365 (p=0.30) | 0.400 (p=0.75) | −0.632 (p=0.50) |
| coverage in picks, worst topic | + | 1.22x | 0.304 (p=0.40) | 0.800 (p=0.33) | −0.949 (p=0.16) |
| non-resident picks, worst topic | − | 1.44x | −0.304 (p=0.40) | −0.800 (p=0.33) | 0.949 (p=0.16) |
| non-resident picks, mean | − | 1.41x | −0.328 (p=0.36) | −1.000 (p=0.08) | 0.949 (p=0.16) |
| all-six-resident rate, worst topic | + | 2.58x | 0.310 (p=0.39) | 0.000 (p=1.00) | −0.316 (p=1.00) |
| all-six-resident rate, mean | + | 2.18x | 0.328 (p=0.36) | 1.000 (p=0.08) | −0.949 (p=0.16) |
| **worst layer's resident pick fraction** | + | 1.57x | **0.438 (p=0.21)** | 1.000 (p=0.08) | −0.949 (p=0.16) |

### The verdict

**Nothing computable from a `coverage.json` predicts the gate at significance, the shipped bar
included.** Worst-layer resident pick fraction is the best of the eight against strict passes
(rho 0.438) and the shipped coverage bar the third (0.353), but at n = 10 the 0.05 threshold is
around |rho| = 0.65 and no candidate is close. The last two columns look decisive and are not:
they are four points, where a perfect rank correlation still only reaches p = 0.08.

Two things make the cross-profile comparison weaker than its n suggests, and both are worth stating
rather than correcting away:

* **Every profile is a different exam.** Medicine's 6 of 7 and Backend's 3 of 10 are different
  prompts, checked by different structural rules. Ranking ten profiles by pass rate ranks the
  suites as much as the keep-sets.
* **The gate is ten Bernoulli trials.** At 10 runs and a pass rate near 0.6 the standard error is
  about 1.5 runs, so a three-run swing is barely two sigma. The gate cannot resolve what these
  predictors differ by.

The measurement that removes the first confound is the same profile gated twice on the same prompts
with only the keep-set changed. Four exist, and they are the runs that discredited the coverage bar
in the first place:

| profile | keep | strict | Δ cov (rank) | Δ cov (pick) | Δ nonres | Δ all6 | Δ worst layer |
|---|---|---|---|---|---|---|---|
| European languages | 0.40 → 0.36, +`reasoning_lang` | 6/10 → 3/10 | −0.0174 | −0.0778 | +0.467 | −0.0509 | −0.1432 |
| World languages | 0.40 → 0.36, +`reasoning_lang` | 3/10 → 6/10 | −0.0155 | −0.0591 | +0.355 | −0.0305 | −0.0896 |
| Backend | 0.40 → 0.36 | 5/10 → 3/10 | −0.0085 | −0.0490 | +0.294 | −0.0291 | −0.0499 |
| Data and research | 0.40 → 0.36 | 5/11 → 8/11 | −0.0080 | −0.0509 | +0.305 | −0.0296 | −0.0623 |

Every predictor says "worse" in all four, because every swap is a drop in the keep fraction and
every one of these numbers is monotone in it. The gate went two ways. **2 of 4 for all eight
candidates, the tail metric included** — and the same holds one level down, per prompt: in the two
language swaps, every topic's tail number falls, while `en-explain`, `ru-essay` and `zh-essay` flip
to PASS and `en-explain`, `pt-essay` and `xl-en-fr` flip to FAIL. No scalar on this file separates
the two directions.

So: **the tail metric is a better instrument and not yet a better predictor.** What it demonstrably
buys is resolution. The coverage bar puts the ten shipped keep-sets inside 1.03x of each other —
0.936 to 0.965, a band narrower than the 0.005 a 513-expert swap moves it by, which is why that
swap was invisible. `all6` puts the same ten across 2.58x, and moves by 0.03 to 0.05 on the swaps
above, six to ten times the bar's own movement. A number that can show a change happened is worth
having even before it can say which way the change will go.

What it does not buy is permission to skip the gate. Read the tail alongside the bar, and keep
reading `GATE.md`.

Reproduce, or re-run after a fresh gate:

```bash
python3 tools/tail_metric.py                                       # the tables above
python3 tools/tail_metric.py --profile frontend                    # one profile
python3 tools/tail_metric.py --json results/keepsets/tail_metric.json
python3 tools/test_tail_metric.py                                  # incl. the exact-vs-estimate check
```
