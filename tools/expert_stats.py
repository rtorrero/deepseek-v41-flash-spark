#!/usr/bin/env python3
"""
expert_stats.py -- turn the per-layer router traces from expert_trace.py into the numbers
that decide the expert strategy (Phase 1):

  * per-layer expert usage histogram (share of routed slots per expert)
  * per-layer expert SALIENCY histogram: how much magnitude each expert contributed, not how
    often it was picked (`saliency_<topic>` next to `counts_<topic>`; see `saliency_hist`)
  * per-layer TOP-1 histogram: of the six experts a token takes, which one carried the largest
    gate weight (`top1_<topic>`; see `top1_hist`). `counts` asks how often an expert was taken,
    `top1` how often it led -- the Weight Atlas draws the second as `top1_share`
  * per-layer CO-ROUTING table: which two experts a token tends to take together, the top pairs
    by count and by lift among the C(topk, 2) pairs of each token (`pairs_<topic>`; see
    `pair_table`). Written to a sibling `pairs.json` by default -- see `--pairs`
  * per-layer and global cumulative coverage curves: fraction of routed slots covered by
    the top-N% (or top-K) experts; global = best allocation across layers, i.e. experts
    ranked by frequency over all layers
  * block-level unique experts: DSpark verifies blocks of up to 6 tokens (1 + 5 drafts)
    at once, so what matters for a streaming cache is the union of experts touched by
    consecutive tokens, per layer
  * cache simulation: LRU over (layer, expert) with a given number of resident experts
    (FP4, 18.8 MB each), hit rate per token and per 6-token block, split by category
  * memory projection for strategies A (hot cache + stream) and C (hot FP4 + cold low-bit)

Writes results/<name>/{coverage.md, coverage.json, coverage.png, layer_hist.png}, and
results/<name>/pairs.json unless `--pairs` says otherwise.

`--merge SHIPPED --runs DIR ...` is the other half: it rebuilds an existing stats file out of
fresh reductions and replaces it only if every `counts_<topic>` and `saliency_<topic>` comes
back element-for-element identical. A shipped file is a merge of several traces -- and at least
one topic is the SUM of two of them -- so which traces make up a topic is derived by matching
the histograms, never assumed from a directory name. See `merge_reductions`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

EXPERT_BYTES_FP4 = 3 * (5_898_240 + 368_640)  # w1/w2/w3 packed FP4 + UE8M0 scales, from the safetensors header
N_EXP = 384


def load(trace_dir: str):
    layers = {}
    for p in sorted(glob.glob(os.path.join(trace_dir, "trace", "layer*.npz")),
                    key=lambda p: int(re.search(r"layer(\d+)", p).group(1))):
        L = int(re.search(r"layer(\d+)", p).group(1))
        z = np.load(p)
        d = {"idx": z["indices"].astype(np.int64), "w": z["weights"].astype(np.float32), "cat": z["category"]}
        # `out_norms` arrived with the saliency tracer (2026-09-13). A trace taken before it has
        # every other array and must still process: the layer simply gets no saliency histogram.
        if "contrib_norms" in z.files and z["contrib_norms"].shape == d["idx"].shape:
            # gate_weight * ||expert(x)||, stored bounded and in fp32 since 2026-09-13
            d["sal"] = z["contrib_norms"].astype(np.float32)
        elif "out_norms" in z.files and z["out_norms"].shape == d["idx"].shape:
            # The first saliency tracer stored ||expert(x)|| = ||contrib|| / weight in fp16, which
            # overflowed on a few hundred picks in the deepest layers. weight * out_norm recovers
            # ||contrib|| where it is finite; where it is not, the pick is clamped to the layer's
            # largest finite contribution and counted, so a keep-set can still be built from the
            # trace while the number of clamped picks stays on the record.
            sal = d["w"] * z["out_norms"].astype(np.float32)
            bad = ~np.isfinite(sal)
            if bad.any():
                sal[bad] = sal[~bad].max() if (~bad).any() else 0.0
                d["clamped"] = int(bad.sum())
            d["sal"] = sal
        layers[L] = d
    meta = json.load(open(os.path.join(trace_dir, "meta.json")))
    return layers, meta


def saliency_hist(idx: np.ndarray, sal: np.ndarray) -> np.ndarray:
    """REAP's saliency (Lasby et al., Cerebras, ICLR 2026, arXiv 2510.13999) as a 384-wide
    histogram: for expert e, the total of `gate_weight(t, e) * ||expert_e(x_t)||` over the tokens
    routed to it -- the magnitude it actually contributed to the residual stream.

    SUM, where REAP's definition is a MEAN over the tokens routed to the expert. The engine
    normalises every histogram by its own per-layer total and then takes the top N, so a mean and
    a sum differ by exactly the count factor: the mean asks "how much does this expert contribute
    WHEN it fires", the sum asks "how much of this layer's output does it account for". The second
    is the question a keep-set asks, and it is the one that composes with `counts` -- sum is
    frequency x magnitude, so the two histograms are the same measurement with and without the
    magnitude factor, and `DSV41_PRUNE_SOURCE` switches between them without changing the rules
    that rank them. An expert fired once at enormous magnitude is a mean-ranking's top expert and
    is worth almost nothing to a cache policy.

    Why this matters: REAP benchmarked Kimi-K2 -- 384 routed experts, one shared, auxiliary-
    loss-free routing, the same shape as this model -- and found frequency-based pruning collapses
    (LiveCodeBench 0.434 -> 0.082 at 75 % kept, 0.000 at 50 %) where saliency holds (0.440/0.429).
    """
    return np.bincount(idx.reshape(-1), weights=sal.reshape(-1), minlength=N_EXP)


def top1_hist(idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """How often each expert was the token's FIRST pick, as a 384-wide histogram.

    The tracer stores `indices` in the router's own selection order -- `(scores +
    gate_bias).topk(k)` in tools/v41_ref.py -- and `weights` as the renormalised gate weights of
    those same picks, gathered in that order. The two orders are NOT the same: the selection bias
    steers which six experts are taken without entering the weight that multiplies their output,
    so on the reference trace only 44 % of tokens have their six weights in descending order,
    and column 0 is the largest weight on 95.2 % of them. First pick therefore means
    `weights.argmax(1)` -- the expert that carried the most of this token through the layer --
    and not column 0, which is first only in the selection.

    Ties break to the lower column, i.e. to the router's own order, which is the sensible
    tiebreak: the weights are stored as fp16 and equal neighbours do happen.

    Rows of this histogram sum to the number of tokens, where `counts` sums to tokens x topk.
    """
    n = idx.shape[0]
    if n == 0:
        return np.zeros(N_EXP, dtype=np.int64)
    return np.bincount(idx[np.arange(n), w.argmax(1)], minlength=N_EXP)


def pair_table(idx: np.ndarray, top: int = 32, min_count: int = 5) -> list:
    """Which experts this layer takes TOGETHER: `[[a, b, count, lift], ...]`, a < b.

    Every token contributes the C(topk, 2) = 15 unordered pairs of its picks. `count` is how
    many tokens took both; `lift` is that count against the count two independently routed
    experts of the same marginal frequencies would produce:

        lift(a, b) = count(a, b) / (count(a) * count(b) / tokens)

    which is exactly the definition the Weight Atlas prints under its co-routing ring. An expert
    is picked at most once per token, so `count(a)` is its routing count and needs no separate
    pass.

    The table is the union of the `top` pairs by count and the `top` pairs by lift, so both of
    the page's two views are served from one list; pairs seen fewer than `min_count` times are
    not eligible for the lift half, because a pair seen twice can carry an enormous lift and
    says nothing. Entries are ordered by count, descending.

    Truncation is the point: the full table is C(384, 2) = 73,536 rows per layer per topic and
    almost all of it is noise at a few thousand tokens a topic. What is kept is ~64 rows.
    """
    n, k = idx.shape
    if n == 0 or k < 2:
        return []
    ii, jj = np.triu_indices(k, 1)
    a, b = idx[:, ii], idx[:, jj]
    key = (np.minimum(a, b) * N_EXP + np.maximum(a, b)).reshape(-1)
    uk, cnt = np.unique(key, return_counts=True)
    ea, eb = uk // N_EXP, uk % N_EXP
    marg = np.bincount(idx.reshape(-1), minlength=N_EXP).astype(np.float64)
    expected = marg[ea] * marg[eb] / n
    lift = np.where(expected > 0, cnt / np.maximum(expected, 1e-12), 0.0)
    keep = set(np.argsort(-cnt)[:top].tolist())
    eligible = np.nonzero(cnt >= min_count)[0]
    if eligible.size:
        keep |= set(eligible[np.argsort(-lift[eligible])[:top]].tolist())
    order = sorted(keep, key=lambda i: (-int(cnt[i]), int(ea[i]), int(eb[i])))
    return [[int(ea[i]), int(eb[i]), int(cnt[i]), float(f"{lift[i]:.4g}")] for i in order]


def coverage_curve(counts: np.ndarray):
    c = np.sort(counts)[::-1]
    return np.cumsum(c) / max(c.sum(), 1)


def lru_sim(layers: dict, budget: int, block: int = 6, order: list[int] | None = None):
    """Token-by-token LRU over (layer, expert) keys with `budget` resident experts.
    Returns per-token hit rate and per-block (union of `block` consecutive tokens) hit rate."""
    from collections import OrderedDict
    Ls = sorted(layers)
    n_tok = layers[Ls[0]]["idx"].shape[0]

    def run(step: int):
        """One LRU simulation where the unit of work is `step` consecutive tokens (1 = per token,
        6 = a DSpark verify block). Hit = expert already resident when the unit needs it."""
        cache: OrderedDict = OrderedDict()
        hits = misses = 0
        for t0 in range(0, n_tok, step):
            t1 = min(n_tok, t0 + step)
            for L in Ls:
                for e in np.unique(layers[L]["idx"][t0:t1]):
                    k = (L, int(e))
                    if k in cache:
                        cache.move_to_end(k)
                        hits += 1
                    else:
                        misses += 1
                        cache[k] = 1
                        if len(cache) > budget:
                            cache.popitem(last=False)
        return hits / (hits + misses)

    return run(1), run(block)


# --- merging several reductions into one stats file -------------------------
# A shipped coverage.json is not the output of one trace. Topics were traced separately and
# their histograms copied in, and at least one topic is the UNION of two traces: the
# `reasoning_code` corpus was collected in two passes (14 records, then 40 more), each traced
# and reduced on its own, and the shipped `counts_reasoning_code` is the two ADDED TOGETHER --
# 118,272 routed slots a layer against 21,258 for the smaller pass alone. Any merge that copies
# one trace's histogram across instead of summing the set silently ships a topic measured on a
# third of its corpus, and a keep-set ranked on it is wrong without ever looking wrong.
#
# So the trace set is not guessed from directory names: for each topic it is DERIVED, by
# finding which subset of the reductions on hand adds up to the shipped histogram exactly.

TOPIC_FAMILIES = ("counts_", "saliency_", "top1_")
PAIR_FAMILY = "pairs_"


def load_reduction(d: str) -> dict:
    """One `--out` directory of this tool: coverage.json, and pairs.json beside it if written."""
    d = os.path.normpath(d)
    if not os.path.exists(os.path.join(d, "coverage.json")):
        raise SystemExit(f"not a reduction directory (no coverage.json): {d}")
    cov = json.load(open(os.path.join(d, "coverage.json")))
    side = os.path.join(d, "pairs.json")
    return {"dir": d, "cov": cov,
            "pairs": json.load(open(side)) if os.path.exists(side) else None}


def topic_rows(cov: dict, key: str):
    """`{layer: row}` for one histogram key, or None unless every layer carries it. A histogram
    that stops half way down the model is not a histogram this file can use."""
    pl = cov.get("per_layer") or {}
    rows = {L: pl[L][key] for L in pl if key in pl[L]}
    return rows if rows and len(rows) == len(pl) else None


def sum_rows(rows_list: list) -> dict:
    out = {}
    for rows in rows_list:
        for L, row in rows.items():
            if L not in out:
                out[L] = list(row)
            else:
                out[L] = [a + b for a, b in zip(out[L], row)]
    return out


def rows_equal(a: dict, b: dict, rtol: float = 1e-12) -> bool:
    """Element-by-element equality.

    `counts_*` and `top1_*` are integers and compare exactly. `saliency_*` is a sum of floats,
    and a sum re-taken in a different association is allowed to differ in its last bits -- but
    only there: `rtol` is 1e-12, twelve orders of magnitude tighter than the ~5x a missing
    trace moves a histogram by, so this cannot pass a merge that dropped one."""
    if set(a) != set(b):
        return False
    for L in a:
        x, y = a[L], b[L]
        if len(x) != len(y):
            return False
        for u, v in zip(x, y):
            if u != v and abs(u - v) > rtol * max(abs(u), abs(v)):
                return False
    return True


def choose_traces(target: dict, cands: list, max_cands: int = 14):
    """Which of `cands` (each a `{layer: row}`) add up to `target`; None when no subset does.

    Totals first -- one number a candidate, so the search over subsets is arithmetic on scalars
    -- and only the subsets whose total lands on the target are then compared element by
    element. The smallest such subset wins, so a reduction that happens to be all zeros for a
    topic does not get counted in.
    """
    n = len(cands)
    if n == 0 or n > max_cands:
        return None
    want = sum(sum(r) for r in target.values())
    totals = [sum(sum(r) for r in rows.values()) for rows in cands]
    best = None
    for mask in range(1, 1 << n):
        picked = [i for i in range(n) if mask >> i & 1]
        if best is not None and len(picked) >= len(best):
            continue
        got = sum(totals[i] for i in picked)
        if got != want and abs(got - want) > 1e-9 * max(1.0, abs(want)):
            continue
        if rows_equal(sum_rows([cands[i] for i in picked]), target):
            best = picked
    return best


def merge_pair_tables(tables: list, counts_row: list, topk: int = 6,
                      top: int = 32, min_count: int = 5) -> list:
    """The co-routing tables of several traces of one topic, added into one.

    Pair counts add the way routing counts do. The lift does not: it is recomputed here from
    the SUMMED counts and the union's own marginals -- `counts_row` is the merged
    `counts_<topic>` of this layer, whose total divides back to the union's tokens -- so the
    lift a merged table reports is the union's lift and not an average of two.

    The truncation each input carries is inherited: a pair that was outside one trace's kept
    rows contributes nothing from that trace, so a merged count can be low for a pair that is
    frequent in one pass and marginal in the other. Same class of approximation as the
    truncation itself, and the alternative is keeping all C(384, 2) = 73,536 rows per topic
    per layer.
    """
    agg: dict = {}
    for tb in tables:
        for a, b, n, _lift in tb or []:
            agg[(int(a), int(b))] = agg.get((int(a), int(b)), 0) + int(n)
    if not agg:
        return []
    tokens = sum(counts_row) / topk
    rows = []
    for (a, b), n in agg.items():
        expected = counts_row[a] * counts_row[b] / tokens if tokens > 0 else 0.0
        rows.append((a, b, n, (n / expected) if expected > 0 else 0.0))
    keep = {(a, b) for a, b, _, _ in sorted(rows, key=lambda r: (-r[2], r[0], r[1]))[:top]}
    eligible = [r for r in rows if r[2] >= min_count]
    keep |= {(a, b) for a, b, _, _ in sorted(eligible, key=lambda r: (-r[3], r[0], r[1]))[:top]}
    out = [r for r in rows if (r[0], r[1]) in keep]
    out.sort(key=lambda r: (-r[2], r[0], r[1]))
    return [[a, b, n, float(f"{lf:.4g}")] for a, b, n, lf in out]


def merge_reductions(shipped_path: str, run_dirs: list, log=print) -> dict:
    """Rebuild a shipped stats file out of fresh reductions, and say whether it came out the same.

    Returns `{"cov", "pairs", "mismatch", "topics", "chosen"}`. `mismatch` is the list of
    histogram keys that did NOT come back identical; the caller must refuse to install the
    result when it is non-empty."""
    shipped = json.load(open(shipped_path))
    ship_pl = shipped["per_layer"]
    topics = sorted({k.split("_", 1)[1] for row in ship_pl.values() for k in row
                     if k.startswith("counts_")})
    runs = [load_reduction(d) for d in run_dirs]
    log(f"  shipped file: {len(topics)} topics, {len(ship_pl)} layers")

    covered = {r["dir"]: len([t for t in topics if topic_rows(r["cov"], "counts_" + t)])
               for r in runs}
    for r in runs:
        log(f"  {os.path.basename(r['dir'])}: carries {covered[r['dir']]} of the shipped topics")
    runs.sort(key=lambda r: -covered[r["dir"]])

    # The base supplies everything that is not per-topic: the per-layer scalars (`used`, `cov`,
    # `entropy_bits`, `block6_unique_mean`), the mixed histograms and the `global` budget
    # ladder. They describe one trace, as they always did -- the largest one.
    merged = json.loads(json.dumps(runs[0]["cov"]))
    for row in merged["per_layer"].values():
        for key in [k for k in row if k.startswith(TOPIC_FAMILIES)]:
            del row[key]
    meta = next((r["pairs"] for r in runs if r["pairs"]), None) or {}
    topk = int(meta.get("topk", 6) or 6)
    ptop, pmin = int(meta.get("top", 32) or 32), int(meta.get("min_count", 5) or 5)
    merged_pairs = {"schema": 1, "min_count": pmin, "top": ptop, "topk": topk,
                    "per_layer": {L: {} for L in merged["per_layer"]}}

    mismatch, chosen = [], {}
    for t in topics:
        target = topic_rows(shipped, "counts_" + t)
        cands = [r for r in runs if topic_rows(r["cov"], "counts_" + t)]
        pick = choose_traces(target, [topic_rows(r["cov"], "counts_" + t) for r in cands])
        if pick is None:
            # Best effort, so the refused .new file still shows what this box does have; the
            # verification below is what records the failure, once, with the numbers.
            pick = list(range(len(cands)))
            chosen[t] = ["<no subset reproduces the shipped histogram>"]
        else:
            chosen[t] = [os.path.basename(cands[i]["dir"]) for i in pick]
        used = [cands[i] for i in pick]
        if len(used) > 1:
            log(f"  {t}: the union of {len(used)} traces -- {', '.join(chosen[t])}")
        for fam in TOPIC_FAMILIES:
            rows = [topic_rows(r["cov"], fam + t) for r in used]
            if not rows or any(x is None for x in rows):
                continue
            for L, row in sum_rows(rows).items():
                merged["per_layer"][L][fam + t] = row
        tables = [(r["pairs"] or {}).get("per_layer", {}) for r in used]
        for L in merged["per_layer"]:
            got = [tb.get(L, {}).get(PAIR_FAMILY + t) for tb in tables]
            got = [g for g in got if g]
            if got:
                merged_pairs["per_layer"][L][PAIR_FAMILY + t] = merge_pair_tables(
                    got, merged["per_layer"][L].get("counts_" + t, []), topk, ptop, pmin)

    # --- the assertion that decides whether this may replace anything -------
    # Key presence is not enough: a merge that dropped a trace carries every key and the wrong
    # numbers in one of them. Every counts_<topic> and saliency_<topic> array in the rebuilt
    # file has to come back element-for-element identical to the shipped one.
    for t in topics:
        for fam in ("counts_", "saliency_"):
            want = topic_rows(shipped, fam + t)
            if want is None:
                continue
            got = topic_rows(merged, fam + t)
            if got is None:
                mismatch.append(fam + t + " (absent)")
            elif not rows_equal(got, want) and fam + t not in mismatch:
                mismatch.append(fam + t)
    for fam in ("counts", "saliency"):
        want, got = topic_rows(shipped, fam), topic_rows(merged, fam)
        if want and got and not rows_equal(got, want):
            # scoped to the base trace and read by nothing -- worth saying, not worth refusing
            log(f"  note: the mixed `{fam}` histogram differs from the shipped one "
                f"(it describes whichever single trace is the base, and always did)")
    have_top1 = {k.split("_", 1)[1] for row in merged["per_layer"].values() for k in row
                 if k.startswith("top1_")}
    log(f"  rebuilt: {len(have_top1 & set(topics))} of {len(topics)} topics carry top1_<topic>")
    return {"cov": merged, "pairs": merged_pairs, "mismatch": sorted(set(mismatch)),
            "topics": topics, "chosen": chosen}


def merge_main(a) -> int:
    """`--merge SHIPPED --runs DIR ...`: rebuild SHIPPED from fresh reductions and install it
    only if every histogram came back identical."""
    r = merge_reductions(a.merge, a.runs)
    stats_dir = os.path.dirname(os.path.abspath(a.merge))
    if r["mismatch"]:
        new = os.path.join(stats_dir, "coverage.new.json")
        json.dump(r["cov"], open(new, "w"))
        json.dump(r["pairs"], open(os.path.join(stats_dir, "pairs.new.json"), "w"))
        print(f"  REFUSED: {len(r['mismatch'])} histograms did not come back identical")
        print("  " + ", ".join(r["mismatch"][:10]) + (" ..." if len(r["mismatch"]) > 10 else ""))
        print(f"  kept {a.merge} untouched and wrote {new} instead")
        print("  the usual cause is a trace this box no longer has: a topic whose shipped "
              "histogram is the sum of two traces needs both of them on --runs")
        return 5
    import shutil
    shutil.copy2(a.merge, a.merge + ".before-extras")
    json.dump(r["cov"], open(a.merge, "w"))
    json.dump(r["pairs"], open(os.path.join(stats_dir, "pairs.json"), "w"))
    print(f"  every counts_<topic> and saliency_<topic> came back identical; replaced {a.merge} "
          f"({os.path.getsize(a.merge) / 1e6:.1f} MB, previous kept as .before-extras)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="a directory of per-layer traces to reduce "
                                    "(required unless --merge)")
    ap.add_argument("--out", help="where the reduction goes (required unless --merge)")
    ap.add_argument("--merge", metavar="SHIPPED",
                    help="merge mode: rebuild the stats file SHIPPED out of the reductions "
                         "named by --runs and replace it ONLY if every counts_<topic> and "
                         "saliency_<topic> comes back element-for-element identical. A topic "
                         "whose shipped histogram is the sum of several traces is summed the "
                         "same way; which traces those are is derived, not assumed")
    ap.add_argument("--runs", nargs="*", default=[], metavar="DIR",
                    help="reduction directories (each holding coverage.json) for --merge")
    ap.add_argument("--budgets", default="1000,1500,2000,3000,4000,5000,6000,8000")
    ap.add_argument("--cov-curves", action="store_true",
                    help="also write the per-category coverage curves; they are derived from the "
                         "histograms, nothing reads them back, and at 35 topics they are 7.7 MB "
                         "of a 9.8 MB file")
    ap.add_argument("--pairs", choices=("sibling", "inline", "off"), default="sibling",
                    help="where the co-routing tables go. `sibling` (default) writes them to "
                         "pairs.json next to coverage.json -- at 39 topics they are ~2 MB and "
                         "only tools/atlas_export.py reads them, where coverage.json is opened "
                         "by the engine on every start. `inline` puts them in coverage.json "
                         "under the same `pairs_<topic>` keys; `off` computes none")
    ap.add_argument("--pairs-top", type=int, default=32, metavar="N",
                    help="pairs kept per layer per topic: the top N by count and the top N by "
                         "lift, unioned (default %(default)s)")
    ap.add_argument("--pair-min-count", type=int, default=5, metavar="N",
                    help="a pair needs this many co-occurrences before its lift is ranked "
                         "(default %(default)s)")
    a = ap.parse_args()
    if a.merge:
        if not a.runs:
            ap.error("--merge needs --runs DIR [DIR ...]")
        return merge_main(a)
    if not (a.trace and a.out):
        ap.error("--trace and --out are required (or --merge SHIPPED --runs DIR ...)")
    os.makedirs(a.out, exist_ok=True)
    layers, meta = load(a.trace)
    Ls = sorted(layers)
    budgets = [int(x) for x in a.budgets.split(",")]
    n_tok = layers[Ls[0]]["idx"].shape[0]
    cats = sorted(set(layers[Ls[0]]["cat"].tolist()))
    curves = a.cov_curves

    per_layer = {}
    pairs_layer: dict = {}
    pairs_on = a.pairs != "off"
    glob_counts = {}
    no_norms = [L for L in Ls if "sal" not in layers[L]]
    if no_norms:
        print(f"note: {len(no_norms)} of {len(Ls)} layers carry no `contrib_norms`/`out_norms` (layers "
              f"{no_norms[0]}-{no_norms[-1]}): they were traced before tools/expert_trace.py "
              f"recorded expert-output norms, so they get no saliency_* histogram and "
              f"DSV41_PRUNE_SOURCE=saliency will refuse this file. Re-trace to use it.")
    for L in Ls:
        idx = layers[L]["idx"]
        counts = np.bincount(idx.reshape(-1), minlength=N_EXP)
        cov = coverage_curve(counts)
        used = int((counts > 0).sum())
        per_layer[L] = {"used": used, "counts": counts, "cov": cov,
                        "top50": float(cov[N_EXP // 2 - 1]), "top25": float(cov[N_EXP // 4 - 1]),
                        "top10": float(cov[int(N_EXP * 0.10) - 1]),
                        "entropy_bits": float(-(counts / counts.sum() * np.log2(np.where(counts > 0, counts / counts.sum(), 1))).sum())}
        # block uniqueness (union over 6 consecutive tokens)
        bl = [len(np.unique(idx[t:t + 6])) for t in range(0, n_tok - 5, 6)]
        per_layer[L]["block6_unique_mean"] = float(np.mean(bl))
        nrm = layers[L].get("sal")
        if layers[L].get("clamped"):
            per_layer[L]["saliency_clamped_picks"] = layers[L]["clamped"]
        if nrm is not None:
            # The mixed histogram, next to `counts`: what the whole corpus's routing contributed.
            per_layer[L]["saliency"] = saliency_hist(idx, nrm)
        # Who LED, next to who was taken. Needs nothing the tracer did not already store: the
        # gate weights are in the trace, and the largest of a token's six names its first pick.
        wgt = layers[L]["w"]
        per_layer[L]["top1"] = top1_hist(idx, wgt)
        if pairs_on:
            pairs_layer[L] = {"pairs": pair_table(idx, a.pairs_top, a.pair_min_count)}
        for c in cats:
            m = layers[L]["cat"] == c
            cat_counts = np.bincount(idx[m].reshape(-1), minlength=N_EXP)
            per_layer[L][f"top1_{c}"] = top1_hist(idx[m], wgt[m])
            if pairs_on:
                pairs_layer[L][f"pairs_{c}"] = pair_table(idx[m], a.pairs_top, a.pair_min_count)
            if nrm is not None:
                # One per topic, next to counts_<topic> and read the same way: the engine's
                # DSV41_PRUNE_SOURCE picks which of the two families ranks the keep-set.
                per_layer[L][f"saliency_{c}"] = saliency_hist(idx[m], nrm[m])
            # the histogram itself, not just its coverage curve: the engine's pruned mode ranks
            # experts per category, and reading it from here means a checkout does not need the
            # raw per-layer trace arrays (tens of MB) to reproduce a keep-set.
            per_layer[L][f"counts_{c}"] = cat_counts
            # The per-category coverage CURVE is derived from that histogram in one pass and
            # nothing reads it back -- but it is 384 full-precision floats per category per
            # layer, which at 35 topics is 7.7 MB of a 9.8 MB file. Off by default; the
            # histograms above are what a keep-set is actually built from.
            if curves:
                per_layer[L][f"cov_{c}"] = coverage_curve(cat_counts)
        for e in range(N_EXP):
            glob_counts[(L, e)] = int(counts[e])

    # global coverage by budget (rank all (layer, expert) by frequency)
    gc = np.array(sorted(glob_counts.values(), reverse=True), dtype=np.float64)
    gcov = np.cumsum(gc) / gc.sum()
    n_keys = len(gc)
    # category overlap: experts in the top-K set of coding vs general
    lines = []
    lines.append(f"# Expert coverage -- {meta['n_tokens']} tokens, {meta['n_seqs']} sequences, layers {Ls[0]}-{Ls[-1]}\n")
    sal_note = ("and `saliency_<topic>` (gate weight x expert-output norm, REAP arXiv 2510.13999); "
                "`DSV41_PRUNE_SOURCE` picks which the engine ranks a keep-set by"
                if not no_norms else
                "only -- this trace carries no `out_norms`, so there is no saliency histogram and "
                "`DSV41_PRUNE_SOURCE=saliency` will refuse this file")
    lines.append(f"Histograms per layer: `counts_<topic>` (routing frequency) {sal_note}.\n")
    lines.append("Also per layer: `top1_<topic>`, how often each expert carried a token's largest gate "
                 "weight (rows sum to tokens, where `counts` sums to tokens x top-k)"
                 + (", and a co-routing table `pairs_<topic>` of `[a, b, count, lift]`"
                    + (" in `pairs.json` beside this file" if a.pairs == "sibling" else " in `coverage.json`")
                    if pairs_on else "") + ".\n")
    lines.append("## Per layer\n")
    lines.append("| layer | experts used | top-10% covers | top-25% covers | top-50% covers | entropy (bits, max 8.58) | unique experts / 6-token block (max 36) |")
    lines.append("|---|---|---|---|---|---|---|")
    for L in Ls:
        p = per_layer[L]
        lines.append(f"| {L} | {p['used']} | {p['top10']:.3f} | {p['top25']:.3f} | {p['top50']:.3f} | {p['entropy_bits']:.2f} | {p['block6_unique_mean']:.1f} |")
    lines.append("")
    lines.append(f"## Global coverage vs resident-expert budget (layers traced: {len(Ls)} of 40, {n_keys} (layer,expert) keys)\n")
    lines.append("Static resident set = the most frequent (layer, expert) pairs overall. 'covers' = share of routed slots that hit the resident set. Memory = FP4 experts only (18.8 MB each).\n")
    lines.append("| budget (experts) | share of all keys | resident GB (FP4) | static coverage | LRU hit/token | LRU hit/6-token block |")
    lines.append("|---|---|---|---|---|---|")
    res = {"per_layer": {}, "global": []}
    for b in budgets:
        b_eff = min(b, n_keys)
        cov = float(gcov[b_eff - 1])
        hr_tok, hr_blk = lru_sim(layers, b_eff)
        gb = b * EXPERT_BYTES_FP4 / 1e9
        lines.append(f"| {b} | {b_eff / n_keys:.2f} | {gb:.1f} | {cov:.3f} | {hr_tok:.3f} | {hr_blk:.3f} |")
        res["global"].append({"budget": b, "static_coverage": cov, "lru_hit_token": hr_tok, "lru_hit_block6": hr_blk, "resident_gb_fp4": gb})
    lines.append("")
    if len(cats) == 2:
        lines.append("## Coding vs general: overlap of the per-layer top-25% sets\n")
        lines.append("| layer | Jaccard(top25 coding, top25 general) | coding slots covered by general's top25 |")
        lines.append("|---|---|---|")
        for L in Ls:
            idx, cat = layers[L]["idx"], layers[L]["cat"]
            sets = {}
            for c in cats:
                cnt = np.bincount(idx[cat == c].reshape(-1), minlength=N_EXP)
                sets[c] = set(np.argsort(cnt)[::-1][: N_EXP // 4].tolist())
            j = len(sets[cats[0]] & sets[cats[1]]) / len(sets[cats[0]] | sets[cats[1]])
            cod = idx[cat == "coding"].reshape(-1)
            cross = float(np.isin(cod, list(sets["general"])).mean()) if cod.size else float("nan")
            lines.append(f"| {L} | {j:.2f} | {cross:.3f} |")
        lines.append("")
    for L in Ls:
        res["per_layer"][L] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in per_layer[L].items()
                               if k not in ("counts",)}
        res["per_layer"][L]["counts"] = per_layer[L]["counts"].tolist()

    # The co-routing tables. They are the only thing written here that nothing in the engine
    # reads -- tools/atlas_export.py draws them and that is all -- and at 39 topics they are
    # ~2 MB against coverage.json's 13 MB, which the engine parses on every start. So they go
    # beside it by default rather than into it, and the exporter looks in both places.
    pairs_meta = {"schema": 1, "min_count": a.pair_min_count, "top": a.pairs_top,
                  "topk": int(layers[Ls[0]]["idx"].shape[1]) if Ls else 0}
    if pairs_on and a.pairs == "inline":
        for L in Ls:
            res["per_layer"][L].update(pairs_layer[L])
        res["pairs_meta"] = pairs_meta
    json.dump(res, open(os.path.join(a.out, "coverage.json"), "w"))
    if pairs_on and a.pairs == "sibling":
        json.dump({**pairs_meta, "per_layer": {str(L): pairs_layer[L] for L in Ls}},
                  open(os.path.join(a.out, "pairs.json"), "w"))
    open(os.path.join(a.out, "coverage.md"), "w").write("\n".join(lines))
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(1, N_EXP + 1) / N_EXP * 100
        for L in Ls:
            ax[0].plot(x, per_layer[L]["cov"], lw=0.8, alpha=0.7, label=f"L{L}" if len(Ls) <= 8 else None)
        ax[0].set_xlabel("top N% of experts in the layer"); ax[0].set_ylabel("share of routed slots covered")
        ax[0].set_title("per-layer cumulative coverage"); ax[0].grid(alpha=0.3)
        if len(Ls) <= 8: ax[0].legend()
        ax[1].plot(np.arange(1, n_keys + 1), gcov)
        for b in budgets:
            if b <= n_keys: ax[1].axvline(b, color="orange", lw=0.6, alpha=0.6)
        ax[1].set_xlabel("resident (layer, expert) budget"); ax[1].set_ylabel("static coverage")
        ax[1].set_title(f"global coverage, {len(Ls)} layers traced"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "coverage.png"), dpi=130)
        fig, ax = plt.subplots(figsize=(13, 4))
        for L in Ls:
            ax.plot(np.sort(per_layer[L]["counts"])[::-1] / per_layer[L]["counts"].sum(), lw=0.8, alpha=0.7)
        ax.set_yscale("log"); ax.set_xlabel("expert rank within layer"); ax.set_ylabel("share of slots")
        ax.set_title("expert usage, sorted, per layer (log)"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "layer_hist.png"), dpi=130)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
