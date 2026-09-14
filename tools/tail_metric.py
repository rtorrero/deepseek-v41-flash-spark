#!/usr/bin/env python3
"""
tail_metric.py -- does the coverage bar predict the generation gate, and what does?

The bar on `./tune.sh` is COVERAGE: the fraction of a topic's measured routing that
lands on a resident expert. Under the pair the box is served with -- `saliency`
ranked by `maxmin` -- it stopped being able to separate anything. Saliency mass is
top-heavy; half of a layer's sits on one expert, so a keep-set swap of 513 of 5,560
resident experts moved every bar by about 0.005 and flipped the gate from 6 of 10 to
3 of 10 (docs/keep-sets.md, 2026-09-14; RESULTS.md, 2026-09-14 03:40). Ten shipped
profiles read 0.94 to 0.98 while their gates run from 3 of 10 to 8 of 11.

What a generation trips over is not a topic's mass. `DSV41_PRUNE_MODE=substitute`
masks the router's logits to the resident experts and takes the top-6 of what is
left, so a token whose six picks in a layer are NOT all resident is computed with
experts it did not ask for, at full renormalised weight, and the error feeds the next
layer's mixing coefficients as well. That is a property of the TAIL of a token's
routing, and a mean over a corpus is the wrong instrument for it.

So this tool computes, for every shipped profile, at the keep fraction its gate
record was actually run at:

  cov (saliency)  the bar as it ships -- the number this is measured against
  cov (picks)     the same keep-set measured in PICKS instead of magnitude
  nonres          expected non-resident picks per token per layer, 0 to 6. Exact.
  all6            the rate at which all six picks of a token-layer are resident,
                  estimated under independence (biased low; see budget.tail_curves)
  worst layer     the worst single layer's resident pick fraction

and ranks them against what the gate actually did (strict passes, finished answers,
hard misses), by Spearman and by the plain numbers.

  python3 tools/tail_metric.py
  python3 tools/tail_metric.py --profile frontend
  python3 tools/tail_metric.py --json results/keepsets/tail_metric.json

No torch, no model load, no network. Reads `results/keepsets/topics/coverage.json`,
`results/keepsets/gates.json` and the `GATE.md` files those name.

The record-selection rule and the gate-card parser below are a deliberate second
implementation of the ones in `tools/tune.py`, not an import of them: that module
pulls `curses` at import time, and a metrics CLI has no business needing a terminal.
`tools/test_tail_metric.py` holds the two to the same record for every shipped
profile, so they cannot drift apart quietly.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_DEFAULT = os.path.join(ROOT, "results", "keepsets", "topics", "coverage.json")
GATES_DEFAULT = os.path.join(ROOT, "results", "keepsets", "gates.json")
TUNE_PY = os.path.join(ROOT, "tools", "tune.py")
GATE_HEAD = "# Generation gate — "


# --- what ships -------------------------------------------------------------

def read_profiles(path: str = TUNE_PY) -> list:
    """`PROFILES` out of tools/tune.py, by parsing it rather than importing it.

    The same trick tools/test_budget_rank.py plays on the engine: the list is a
    literal, so `ast` can lift it whole and nothing in that module runs. An
    import would pull `curses`, argparse, a config search and an atlas exporter
    to read ten tuples.

    Returns (name, blurb, topics, record, rank) with rank defaulting to the
    module's own default where a profile does not name one.
    """
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "PROFILES" for t in node.targets):
            continue
        out = []
        for row in ast.literal_eval(node.value):
            name, blurb, topics, record = row[0], row[1], list(row[2]), row[3]
            rank = row[4] if len(row) > 4 and row[4] else B.RANK_DEFAULT
            out.append((name, blurb, topics, record, rank))
        return out
    raise ValueError(f"no PROFILES assignment in {path}")


# --- the gate cards ---------------------------------------------------------
# A second reading of tools/tune.py's `_card` / `_counts` / `read_gate` /
# `gate_for`, with one addition: the `misses by kind` tail of the second verdict
# sentence, which the screen does not need and a correlation does.

def _card(lines: list, key: str) -> str | None:
    want = f"| {key} |"
    for ln in lines:
        if ln.startswith(want):
            return ln.split("|")[2].strip()
    return None


def _counts(lines: list) -> tuple:
    """(runs, strict, finished, kinds) out of the two verdict sentences.

    `strict` is the gate -- the thing the prompt asked for is in the output, the
    think block included. `finished` is the strict passes plus the runs whose
    only fault was a repeated window inside the deliberation. The difference
    between them is the HARD misses: think-exit, guard, corrupt, content --
    every shape that is not "wrote the right answer, redrafted a line on the way
    there". `kinds` breaks those out where the card carries them; cards written
    before 2026-09-13 do not, which is why `hard` is taken as runs - finished
    and not as a sum of the four.
    """
    runs = strict = finished = None
    kinds = None
    for ln in lines:
        if ln.startswith("**Verdict: PASS**"):
            m = re.search(r"all (\d+) runs", ln)
            if m:
                runs = strict = int(m.group(1))
        elif ln.startswith("**Verdict: FAIL**"):
            m = re.search(r"(\d+) of (\d+) runs failed", ln)
            if m:
                runs, strict = int(m.group(2)), int(m.group(2)) - int(m.group(1))
        else:
            m = re.match(r"(\d+) of (\d+) finished a correct answer", ln)
            if m:
                finished, runs = int(m.group(1)), runs or int(m.group(2))
                k = re.search(r"misses by kind: ([^.]*)", ln)
                if k:
                    kinds = {}
                    for part in k.group(1).split(","):
                        nm, _, v = part.strip().rpartition(" ")
                        if nm and v.isdigit():
                            kinds[nm] = int(v)
    return runs, strict, finished, kinds


def read_gate(path: str) -> list:
    """Every gate section in one GATE.md, oldest first. A file that is not a gate
    log at all yields nothing, which is the same answer as no file."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return []
    out = []
    for chunk in text.split(GATE_HEAD)[1:]:
        lines = chunk.splitlines()
        topics = [t.strip() for t in (_card(lines, "topics") or "").split(",") if t.strip()]
        runs, strict, finished, kinds = _counts(lines)
        if not topics or strict is None:
            continue
        out.append({"run": lines[0].strip(), "topics": topics, "runs": runs,
                    "strict": strict, "finished": finished, "kinds": kinds,
                    # A `| only |` row means named prompts were re-run: it says
                    # those pass and nothing about the ones that were not run,
                    # so it is never a profile's record.
                    "filtered": _card(lines, "only") is not None})
    return out


def gate_for(record: str | None, topics, root: str = ROOT) -> dict | None:
    """The newest full run in this profile's GATE.md whose topic list is EXACTLY
    the profile's. Exactly, because a run on a different bundle is a different
    measurement: `reasoning_lang` moved European languages from 6 of 10 to 3 of
    10 and World languages from 3 of 10 to 6 of 10 on the same keep-set and the
    same night."""
    if not record:
        return None
    path = os.path.join(root, "results", "keepsets", record, "GATE.md")
    want = sorted(topics)
    runs = [g for g in read_gate(path) if not g["filtered"] and sorted(g["topics"]) == want]
    if not runs:
        return None
    g = dict(runs[-1])
    g["record"] = os.path.relpath(path, root)
    return g


def read_gates_index(path: str = GATES_DEFAULT) -> dict:
    """{(record, run): {keep, rank, source}} out of results/keepsets/gates.json --
    the one thing a gate card written before 2026-09-14 does not carry."""
    try:
        raw = json.load(open(path))
    except (OSError, ValueError):
        return {}
    out = {}
    for e in raw.get("runs", []) if isinstance(raw, dict) else []:
        if isinstance(e, dict) and e.get("record") and e.get("run"):
            out[(e["record"], e["run"])] = {k: e.get(k) for k in ("keep", "rank", "source",
                                                                  "measured_in")}
    return out


# --- statistics -------------------------------------------------------------

def _ranks(xs: list) -> list:
    """Competition-free average ranks, so ties do not invent an ordering."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def permutation_p(xs: list, ys: list, trials: int = 20000, seed: int = 20260914) -> float | None:
    """Two-sided p for a Spearman rho, by shuffling one side.

    Not a table lookup and not scipy: n is ten, where the asymptotic t is a poor
    approximation and the repository has no scientific stack. A seeded shuffle
    gives the same answer on every box, which a number that goes into a document
    has to. At n = 10 nothing below |rho| ~ 0.65 will clear 0.05, and saying so
    is the point of computing it.
    """
    import random
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    if rho is None:
        return None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        rng.shuffle(b)
        r = spearman(a, b)
        if r is not None and abs(r) >= abs(rho) - 1e-12:
            hits += 1
    return (hits + 1) / (trials + 1)


def spearman(xs: list, ys: list) -> float | None:
    """Rank correlation. Pearson on the average ranks, so ties are handled -- the
    6/n formula is wrong the moment two profiles gate identically, and at n = 10
    two of them do."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rx = _ranks([p[0] for p in pairs])
    ry = _ranks([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


# --- the per-profile numbers ------------------------------------------------

# Each predictor, and how a profile's topics are reduced to one number. A request
# spans the whole bundle and degenerates at whichever topic the keep-set serves
# least, so every reduction is the WORST topic -- max where more is worse.
PREDICTORS = (
    ("cov_sal_worst", "coverage, ranking family, worst topic", "min"),
    ("cov_sal_mean", "coverage, ranking family, mean", "mean"),
    ("cov_picks_worst", "coverage in picks, worst topic", "min"),
    ("nonres_worst", "non-resident picks/token/layer, worst topic", "max"),
    ("nonres_mean", "non-resident picks/token/layer, mean", "mean"),
    ("all6_worst", "all-six-resident rate, worst topic", "min"),
    ("all6_mean", "all-six-resident rate, mean", "mean"),
    ("worst_layer", "worst layer's resident pick fraction", "min"),
)

# Which way a predictor should point if it predicts the gate at all: +1 means a
# larger value should come with more passes.
SIGN = {"cov_sal_worst": +1, "cov_sal_mean": +1, "cov_picks_worst": +1,
        "nonres_worst": -1, "nonres_mean": -1, "all6_worst": +1, "all6_mean": +1,
        "worst_layer": +1}

GATE_AXES = (("strict_rate", "strict passes / total"),
             ("finished_rate", "finished / total"),
             ("hard_misses", "hard misses (runs - finished)"))

# The cross-profile correlation above has a confound it cannot remove: every
# profile is gated on its OWN prompt suite, so comparing Medicine's 6 of 7 with
# Backend's 3 of 10 compares two different exams. The measurement that does not
# have it is the same profile gated twice on the same prompts with the keep-set
# changed underneath it. Four of those exist, and they are the runs that
# discredited the coverage bar in the first place.
#
# `a` is the earlier configuration, `b` the later. The keep fraction, the rank
# and the source of each are in the RESULTS.md addendum named in `cite` --
# neither run's card carries them, which is what results/keepsets/gates.json
# exists to fix for the records and what this table does for their siblings.
SWAPS = (
    ("european_languages", "European languages",
     ("2026-09-13 22:05", 0.40, "maxmin", "saliency"),
     ("2026-09-14 01:54", 0.36, "maxmin", "saliency"),
     "RESULTS.md, 2026-09-14 02:55 addendum to §5"),
    ("world_languages", "World languages",
     ("2026-09-13 22:33", 0.40, "maxmin", "saliency"),
     ("2026-09-14 02:29", 0.36, "maxmin", "saliency"),
     "RESULTS.md, 2026-09-14 03:40 addendum to §5"),
    ("backend", "Backend",
     ("2026-09-13 18:42", 0.40, "maxmin", "saliency"),
     ("2026-09-14 03:36", 0.36, "maxmin", "saliency"),
     "RESULTS.md, 2026-09-14 06:20 addendum to §5"),
    ("data_and_research", "Data and research",
     ("2026-09-13 20:26", 0.40, "maxmin", "saliency"),
     ("2026-09-14 03:56", 0.36, "maxmin", "saliency"),
     "RESULTS.md, 2026-09-14 06:20 addendum to §5"),
)


class Indexes:
    """One TopicIndex per histogram family, built on demand. Loading the shipped
    catalogue is a fifth of a second and there are two families; a run of this
    tool touches every profile, so it is loaded once and not eleven times."""

    def __init__(self, stats: str):
        self.stats = stats
        self._by_source: dict = {}

    def __call__(self, source: str) -> B.TopicIndex:
        ix = self._by_source.get(source)
        if ix is None:
            ix = self._by_source[source] = B.TopicIndex(self.stats, source=source)
        return ix


def measure(ixs: Indexes, topics, keep: float, rank: str, source: str) -> dict | None:
    """Every predictor for one keep-set: this bundle of topics, this keep
    fraction, ranked this way on this histogram family. `None` when the stats
    file does not carry every topic of the bundle -- a coverage number over part
    of a bundle is not the bundle's, and `maxmin` in particular allocates the
    whole selection at once, so dropping a topic changes every other topic's
    keep-set rather than only its own row."""
    ix = ixs(source)
    sel = tuple(t for t in topics if t in ix.counts)
    if len(sel) != len(topics) or not sel:
        return None
    only = tuple(sorted(sel))
    cov = ix.coverage(sel, keep, rank=rank)
    tail = ix.tail(sel, keep, rank=rank, only=only)
    cs = [cov[t] for t in sel if t in cov]
    cp = [tail[t]["cov_picks"] for t in sel if t in tail]
    nr = [tail[t]["nonres"] for t in sel if t in tail]
    a6 = [tail[t]["all6"] for t in sel if t in tail]
    wl = [tail[t]["worst_layer"] for t in sel if t in tail]
    if not (cs and cp):
        return None
    return {"cov_sal_worst": min(cs), "cov_sal_mean": sum(cs) / len(cs),
            "cov_picks_worst": min(cp),
            "nonres_worst": max(nr), "nonres_mean": sum(nr) / len(nr),
            "all6_worst": min(a6), "all6_mean": sum(a6) / len(a6),
            "worst_layer": min(wl)}


def profile_rows(stats: str = STATS_DEFAULT, gates: str = GATES_DEFAULT,
                 tune_py: str = TUNE_PY, root: str = ROOT, want: str | None = None,
                 ixs: Indexes | None = None) -> list:
    """One row per shipped profile that has both a gate record and the topics its
    keep-set is built from. Every number is measured at the keep fraction, the
    ranking rule and the histogram family that profile's gate was RUN at -- a
    metric computed on a different keep-set is a metric for a different run."""
    index = read_gates_index(gates)
    ixs = ixs or Indexes(stats)
    rows = []
    for name, _blurb, topics, record, rank in read_profiles(tune_py):
        if want and want not in (record, name.lower()):
            continue
        g = gate_for(record, topics, root=root)
        if not g:
            continue
        cfg = index.get((record, g["run"]), {})
        keep, source = cfg.get("keep"), cfg.get("source") or B.SOURCE_DEFAULT
        rank = cfg.get("rank") or rank
        if keep is None:
            continue
        m = measure(ixs, topics, keep, rank, source)
        if m is None:
            continue
        runs, strict, finished = g["runs"], g["strict"], g["finished"]
        rows.append({
            "profile": name, "record": record, "run": g["run"], "gate_file": g["record"],
            "measured_in": cfg.get("measured_in"),
            "keep": keep, "n_keep": B.keep_n(keep), "rank": rank, "source": source,
            "topics": list(topics),
            "runs": runs, "strict": strict, "finished": finished, "kinds": g["kinds"],
            "strict_rate": strict / runs if runs else None,
            "finished_rate": finished / runs if (runs and finished is not None) else None,
            "hard_misses": (runs - finished) if (runs and finished is not None) else None,
            **m,
        })
    return rows


def swap_rows(stats: str = STATS_DEFAULT, root: str = ROOT, want: str | None = None,
              ixs: Indexes | None = None) -> list:
    """The four same-profile A/B swaps in SWAPS, each measured on both of its
    keep-sets. Same prompts, same server, same night; only the keep-set moved.
    This is where a predictor either moves with the gate or does not."""
    ixs = ixs or Indexes(stats)
    out = []
    for record, name, a, b, cite in SWAPS:
        if want and want not in (record, name.lower()):
            continue
        runs = {g["run"]: g for g in read_gate(os.path.join(root, "results", "keepsets",
                                                            record, "GATE.md"))}
        side = {}
        for tag, (run, keep, rank, source) in (("a", a), ("b", b)):
            g = runs.get(run)
            if not g:
                break
            m = measure(ixs, g["topics"], keep, rank, source)
            if m is None:
                break
            side[tag] = {"run": run, "keep": keep, "rank": rank, "source": source,
                         "topics": sorted(g["topics"]), "runs": g["runs"],
                         "strict": g["strict"], "finished": g["finished"],
                         "strict_rate": g["strict"] / g["runs"] if g["runs"] else None, **m}
        if len(side) != 2:
            continue
        out.append({"profile": name, "record": record, "cite": cite, "a": side["a"],
                    "b": side["b"],
                    "delta": {k: side["b"][k] - side["a"][k]
                              for k, _l, _r in PREDICTORS} |
                             {"strict_rate": side["b"]["strict_rate"] - side["a"]["strict_rate"]}})
    return out


def correlations(rows: list) -> dict:
    """Spearman of every predictor against every gate axis, over the profiles."""
    out = {}
    for key, _label, _red in PREDICTORS:
        out[key] = {axis: {"rho": spearman([r[key] for r in rows], [r[axis] for r in rows]),
                           "p": permutation_p([r[key] for r in rows], [r[axis] for r in rows])}
                    for axis, _ in GATE_AXES}
    # The concrete thing a predictor buys, before any question of whether it
    # predicts: how far apart it puts the ten shipped keep-sets. As a RATIO,
    # because these columns are not in the same units -- `nonres` runs 0 to 6
    # and a coverage runs 0 to 1, so their differences cannot be read side by
    # side and their ratios can. A bar that spans 1.03x cannot show a user that
    # anything changed; one that spans 2.6x can.
    out["_spread"] = {key: (max(r[key] for r in rows) / min(r[key] for r in rows)
                            if min(r[key] for r in rows) > 0 else float("inf"))
                      for key, _l, _r in PREDICTORS}
    # How many profiles each axis was actually computed over. Only the four
    # cards written on 2026-09-14 carry the second verdict sentence, so
    # `finished` and `hard misses` are a correlation over four points and must
    # never be read next to `strict` as though both were over ten.
    out["_n"] = {axis: sum(1 for r in rows if r[axis] is not None) for axis, _ in GATE_AXES}
    return out


# --- the exact metric, where the per-token arrays exist ----------------------
# `tail_curves` reads histograms, because histograms are what a `coverage.json`
# ships. The definition it approximates is per token, and it is worth having the
# real one in the same file as the estimate of it: it is what makes the bias
# quoted in budget.tail_curves a measurement rather than a claim.
#
# The shipped topic catalogue (results/keepsets/topics/coverage.json) carries
# `counts_<topic>` and `saliency_<topic>` and NO per-token arrays -- the trace
# they came from is not in the checkout. What is in the checkout is
# results/trace-full-20260910/, a 40-layer, 10,760-token trace over two
# categories, from before the saliency tracer. That is enough to hold the
# estimate to the truth, which is all it is used for.

def exact_from_trace(trace_dir: str, keep_sets: dict, category: str | None = None) -> dict:
    """The metric computed the way it is defined, from `indices` [tokens, 6].

    `keep_sets` is {layer: iterable of resident expert ids}. Returns the exact
    `nonres` and `all6` over the token-layer pairs of `category` (all of them
    when it is None), plus `clean_token`: the share of tokens whose picks are
    resident in EVERY traced layer at once.

    That last one is in here to be reported, not to be used. Over forty layers
    it is a product of forty terms and it is zero to five places at every keep
    fraction this repository has served at -- which is the answer to "what
    fraction of tokens is affected": all of them. The quantity with any
    resolution is the token-LAYER, and that is what `all6` counts.
    """
    import glob
    import numpy as np

    paths = sorted(glob.glob(os.path.join(trace_dir, "trace", "layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)", p).group(1)))
    if not paths:
        raise FileNotFoundError(f"no trace/layer*.npz under {trace_dir}")
    resident = None
    n_layers = 0
    for p in paths:
        L = int(re.search(r"layer(\d+)", p).group(1))
        if L not in keep_sets:
            continue
        z = np.load(p)
        idx = z["indices"].astype(np.int64)
        if category is not None:
            idx = idx[z["category"] == category]
        if not idx.size:
            continue
        mask = np.zeros(B.N_EXPERTS, dtype=bool)
        mask[np.asarray(list(keep_sets[L]), dtype=np.int64)] = True
        col = mask[idx].sum(1).astype(np.int16)        # resident picks of each token here
        resident = col[:, None] if resident is None else np.concatenate(
            [resident, col[:, None]], axis=1)
        n_layers += 1
    if resident is None:
        raise ValueError("no traced layer matched the keep-set")
    return {"nonres": float((B.TOPK - resident).mean()),
            "all6": float((resident == B.TOPK).mean()),
            "clean_token": float((resident == B.TOPK).all(1).mean()),
            "tokens": int(resident.shape[0]), "layers": n_layers}


# --- printing ---------------------------------------------------------------

def _f(v, w=6, p=3):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"


def _rule(head: str) -> str:
    """A markdown separator whose cells are as wide as the header's, so the
    table is readable in a terminal as well as rendered."""
    return "|" + "|".join("-" * len(c) for c in head.split("|")[1:-1]) + "|"


def print_table(rows: list, out=sys.stdout) -> None:
    print("\n# Per profile, each at the keep fraction its gate record was run at\n", file=out)
    head = (f"| {'profile':<20} | {'keep':>4} | {'strict':>7} | {'fin':>6} | "
            f"{'hard':>4} | {'cov(rank)':>9} | {'cov(pick)':>9} | {'nonres':>6} | "
            f"{'all6':>6} | {'worstL':>6} |")
    print(head, file=out)
    print(_rule(head), file=out)
    for r in rows:
        strict = f"{r['strict']}/{r['runs']}"
        fin = f"{r['finished']}/{r['runs']}" if r["finished"] is not None else "--"
        hard = str(r["hard_misses"]) if r["hard_misses"] is not None else "--"
        print(f"| {r['profile']:<20} | {r['keep']:>4.2f} | {strict:>7} | "
              f"{fin:>6} | {hard:>4} | {_f(r['cov_sal_worst'],9,4)} | "
              f"{_f(r['cov_picks_worst'],9,4)} | {_f(r['nonres_worst'])} | "
              f"{_f(r['all6_worst'],6,4)} | {_f(r['worst_layer'],6,4)} |", file=out)
    src = sorted({r["source"] for r in rows})
    print(f"\ncov(rank) = coverage on the family the keep-set was ranked by ({', '.join(src)}) --\n"
          "the bar as it ships. cov(pick)/nonres/all6/worstL are the same keep-set measured in\n"
          "PICKS. Every column is the profile's WORST topic: a request spans the bundle and comes\n"
          "apart at the weakest of it. `hard` = runs - finished, the misses that are not a redraft.",
          file=out)


def print_correlations(rows: list, out=sys.stdout) -> None:
    cor = correlations(rows)
    n, spread = cor["_n"], cor["_spread"]
    print(f"\n# Spearman against the gate, across {len(rows)} profiles"
          "  (rho, p from a seeded permutation test)\n", file=out)
    head = (f"| {'predictor':<44} | {'want':>4} | {'max/min':>7} | " +
            " | ".join(f"{lbl + f' (n={n[k]})':>29}" for k, lbl in GATE_AXES) + " |")
    print(head, file=out)
    print(_rule(head), file=out)
    for key, label, _red in PREDICTORS:
        cells = " | ".join(
            f"{_f(cor[key][axis]['rho'], 6, 3)}  p={_f(cor[key][axis]['p'], 5, 3)}".rjust(29)
            for axis, _ in GATE_AXES)
        print(f"| {label:<44} | {SIGN[key]:>+4d} | {spread[key]:>6.2f}x | {cells} |", file=out)
    print("\n`want` is the sign a predictor should carry if it predicts at all: a keep-set that\n"
          "keeps more of a token's picks should pass more runs and miss harder less often.\n"
          "`max/min` is the ratio of the largest to the smallest value over the ten shipped\n"
          "keep-sets -- what the predictor can show a user at all, before the question of\n"
          "whether it predicts anything. Units differ between rows, ratios do not.\n"
          "Read the last two columns with their n: only the four cards written on 2026-09-14\n"
          "carry the second verdict sentence, so those are four points and prove nothing.\n"
          "And read the first column knowing its confound -- every profile is gated on its OWN\n"
          "prompt suite, so this compares ten different exams. The swaps below hold the exam\n"
          "fixed and move only the keep-set.", file=out)


def print_swaps(swaps: list, out=sys.stdout) -> None:
    print("\n# The same profile, the same prompts, a different keep-set\n", file=out)
    head = (f"| {'profile':<20} | {'keep':>11} | {'strict':>13} | {'d cov(rank)':>11} | "
            f"{'d cov(pick)':>11} | {'d nonres':>8} | {'d all6':>8} | {'d worstL':>8} |")
    print(head, file=out)
    print(_rule(head), file=out)
    for s in swaps:
        a, b, d = s["a"], s["b"], s["delta"]
        moved = f"{a['strict']}/{a['runs']} -> {b['strict']}/{b['runs']}"
        print(f"| {s['profile']:<20} | {a['keep']:.2f}->{b['keep']:.2f} | {moved:>13} | "
              f"{d['cov_sal_worst']:>+11.4f} | {d['cov_picks_worst']:>+11.4f} | "
              f"{d['nonres_worst']:>+8.3f} | {d['all6_worst']:>+8.4f} | "
              f"{d['worst_layer']:>+8.4f} |", file=out)
    agree = {k: sum(1 for s in swaps
                    if s["delta"][k] * SIGN[k] * s["delta"]["strict_rate"] > 0)
             for k, _l, _r in PREDICTORS}
    flat = {k: sum(1 for s in swaps if abs(s["delta"][k]) < 1e-9) for k, _l, _r in PREDICTORS}
    print(f"\nDirection agreed with the gate, out of {len(swaps)} swaps (ties count as no):",
          file=out)
    for key, label, _red in PREDICTORS:
        print(f"  {label:<44} {agree[key]}/{len(swaps)}"
              + (f"  ({flat[key]} flat)" if flat[key] else ""), file=out)
    for s in swaps:
        a, b = s["a"], s["b"]
        diff = sorted(set(b["topics"]) ^ set(a["topics"]))
        print(f"\n{s['profile']}: {a['run']} keep {a['keep']} -> {b['run']} keep {b['keep']}"
              + (f", bundle {'+' if set(b['topics']) > set(a['topics']) else '-'}"
                 f"{','.join(diff)}" if diff else ", same bundle")
              + f"\n  configuration: {s['cite']}", file=out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", default=STATS_DEFAULT,
                    help="the coverage.json the keep-sets are built from")
    ap.add_argument("--gates", default=GATES_DEFAULT,
                    help="results/keepsets/gates.json -- which run is each profile's record")
    ap.add_argument("--tune", default=TUNE_PY, help="the module PROFILES is read out of")
    ap.add_argument("--profile", default=None,
                    help="one profile, by record name (e.g. frontend); default is all of them")
    ap.add_argument("--json", default=None, help="also write the rows and correlations here")
    a = ap.parse_args()

    for p in (a.stats, a.gates, a.tune):
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr)
            return 2
    ixs = Indexes(a.stats)
    rows = profile_rows(a.stats, a.gates, a.tune, want=a.profile, ixs=ixs)
    if not rows:
        print("no profile has both a gate record and every one of its topics in this stats file.",
              file=sys.stderr)
        return 1
    print_table(rows)
    if len(rows) >= 3:
        print_correlations(rows)
    swaps = swap_rows(a.stats, want=a.profile, ixs=ixs)
    if swaps:
        print_swaps(swaps)
    if a.json:
        payload = {"stats": os.path.relpath(a.stats, ROOT),
                   "gates": os.path.relpath(a.gates, ROOT),
                   "profiles": rows,
                   "swaps": swaps,
                   "spearman": correlations(rows) if len(rows) >= 3 else {}}
        with open(a.json, "w") as f:
            json.dump(payload, f, indent=1)
            f.write("\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
