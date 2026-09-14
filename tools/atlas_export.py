#!/usr/bin/env python3
"""atlas_export.py -- this box's expert routing, in the files Weight Atlas reads.

Weight Atlas (`tools/atlas/`, MIT, github.com/alesha-pro/atlas) draws a MoE model's expert
field as one grid of layers x experts, with a domain slice, a ranked ordering and set
outlines over it. A keep-set is exactly that shape -- per layer, which of the 384 routed
experts stay resident -- and it is far easier to argue about as a visible band of columns
than as a list of expert ids. This writes our measurements into the three files that page
loads. `a` in `./tune.sh` runs it and serves the result on loopback.

Reads, out of the checkout and nothing else:

  results/keepsets/topics/coverage.json   per-layer counts_<topic> / saliency_<topic> /
                                          top1_<topic> histograms, 40 layers x 384 routed
                                          experts
  results/keepsets/topics/pairs.json      the co-routing tables beside it, when the trace kept
                                          them (tools/expert_stats.py --pairs); optional, and
                                          the two fields they fill are simply left out when it
                                          is not there
  results/keepsets/gates.json             the generation-gate record per shipped profile:
                                          which keep fraction, ranking rule and histogram
                                          family it was actually measured at
  tools/budget.py                         TopicIndex + keep_n, imported as a library, so the
                                          keep-sets drawn here are the engine's own sets and
                                          not a second implementation of them
  tools/tune.py                           PROFILES: which topics each shipped profile names

Writes, under `tools/atlas/models/` (gitignored -- it is generated, ~5.6 MB):

  <slug>/insights.json    the evidence file the expert-atlas card reads. Only the `routing`
                          block and the parts of `meta` that card uses are filled; the other
                          eleven cards of that region have no evidence here and are skipped
                          by the page.
  <slug>/atlas.jsonl      an architecture-derived weight inventory with every measured
                          statistic left at 0, which is what the GLM model shipped with the
                          page does too -- so the weight wall renders hatched, "not scanned".
  manifest.json           one entry, so the page opens straight on this model.

Nothing here is a guess: every number written is either read out of coverage.json or computed
from it with the arithmetic named beside it.

    python3 tools/atlas_export.py [--stats PATH] [--gates PATH] [--out DIR] [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
ATLAS_DIR = os.path.join(TOOLS, "atlas")
OUT_DEFAULT = os.path.join(ATLAS_DIR, "models")
STATS_DEFAULT = os.path.join(ROOT, "results", "keepsets", "topics", "coverage.json")
GATES_DEFAULT = os.path.join(ROOT, "results", "keepsets", "gates.json")
PAIRS_SIBLING = "pairs.json"   # tools/expert_stats.py --pairs sibling, beside the stats file
TUNE_PY = os.path.join(TOOLS, "tune.py")

SLUG = "deepseek-v4.1-flash"
NAME = "DeepSeek-V4.1-Flash"

# Layer and expert counts come from budget.py, which is where the engine's own copy of
# config.json's shapes lives. The four below are only used to size the weight inventory and
# are the ones `tools/v41_ref.py` Args states; they need no torch to be quoted.
DIM = 5120               # hidden size
MOE_INTER = 2304         # moe_inter_dim
VOCAB = 129280
Q_LORA = 1280


# --- freshness --------------------------------------------------------------
# The export is a picture of three files. When any of them is newer than the picture, the
# picture is wrong, and `a` in the TUI has to re-take it before it serves anything.

def inputs(stats: str = None, gates: str = None) -> list:
    """Everything the export is derived from: the trace, the gate index, and the profile
    list -- `tools/tune.py` is in here because PROFILES is what decides which ten keep-sets
    are drawn at all."""
    # The co-routing sibling is deliberately NOT in here: tools/expert_stats.py writes
    # pairs.json and coverage.json in the same run, so the stats file's own mtime already
    # moves whenever the tables do, and a second entry would only make the check noisier.
    return [stats or STATS_DEFAULT, gates or GATES_DEFAULT, TUNE_PY]


def pairs_path(stats: str) -> str:
    """Where the co-routing tables live when they are not inside the stats file itself."""
    return os.path.join(os.path.dirname(os.path.abspath(stats)), PAIRS_SIBLING)


def is_stale(out: str = None, stats: str = None, gates: str = None, slug: str = SLUG) -> bool:
    """True when the export is missing or older than anything it was made from."""
    out = out or OUT_DEFAULT
    made = [os.path.join(out, "manifest.json"),
            os.path.join(out, slug, "insights.json"),
            os.path.join(out, slug, "atlas.jsonl")]
    if not all(os.path.exists(p) for p in made):
        return True
    oldest = min(os.path.getmtime(p) for p in made)
    for src in inputs(stats, gates):
        if os.path.exists(src) and os.path.getmtime(src) > oldest:
            return True
    return False


def read_profiles(path: str = None) -> list:
    """tools/tune.py's PROFILES without importing it -- tune.py pulls in curses, and it
    imports this module, so an import here would close a circle. The list is a literal."""
    src = open(path or TUNE_PY, encoding="utf-8").read()
    start = src.index("PROFILES = [")
    end = src.index("\n]", start) + 2
    ns: dict = {}
    exec(compile(src[start:end], "tune.py", "exec"), ns)          # noqa: S102
    return ns["PROFILES"]


# --- small arithmetic -------------------------------------------------------

def g4(x: float) -> float:
    """4 significant digits. 40 x 384 x ~50 matrices is 700k numbers, and the full repr of
    each is three times the file for digits nothing on the page can show."""
    return float(f"{x:.4g}")


def gini(v) -> float:
    """Gini of a non-negative vector -- how unevenly this layer's routing is spread."""
    s = sorted(float(x) for x in v)
    n = len(s)
    tot = sum(s)
    if tot <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(s, 1):
        cum += i * x
    return (2 * cum) / (n * tot) - (n + 1) / n


def shannon_bits(v) -> float:
    tot = sum(v)
    if tot <= 0:
        return 0.0
    h = 0.0
    for x in v:
        if x > 0:
            p = x / tot
            h -= p * math.log2(p)
    return h


# --- the two summaries the router trace did not used to keep --------------------

def _by_topic(per_layer: dict, prefix: str) -> dict:
    """`{topic: {layer: row}}` for every `<prefix><topic>` key present in ALL 40 layers.

    A topic that is only there for some layers is dropped rather than half-drawn: the grid is
    one matrix and a hole in it is not something the page can show."""
    out: dict = {}
    for L in range(B.N_LAYERS):
        for k, v in (per_layer.get(str(L)) or {}).items():
            if k.startswith(prefix):
                out.setdefault(k[len(prefix):], {})[L] = v
    return {t: rows for t, rows in out.items() if len(rows) == B.N_LAYERS}


def read_extras(cov_path: str) -> dict:
    """The `top1_<topic>` histograms and the co-routing tables, if this trace kept them.

    Both arrived after the first traces were taken (2026-09-14) and both are optional: a
    coverage.json written before then has neither, and the two fields they fill are then left
    out of the export entirely, which is what the page's patched cards already handle.

    The tables are looked for inside the stats file first (`--pairs inline`) and then in
    `pairs.json` beside it (`--pairs sibling`, the default -- they are ~2 MB at 39 topics and
    the engine parses coverage.json on every start).
    """
    d = json.load(open(cov_path, encoding="utf-8"))
    pl = d.get("per_layer") or {}
    top1 = _by_topic(pl, "top1_")
    pairs = _by_topic(pl, "pairs_")
    meta = d.get("pairs_meta") or {}
    if not pairs:
        sib = pairs_path(cov_path)
        if os.path.exists(sib):
            side = json.load(open(sib, encoding="utf-8"))
            pairs = _by_topic(side.get("per_layer") or {}, "pairs_")
            meta = {k: v for k, v in side.items() if k != "per_layer"}
    return {"top1": top1, "pairs": pairs, "pairs_meta": meta}


def build_top1_share(top1_by_topic: dict, topics) -> list:
    """`top1_share[L][e]`: the share of tokens whose largest gate weight went to expert e.

    Summed over the same tagged topics every other view here is built from, then normalised
    per layer -- so each row sums to 1, which is what the page's `top1_share` means and what
    its "uniform = 1/E" reference is drawn against.

    There is no per-domain variant: `reap_domains` is the only sliced matrix the card reads,
    and a `top1_domains` would be drawn by nothing."""
    rows = []
    for L in range(B.N_LAYERS):
        acc = [0.0] * B.N_EXPERTS
        for t in topics:
            r = top1_by_topic[t][L]
            for e in range(B.N_EXPERTS):
                acc[e] += float(r[e])
        tot = sum(acc) or 1.0
        rows.append([g4(x / tot) for x in acc])
    return rows


def build_coroute(pairs_by_topic: dict, topics, cnt_all: list, min_count: int, keep: int = 16):
    """The co-routing pair table per layer, in the shape the page's router ring reads:
    `{count: [[a, b, n, frac] x 16], lift: [[a, b, n, lift] x 16], min_count}`.

    `n` is how many tokens took both experts, `frac` is `n / tokens`, and

        lift(a, b) = n / (count(a) * count(b) / tokens)

    is recomputed HERE, from the summed pair counts and the global routing counts in
    `cnt_all`, rather than averaged out of the per-topic lifts -- the marginals are exact, so
    the lift is exact for any pair that reaches this table.

    The one approximation is which pairs reach it. `tools/expert_stats.py` keeps the top ~64
    pairs per topic per layer, not all C(384, 2) = 73,536 of them, so a pair that is rank 33 in
    every single topic and would have been in the global top 16 is not in the candidate pool.
    Keeping the whole table per topic would be 39 x 40 x 73,536 rows to make that impossible.
    """
    out = []
    for L in range(B.N_LAYERS):
        agg: dict = {}
        for t in topics:
            for a, b, n, _lift in pairs_by_topic[t][L]:
                key = (int(a), int(b))
                agg[key] = agg.get(key, 0) + int(n)
        marg = cnt_all[L]
        tokens = sum(marg) / B.TOPK
        rows = []
        for (a, b), n in agg.items():
            expected = marg[a] * marg[b] / tokens if tokens > 0 else 0.0
            rows.append((a, b, n, (n / expected) if expected > 0 else 0.0))
        by_count = sorted(rows, key=lambda r: (-r[2], r[0], r[1]))[:keep]
        eligible = [r for r in rows if r[2] >= min_count]
        by_lift = sorted(eligible, key=lambda r: (-r[3], r[0], r[1]))[:keep]
        out.append({
            "count": [[a, b, n, round(n / tokens, 5) if tokens else 0.0] for a, b, n, _ in by_count],
            "lift": [[a, b, n, round(lf, 2)] for a, b, n, lf in by_lift],
            "min_count": int(min_count),
        })
    return out


# --- the routing block ------------------------------------------------------

def build_routing(cov_path: str, profiles, gates):
    sal = B.TopicIndex(cov_path, source="saliency")
    cnt = B.TopicIndex(cov_path, source="counts")
    topics = [t for t in sal.topics if t in cnt.counts]
    layers = list(range(B.N_LAYERS))
    E = B.N_EXPERTS

    # per (layer, expert) sums over every topic in the file -- the global view. per_layer[L]
    # ["counts"] also exists but is a differently scoped mixed histogram (695,388 routed
    # slots in L0 against 883,344 across the tagged topics), so the global here is built out
    # of the same tagged topics every other view on this page uses.
    sal_all = [[0.0] * E for _ in layers]
    cnt_all = [[0.0] * E for _ in layers]
    for t in topics:
        for L in layers:
            st, ct = sal.counts[t][L], cnt.counts[t][L]
            for e in range(E):
                sal_all[L][e] += st[e]
                cnt_all[L][e] += ct[e]

    reap = [[g4(x) for x in row] for row in sal_all]
    route_share, contribution = [], []
    for L in layers:
        tot = sum(cnt_all[L]) or 1.0
        route_share.append([g4(c / tot) for c in cnt_all[L]])
        # saliency is a SUM of gate_weight x ||expert_out|| over the tokens routed to the
        # expert (tools/expert_stats.py saliency_hist); dividing by the routing count of the
        # same tokens gives REAP's own per-token MEAN, which is what the page calls a
        # contribution.
        contribution.append([g4(sal_all[L][e] / cnt_all[L][e]) if cnt_all[L][e] > 0 else 0.0
                             for e in range(E)])

    # Each topic's row is rescaled to the global row's total for that layer. The page's own
    # domain rows are per-domain MEANS, so a domain row and the global row are already on one
    # scale there and its "divide by all domains" view reads as a pure ratio; our per-topic
    # SUMS are each about 1/39 of the global sum, and left raw that view would be a constant
    # -5.3 log2 offset with the signal buried under it. The rescale is exactly the per-layer
    # normalisation the engine does before it ranks (docs/keep-sets.md), so the ratio then
    # means "this topic's share of the layer against every topic's share".
    reap_domains = OrderedDict()
    for t in topics:
        rows = []
        for L in layers:
            row = sal.counts[t][L]
            s = sum(row)
            k = (sum(sal_all[L]) / s) if s > 0 else 0.0
            rows.append([g4(x * k) for x in row])
        reap_domains[t] = rows

    domains = list(topics)

    # --- the shipped profiles, as outlines ---------------------------------
    # A profile's keep-set is the engine's: per layer the top-N of the water-filling order
    # over the profile's topics under the rule its gate was measured with, N = keep_n(keep).
    gate_by_record = {r["record"]: r for r in gates["runs"]}
    prune_sets = OrderedDict()
    profile_meta = []
    # The trailing fields are optional in tune.py's literal -- a profile may name a
    # thinking default after its ranking rule -- and none of them is drawn here.
    for name, desc, sel, record, rank, *_rest in profiles:
        rec = gate_by_record.get(record)
        if rec is None:
            continue
        keep, rank_used, source = rec["keep"], rec["rank"], rec["source"]
        if source != "saliency":
            raise SystemExit(f"{record}: gate was measured on {source}, this exporter ranks saliency")
        idx = sal if source == "saliency" else cnt
        selection = tuple(sorted(t for t in sel if t in idx.counts))
        order = idx.curves(selection, only=selection, rank=rank_used)[1]
        n = B.keep_n(keep)
        key = f"{name} · keep {round(keep * 100)}%"
        prune_sets[key] = [sorted(order[L][:n]) for L in layers]

        # the same selection as a colour field: per-layer-normalised saliency summed over the
        # profile's topics, rescaled so each row totals what the global row totals -- which
        # makes the page's ratio view read directly.
        field = []
        for L in layers:
            acc = [0.0] * E
            for t in selection:
                row = idx.counts[t][L]
                s = sum(row) or 1.0
                for e in range(E):
                    acc[e] += row[e] / s
            scale = (sum(sal_all[L]) or 1.0) / (len(selection) or 1)
            field.append([g4(a * scale) for a in acc])
        dkey = f"profile/{record}"
        reap_domains[dkey] = field
        domains.append(dkey)
        profile_meta.append({"name": name, "record": record, "desc": desc, "keep": keep,
                             "experts_per_layer": n, "rank": rank_used, "source": source,
                             "topics": list(selection), "prune_set_key": key,
                             "domain_key": dkey, "gate_run": rec["run"]})

    # --- top-1 and co-routing, when the stats file carries them -------------
    # Both are all-or-nothing across the tagged topics: a matrix built from 30 of 39 topic
    # slices would be normalised against a different token population than `route_share` beside
    # it, and the page puts the two in adjacent tiles as though they were one measurement.
    extras = read_extras(cov_path)
    has_top1 = [t for t in topics if t in extras["top1"]]
    has_pairs = [t for t in topics if t in extras["pairs"]]
    top1_all = None
    top1_share = None
    if len(has_top1) == len(topics) and topics:
        top1_all = [[sum(float(extras["top1"][t][L][e]) for t in topics) for e in range(E)]
                    for L in layers]
        top1_share = build_top1_share(extras["top1"], topics)
    elif has_top1:
        print(f"  top1_share left out: {len(has_top1)} of {len(topics)} topics carry a "
              f"top1_<topic> histogram (re-run tools/expert_stats.py over every topic's trace)")
    coroute = None
    if len(has_pairs) == len(topics) and topics:
        coroute = build_coroute(extras["pairs"], topics, cnt_all,
                                int(extras["pairs_meta"].get("min_count", 5)))
    elif has_pairs:
        print(f"  coroute left out: {len(has_pairs)} of {len(topics)} topics carry a "
              f"pairs_<topic> table")

    dynamics_all = []
    for L in layers:
        toks = sum(cnt_all[L]) / B.TOPK
        bits = shannon_bits(cnt_all[L])
        row = {
            "layer": L,
            "tokens": int(round(toks)),
            "effective": g4(2 ** bits),                  # perplexity of the selection histogram
            "entropy": g4(bits / math.log2(E)),          # normalised 0..1
            "selected_gini": g4(gini(cnt_all[L])),
            "saliency_gini": g4(gini(sal_all[L])),
            # margin and mean top-1 weight still need per-token gate weights, which the
            # histograms in coverage.json do not carry; nothing on the expert-atlas card
            # reads either of them.
        }
        if top1_all is not None:
            # how unevenly the FIRST picks are spread, against `selected_gini`'s all-six
            # spread -- the card prints it in the layer-margin tooltip
            row["top1_gini"] = g4(gini(top1_all[L]))
        dynamics_all.append(row)

    routing = {
        "layers": layers,
        "experts": E,
        "domains": domains,
        "reap": reap,
        "reap_domains": reap_domains,
        "route_share": route_share,
        "contribution": contribution,
        "dynamics": {"all": dynamics_all},
        "prune_sets": prune_sets,
    }
    # Optional, and omitted rather than faked when the trace behind this file predates them.
    if top1_share is not None:
        routing["top1_share"] = top1_share
    if coroute is not None:
        routing["coroute"] = coroute
    tokens_per_layer = {str(L): dynamics_all[L]["tokens"] for L in layers}
    return routing, profile_meta, topics, tokens_per_layer


def build_insights(cov_path: str, gates_path: str, profiles):
    gates = json.load(open(gates_path, encoding="utf-8"))
    routing, profile_meta, topics, tokens_layer = build_routing(cov_path, profiles, gates)
    total_tokens = sum(tokens_layer.values()) // B.N_LAYERS
    inv = weight_inventory()
    routed = sum(t["numel"] for t in inv
                 if t["component"] in ("mlp.up", "mlp.down") and ".experts." in t["name"])
    dense = sum(t["numel"] for t in inv) - routed

    meta = {
        "model": NAME,
        "parameters": dense + routed,
        "active_parameters": int(dense + routed * B.TOPK / B.N_EXPERTS),
        "layers": B.N_LAYERS,
        "routed_layers": B.N_LAYERS,
        "experts": B.N_EXPERTS,
        "active_experts": B.TOPK,
        "records": len(topics),
        "estimated_tokens": total_tokens,
        "reap_tokens_per_layer": tokens_layer,
        "routed_tokens_per_layer": total_tokens,
        "tensor_count": len(inv),
        "source": {
            "coverage_json": os.path.relpath(cov_path, ROOT),
            "gates_json": os.path.relpath(gates_path, ROOT),
            "saliency": "sum over traced tokens of gate_weight x ||expert_output||, REAP's "
                        "criterion (Lasby et al., arXiv 2510.13999), tools/expert_stats.py",
            "keep_sets": "tools/budget.py TopicIndex(source='saliency').curves(sel, "
                         "rank='maxmin')[1], top keep_n(keep) per layer",
            "weights": "architecture-derived inventory from tools/v41_ref.py Args; no "
                       "checkpoint scan was taken, so every weight statistic is 0 and the "
                       "wall is hatched",
        },
        "profiles": profile_meta,
    }
    if "top1_share" in routing:
        meta["source"]["top1"] = ("share of tokens whose largest gate weight went to the expert, "
                                  "summed over the topic slices and normalised per layer; "
                                  "tools/expert_stats.py top1_hist")
    if "coroute" in routing:
        meta["source"]["coroute"] = ("tokens that took both experts, out of the top pairs "
                                     "tools/expert_stats.py pair_table keeps per topic; lift = "
                                     "count / (count_a x count_b / tokens), recomputed here "
                                     "against the global routing counts")
    return {"schema": 1, "meta": meta, "routing": routing}


# --- the weight inventory ---------------------------------------------------

ZERO = {k: 0.0 for k in ("mean", "std", "absmax", "absmean", "p50", "p90", "p99", "p999",
                         "p9999", "kurtosis", "skew", "sparsity", "outlier_3s", "outlier_4s",
                         "outlier_6s", "dyn_range")}


def tensor(name, shape, component, layer):
    numel = 1
    for s in shape:
        numel *= s
    return {"name": name, "shape": list(shape), "dtype": "unscanned", "numel": numel,
            **ZERO, "hist_log2": [0.0] * 32, "component": component, "layer": layer,
            "shard": "not-scanned"}


def weight_inventory():
    """Only tensors whose shape follows directly from tools/v41_ref.py Args. Attention beyond
    wq_a, the norms, the MTP/DSpark head and the Engram projections are left out rather than
    guessed, so the parameter totals here are a floor, not the checkpoint's."""
    out = [tensor("model.language_model.embed_tokens.weight", [VOCAB, DIM], "embed", None)]
    for L in range(B.N_LAYERS):
        p = f"model.language_model.layers.{L}."
        out += [
            tensor(p + "attn.wq_a.weight", [Q_LORA, DIM], "attn.q", L),
            tensor(p + "ffn.gate.weight", [B.N_EXPERTS, DIM], "mlp.gate", L),
            tensor(p + "ffn.experts.w1w3.weight", [B.N_EXPERTS, 2 * MOE_INTER, DIM], "mlp.up", L),
            tensor(p + "ffn.experts.w2.weight", [B.N_EXPERTS, DIM, MOE_INTER], "mlp.down", L),
            tensor(p + "ffn.shared_experts.w1w3.weight", [2 * MOE_INTER, DIM], "mlp.up", L),
            tensor(p + "ffn.shared_experts.w2.weight", [DIM, MOE_INTER], "mlp.down", L),
        ]
    out.append(tensor("lm_head.weight", [VOCAB, DIM], "lm_head", None))
    return out


# --- the manifest -----------------------------------------------------------

def manifest_entry(payload: dict, slug: str) -> dict:
    """One entry, so the page opens on this model without being asked.

    `shell` is the wording the page would otherwise take from the checkpoint it shipped with:
    its intro card, two of its stat tiles and its region sub-head are prose about that model,
    and none of it is derived from the data (tools/atlas/UPSTREAM.md)."""
    m, R = payload["meta"], payload["routing"]
    n_topics = len(R["domains"]) - len(R["prune_sets"])
    return {
        "slug": slug,
        "name": NAME,
        "kind": "glm-live",
        "total_params": m["parameters"],
        "active_params": m["active_parameters"],
        "note": (f"{m['layers']} layers x {m['experts']} routed experts, top-{m['active_experts']}. "
                 "Routing measured by tools/expert_trace.py over a tagged corpus and reduced by "
                 "tools/expert_stats.py; the colour is REAP saliency (gate weight x "
                 "expert-output norm). The outlines are the shipped profile keep-sets, each at "
                 "the keep fraction its generation gate was measured at. No weight scan was "
                 "taken: every per-tensor statistic is 0 and the wall is hatched."),
        "shell": {
            "note": (f"A {m['layers']}-layer mixture of {m['experts']} routed experts per layer, "
                     f"{m['active_experts']} of them taken per token, seen through the routing "
                     "trace of a tagged corpus rather than through its weights. The grid below "
                     "is the whole expert field: one column per expert, one row per layer, "
                     "coloured by how much of the output each expert actually carried."),
            "rhythm": ("A keep-set is a cache policy: per layer only the top-N experts stay "
                       "resident, and which N they are is decided by this trace. The outlines "
                       "are the ten shipped profiles, each at the keep fraction its generation "
                       "gate was run at -- so the picture is what the box actually holds."),
            "tiles": [None,
                      [str(m["layers"]), "layers traced"],
                      [str(n_topics), "topic slices"],
                      [f"{m['experts']} → {m['active_experts']}", "routed experts → selected"]],
            "metric": "not scanned",
            "region_sub": ("expert routing measured on the deployed checkpoint · no weight scan "
                           "was taken, so the wall above is hatched"),
        },
    }


# --- output -----------------------------------------------------------------

def export(stats: str = None, gates: str = None, out: str = None, slug: str = SLUG,
           profiles=None) -> dict:
    """Write the three files and return what was written. Raises the usual OSError /
    ValueError if an input is missing or malformed -- the caller decides what to say."""
    stats = stats or STATS_DEFAULT
    gates = gates or GATES_DEFAULT
    out = out or OUT_DEFAULT
    dest = os.path.join(out, slug)
    os.makedirs(dest, exist_ok=True)

    payload = build_insights(stats, gates, profiles if profiles is not None else read_profiles())
    ins = os.path.join(dest, "insights.json")
    jsonl = os.path.join(dest, "atlas.jsonl")
    mpath = os.path.join(out, "manifest.json")
    with open(ins, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    with open(jsonl, "w", encoding="utf-8") as f:
        for t in weight_inventory():
            f.write(json.dumps(t, separators=(",", ":")) + "\n")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump([manifest_entry(payload, slug)], f, indent=2)

    R = payload["routing"]
    return {"insights": ins, "atlas": jsonl, "manifest": mpath, "payload": payload,
            "layers": len(R["layers"]), "experts": R["experts"],
            "topics": len(R["domains"]) - len(R["prune_sets"]),
            "prune_sets": len(R["prune_sets"]), "bytes": os.path.getsize(ins)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", default=STATS_DEFAULT, metavar="PATH",
                    help="the coverage.json the grid is drawn from (default %(default)s)")
    ap.add_argument("--gates", default=GATES_DEFAULT, metavar="PATH",
                    help="the generation-gate index the outlines are budgeted from")
    ap.add_argument("--out", default=OUT_DEFAULT, metavar="DIR",
                    help="where the model files go; the page looks beside its own index.html")
    ap.add_argument("--slug", default=SLUG, help=argparse.SUPPRESS)
    ap.add_argument("--force", action="store_true",
                    help="write even when the export is newer than everything it reads")
    a = ap.parse_args(argv)

    for p in (a.stats, a.gates):
        if not os.path.exists(p):
            print(f"no such file: {p}", file=sys.stderr)
            return 2
    if not a.force and not is_stale(a.out, a.stats, a.gates, a.slug):
        print(f"up to date: {os.path.relpath(a.out, ROOT)} is newer than "
              f"{', '.join(os.path.relpath(x, ROOT) for x in inputs(a.stats, a.gates))}")
        return 0

    print("reading", os.path.relpath(a.stats, ROOT))
    r = export(a.stats, a.gates, a.out, a.slug)
    print(f"  {r['layers']} layers x {r['experts']} experts")
    print(f"  domains: {r['topics']} topic slices + {r['prune_sets']} profile fields")
    for k, v in r["payload"]["routing"]["prune_sets"].items():
        print(f"    {k:38s} {len(v[0])} experts/layer")
    extra = [k for k in ("top1_share", "coroute") if k in r["payload"]["routing"]]
    print("  top-1 / co-routing: " + (", ".join(extra) if extra else
          "neither -- this trace kept no top1_<topic> or pairs_<topic>, both fields omitted"))
    print(f"  wrote {os.path.relpath(r['insights'], ROOT)} ({r['bytes'] / 1e6:.1f} MB)")
    print(f"  wrote {os.path.relpath(r['atlas'], ROOT)}")
    print(f"  wrote {os.path.relpath(r['manifest'], ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
