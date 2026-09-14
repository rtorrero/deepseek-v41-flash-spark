#!/usr/bin/env bash
# trace_extras.sh -- put the top-1 and co-routing summaries into the keep-set stats file, and
# re-take the Weight Atlas export from it.
#
# Nothing about the model changes here. `tools/expert_trace.py` has always stored everything
# these two summaries are made of -- `indices` (the six experts a token took, in the router's
# selection order) and `weights` (their gate weights) -- but `tools/expert_stats.py` only ever
# folded them into `counts_<topic>` and `saliency_<topic>`. It now also writes
# `top1_<topic>` (which of the six carried the largest gate weight) and a co-routing pair
# table, so this script re-runs the REDUCTION over whatever per-layer traces the box still has
# and only falls back to re-tracing the model when it has none.
#
# Usage, from the checkout root:
#
#     nohup tools/trace_extras.sh >/dev/null 2>&1 &
#     tail -f /tmp/trace_extras.log
#
# Environment:
#
#   PYTHON      the interpreter to run the tools with (default `python3`). numpy is the only
#               hard dependency of the reduction; torch is needed only if this has to trace.
#               On a box where the recipe runs out of a virtualenv, point it at that venv's
#               interpreter: `PYTHON=/path/to/.venv/bin/python tools/trace_extras.sh`.
#   ROOT        the checkout to work in (default: the parent of this script's directory).
#   STATS_DIR   where the keep-set stats live (default $ROOT/results/keepsets/topics).
#   TRACE_DIRS  space-separated per-layer trace directories to reduce. Default: every
#               directory under $ROOT/results that holds trace/layer0.npz.
#   MODEL_DIR / ENGRAM_DIR / CORPUS
#               only used when TRACE_DIRS finds nothing and the trace has to be taken again.
#   KEEP_ENGINE=1  do not stop the server first (the reduction is CPU-only; a re-trace is not).
#
# Writes /tmp/trace_extras.log and holds /tmp/patch/trace_extras.pid while it runs.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python3}
STATS_DIR=${STATS_DIR:-$ROOT/results/keepsets/topics}
LOG=${LOG:-/tmp/trace_extras.log}
PIDDIR=${PIDDIR:-/tmp/patch}
PIDFILE="$PIDDIR/trace_extras.pid"
STAGE=${STAGE:-${TMPDIR:-/tmp}/trace-extras-stage}   # scratch: never inside results/

mkdir -p "$PIDDIR"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running as PID $(cat "$PIDFILE") -- see $LOG" >&2
    exit 1
fi

exec >>"$LOG" 2>&1
echo $$ >"$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

say() { echo "$(date '+%H:%M:%S') $*"; }

cd "$ROOT"
say "=== trace_extras.sh in $ROOT with $($PYTHON -V 2>&1) ==="

# --- 1. give the box back its memory ----------------------------------------
# The reduction is numpy on a few hundred MB of npz and does not need the GPU, but a re-trace
# streams one 7.4 GB layer shard at a time and will not fit beside a loaded engine.
if [ "${KEEP_ENGINE:-0}" != "1" ] && [ -x ./stop.sh ]; then
    say "stopping the engine"
    ./stop.sh || say "stop.sh returned $? -- continuing"
fi

# --- 2. what is there to reduce ---------------------------------------------
if [ -z "${TRACE_DIRS:-}" ]; then
    TRACE_DIRS=""
    for d in results/*/; do
        [ -f "${d}trace/layer0.npz" ] && TRACE_DIRS="$TRACE_DIRS ${d%/}"
    done
fi
TRACE_DIRS=$(echo "$TRACE_DIRS" | tr -s ' ' | sed 's/^ //;s/ $//')

if [ -z "$TRACE_DIRS" ]; then
    # ---------------------------------------------------------------------------
    # UNCERTAIN, and deliberately so. The exact command that produced
    # results/keepsets/topics/coverage.json is NOT recoverable from this checkout, and the file
    # is not the output of one command at all:
    #
    #   * results/keepsets/topics/GATE.md names `corpus/trace_topics_v2.jsonl` (430 sequences,
    #     115,898 tokens) as the corpus of the base trace, and 115,898 x 6 = 695,388 is exactly
    #     the mixed `counts` total in layer 0 of the shipped file -- so the base trace is that
    #     corpus and nothing else;
    #   * but the tagged topics in the same layer total 883,344 routed slots (147,224 tokens),
    #     and the file carries 39 `counts_<topic>` histograms where that corpus carried 35. The
    #     four `reasoning*` topics were traced separately, each from its own small corpus, and
    #     their keys were copied across -- the merge recipe `tools/tune.py` prints in its topic
    #     brief ("Nothing already traced has to be traced again").
    #
    # So the reconstruction below is the BASE trace only. Run it, and this script's guard will
    # refuse to replace coverage.json because the reasoning topics will be missing from the
    # result; it will write coverage.new.json instead and say which keys it could not carry.
    # To get a complete file, put every corpus that contributed a topic on TRACE_DIRS (trace
    # each, then re-run this script) -- which is why reusing the traces already on disk is the
    # first thing tried above.
    # ---------------------------------------------------------------------------
    CORPUS=${CORPUS:-corpus/trace_topics_v2.jsonl}
    MODEL_DIR=${MODEL_DIR:-}
    ENGRAM_DIR=${ENGRAM_DIR:-engram_rows_topics}
    OUT=results/trace-topics-extras
    if [ -z "$MODEL_DIR" ] || [ ! -f "$CORPUS" ]; then
        say "no per-layer trace under results/ and nothing to trace with:"
        say "  MODEL_DIR=${MODEL_DIR:-<unset>} CORPUS=$CORPUS (exists: $([ -f "$CORPUS" ] && echo yes || echo no))"
        say "  set MODEL_DIR, ENGRAM_DIR and CORPUS, or point TRACE_DIRS at a trace directory."
        exit 3
    fi
    say "no trace on disk -- re-tracing $CORPUS into $OUT (hours; --resume checkpoints per layer)"
    say "  NOTE: this reconstructs the BASE trace only; see the comment above this line in the script"
    "$PYTHON" tools/expert_trace.py --model-dir "$MODEL_DIR" --corpus "$CORPUS" \
        --engram-dir "$ENGRAM_DIR" --out "$OUT" --layers 0-39 --resume
    TRACE_DIRS="$OUT"
fi

say "reducing: $TRACE_DIRS"

# --- 3. one reduction per trace ---------------------------------------------
rm -rf "$STAGE"
mkdir -p "$STAGE"
for d in $TRACE_DIRS; do
    name=$(basename "$d")
    say "expert_stats over $d"
    # Defaults everywhere: the budget ladder is what the shipped file's `global` block was
    # measured on, and --pairs sibling keeps the co-routing tables out of the file the engine
    # parses on every start.
    "$PYTHON" tools/expert_stats.py --trace "$d" --out "$STAGE/$name" >"$STAGE/$name.log" 2>&1 \
        || { say "  FAILED -- tail of $STAGE/$name.log:"; tail -20 "$STAGE/$name.log"; exit 4; }
    say "  wrote $STAGE/$name/coverage.json"
done

# --- 4. merge, then decide whether it may replace the shipped file -----------
STAGE="$STAGE" STATS_DIR="$STATS_DIR" "$PYTHON" - <<'PY'
"""Merge the per-trace reductions into one stats file and install it only if it loses nothing.

The shipped coverage.json is a merge of several traces (see the comment in trace_extras.sh),
so a run over fewer traces than went into it would silently drop topics -- and a dropped topic
is a keep-set that quietly stops covering a workload. The rule here is the conservative one:
the merged file replaces the shipped one ONLY when it carries every `counts_<topic>` and
`saliency_<topic>` key the shipped one has. Otherwise the shipped file stays exactly where it
is and the merge is written beside it as coverage.new.json for someone to look at.

Topics the shipped file does not name are dropped from the merge, so a stray trace lying about
under results/ (an old two-category run, say) cannot add slices to the atlas.
"""
import glob
import json
import os
import shutil

STAGE = os.environ["STAGE"]
STATS_DIR = os.environ["STATS_DIR"]
OLD = os.path.join(STATS_DIR, "coverage.json")

FAMILIES = ("counts_", "saliency_", "top1_")
PAIRS = "pairs_"


def topic_keys(per_layer, prefixes):
    return {k for row in per_layer.values() for k in row if k.startswith(tuple(prefixes))}


old = json.load(open(OLD, encoding="utf-8"))
old_pl = old["per_layer"]
old_keys = topic_keys(old_pl, ("counts_", "saliency_"))
topics = sorted({k.split("_", 1)[1] for k in old_keys})
print(f"  shipped file: {len(old_keys)} histogram keys over {len(topics)} topics")

runs = sorted(glob.glob(os.path.join(STAGE, "*", "coverage.json")))
if not runs:
    raise SystemExit("  no reduction to merge")


def run_topics(path):
    pl = json.load(open(path, encoding="utf-8"))["per_layer"]
    return {k.split("_", 1)[1] for k in topic_keys(pl, ("counts_",))} & set(topics)


# The run that covers the most of the shipped topics is the base: its per-layer scalars
# (`used`, `cov`, `entropy_bits`, `block6_unique_mean`) and its `global` budget ladder describe
# the largest trace, which is the same thing they described before.
covered = {p: run_topics(p) for p in runs}
runs.sort(key=lambda p: -len(covered[p]))
for p in runs:
    print(f"  {os.path.relpath(p, STAGE)}: {len(covered[p])} of the shipped topics")

merged = json.load(open(runs[0], encoding="utf-8"))
m_pl = merged["per_layer"]
pairs_path = os.path.join(os.path.dirname(runs[0]), "pairs.json")
m_pairs = json.load(open(pairs_path, encoding="utf-8")) if os.path.exists(pairs_path) else None

for p in runs[1:]:
    add = json.load(open(p, encoding="utf-8"))["per_layer"]
    taken = set()
    for layer, rows in add.items():
        if layer not in m_pl:
            continue
        for key, v in rows.items():
            if not key.startswith(FAMILIES) or key.split("_", 1)[1] not in topics:
                continue
            if key in m_pl[layer]:        # first run to carry a topic keeps it
                continue
            m_pl[layer][key] = v
            taken.add(key)
    print(f"  merged {len(taken)} keys from {os.path.relpath(p, STAGE)}")
    side = os.path.join(os.path.dirname(p), "pairs.json")
    if m_pairs is not None and os.path.exists(side):
        other = json.load(open(side, encoding="utf-8"))["per_layer"]
        for layer, rows in other.items():
            if layer not in m_pairs["per_layer"]:
                continue
            for key, v in rows.items():
                if key.startswith(PAIRS) and key.split("_", 1)[1] in topics:
                    m_pairs["per_layer"][layer].setdefault(key, v)

# nothing the shipped file did not name
dropped = set()
for rows in m_pl.values():
    for key in list(rows):
        if key.startswith(FAMILIES) and key.split("_", 1)[1] not in topics:
            del rows[key]
            dropped.add(key)
if m_pairs is not None:
    for rows in m_pairs["per_layer"].values():
        for key in list(rows):
            if key.startswith(PAIRS) and key.split("_", 1)[1] not in topics:
                del rows[key]
if dropped:
    print(f"  dropped {len(dropped)} keys for topics the shipped file does not name: "
          + ", ".join(sorted(dropped)[:6]))

missing = sorted(old_keys - topic_keys(m_pl, ("counts_", "saliency_")))
have_top1 = {k.split("_", 1)[1] for k in topic_keys(m_pl, ("top1_",))}
print(f"  merged file: {len(have_top1 & set(topics))} of {len(topics)} topics "
      f"carry top1_<topic>")

if missing:
    dest = os.path.join(STATS_DIR, "coverage.new.json")
    json.dump(merged, open(dest, "w", encoding="utf-8"))
    if m_pairs is not None:
        json.dump(m_pairs, open(os.path.join(STATS_DIR, "pairs.new.json"), "w", encoding="utf-8"))
    print(f"  KEPT the shipped coverage.json: the merge is missing {len(missing)} of its keys")
    print("  missing: " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    print(f"  wrote {dest} instead -- trace the missing topics and re-run with TRACE_DIRS set")
else:
    shutil.copy2(OLD, OLD + ".before-extras")
    json.dump(merged, open(OLD, "w", encoding="utf-8"))
    if m_pairs is not None:
        json.dump(m_pairs, open(os.path.join(STATS_DIR, "pairs.json"), "w", encoding="utf-8"))
    print(f"  replaced {OLD} (previous kept as coverage.json.before-extras), "
          f"{os.path.getsize(OLD) / 1e6:.1f} MB")
PY

# --- 5. re-take the atlas export --------------------------------------------
# --force because the export is only stale against file times, and a coverage.json that was
# kept rather than replaced has not moved.
say "re-taking the Weight Atlas export"
"$PYTHON" tools/tune.py --atlas-export --force

say "=== done ==="
