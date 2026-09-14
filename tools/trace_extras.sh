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
mkdir -p "$STAGE/runs"
for d in $TRACE_DIRS; do
    name=$(basename "$d")
    say "expert_stats over $d"
    # Defaults everywhere: the budget ladder is what the shipped file's `global` block was
    # measured on, and --pairs sibling keeps the co-routing tables out of the file the engine
    # parses on every start.
    "$PYTHON" tools/expert_stats.py --trace "$d" --out "$STAGE/runs/$name" \
        >"$STAGE/$name.log" 2>&1 \
        || { say "  FAILED -- tail of $STAGE/$name.log:"; tail -20 "$STAGE/$name.log"; exit 4; }
    say "  wrote $STAGE/runs/$name/coverage.json"
done

# --- 4. merge, then decide whether it may replace the shipped file -----------
# The merge is `tools/expert_stats.py --merge`, not a script of its own: a topic whose shipped
# histogram is the SUM of two traces has to be summed the same way (`reasoning_code` is one --
# its corpus was collected in two passes and each was traced separately), and which traces make
# up a topic is derived there by matching histograms rather than guessed from directory names.
# It replaces the shipped file only if every counts_<topic> and saliency_<topic> comes back
# element-for-element identical, and otherwise leaves it alone and writes coverage.new.json.
say "merging into $STATS_DIR/coverage.json"
set +e
"$PYTHON" tools/expert_stats.py --merge "$STATS_DIR/coverage.json" --runs "$STAGE"/runs/*
rc=$?
set -e
if [ "$rc" = "5" ]; then
    say "the merge was refused -- the shipped coverage.json is untouched, see above"
elif [ "$rc" != "0" ]; then
    say "the merge failed with exit $rc"
    exit "$rc"
fi

# --- 5. re-take the atlas export --------------------------------------------
# tools/atlas_export.py, not `tune.py --atlas-export`: --force is this tool's flag and tune.py
# has no such option, so routing it through the TUI made argparse reject the whole command.
# --force because the export is only stale against file times, and a coverage.json that was
# kept rather than replaced has not moved.
say "re-taking the Weight Atlas export"
"$PYTHON" tools/atlas_export.py --force

say "=== done ==="
