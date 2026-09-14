#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_think_controls.sh -- does capping the reasoning span help this box?
#
# Four generation-gate runs at the keep fraction that holds a filled 256k
# context (0.36), two profiles, with the reasoning-span controls on:
#
#   1  Frontend   DSV41_THINK_BUDGET=8000
#   2  Backend    DSV41_THINK_BUDGET=8000
#   3  Frontend   DSV41_THINK_BUDGET=8000  DSV41_THINK_REPEAT_BREAK=12
#   4  Backend    DSV41_THINK_BUDGET=8000  DSV41_THINK_REPEAT_BREAK=12
#
# The controls-off baselines on the same keep-sets are already in the record
# (results/keepsets/{frontend,backend}/GATE.md), so this does not re-run them:
# the comparison is against those sections, and GATE.md now names the controls
# each run was taken with.
#
# Each run stops the server, restarts it with that profile's keep-set and those
# two variables, waits for /health, and appends a dated section to
# results/keepsets/<profile>/GATE.md. A summary of all four goes to
# /tmp/think_controls.log.
#
#   ./tools/verify_think_controls.sh                # all four, in order
#   ./tools/verify_think_controls.sh 3 4            # only those runs (resume)
#   ./tools/verify_think_controls.sh --list         # what it would do, no HTTP
#
# Budget an hour or two per run plus a warm start each, and hold the box: it
# serves one engine at a time and two loads at once wedge it.
#
# Honours: PYTHON (default ./.venv/bin/python, else python3), PORT (from .env,
# else 8000), KEEP (0.36), MAX_SEQ (262144), BUDGET (8000), NGRAM (12), and
# EFFORT / MAX_TOKENS (gate_profile's own defaults when unset).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=/tmp/think_controls.log
KEEP="${KEEP:-0.36}"
MAX_SEQ="${MAX_SEQ:-262144}"
BUDGET="${BUDGET:-8000}"
NGRAM="${NGRAM:-12}"

err() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "=== $*"; echo "=== $*" >> "$LOG"; }

[[ -f tools/gate_profile.py ]] || err "run this from the checkout (tools/gate_profile.py not found)"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x .venv/bin/python ]]; then PYTHON="$PWD/.venv/bin/python"; else PYTHON="python3"; fi
fi
"$PYTHON" -c "import sys" >/dev/null 2>&1 || err "PYTHON=$PYTHON does not run"

PORT="${PORT:-}"
if [[ -z "$PORT" && -f .env ]]; then
    PORT="$(sed -n 's/^[[:space:]]*PORT=\([0-9][0-9]*\).*/\1/p' .env | tail -1 || true)"
fi
PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT/v1"

# A value for either control in .env would beat the exported one: start.sh
# sources .env after capturing its own whitelist, and these two are not on it.
if [[ -f .env ]] && grep -qE '^[[:space:]]*DSV41_THINK_(BUDGET|REPEAT_BREAK)=' .env; then
    err ".env sets DSV41_THINK_BUDGET or DSV41_THINK_REPEAT_BREAK and would override this script; \
comment those lines out and run again"
fi

#                 1         2        3         4
RUN_PROFILE=(-  Frontend Backend  Frontend  Backend)
RUN_NGRAM=(-    0        0        "$NGRAM"  "$NGRAM")

LIST=false
WANT=()
for a in "$@"; do
    case "$a" in
        --list) LIST=true ;;
        [1-4])  WANT+=("$a") ;;
        *)      err "arguments are run numbers 1-4, or --list (not '$a')" ;;
    esac
done
[[ ${#WANT[@]} -eq 0 ]] && WANT=(1 2 3 4)

# The keep-set a profile implies, as ./tune.sh computes it: the same keys the
# tune screen writes into .env, so a run here reproduces a run started there.
profile_env() {
    local out status=0
    out="$("$PYTHON" tools/tune.py --profile "$1" --keep "$KEEP" --max-seq "$MAX_SEQ" --print 2>/dev/null)" \
        || status=$?
    [[ $status -eq 0 ]] || return 1
    printf '%s\n' "$out"
}

# results/keepsets/<slug>/GATE.md, resolved by gate_profile's own slug rule
# rather than by a path spelled out here.
gate_out() {
    "$PYTHON" - "$1" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import gate_profile as G
name, _ = G.find_profile(sys.argv[1])
print(os.path.join("results", "keepsets", G.slug(name), "GATE.md"))
PY
}

one_run() {
    local id="$1"
    local profile="${RUN_PROFILE[$id]}" ngram="${RUN_NGRAM[$id]}"
    local label="run $id: $profile, budget=$BUDGET, repeat-break=$ngram"
    local out penv rc=0
    out="$(gate_out "$profile")" || err "no profile named '$profile' (tools/tune.py:PROFILES)"
    # The budget model reads live memory, so with an engine still loaded the
    # profile does not "fit". Derive after the stop; for --list, show what can
    # be shown now.
    penv="$(profile_env "$profile")" || penv=""

    if $LIST; then
        echo "$label"
        echo "    record  $out"
        if [ -n "$penv" ]; then printf '%s\n' "$penv" | sed 's/^/    env     /'
        else echo "    env     (an engine is loaded now; derived after ./stop.sh at run time)"; fi
        echo "    env     DSV41_THINK_BUDGET=$BUDGET DSV41_THINK_REPEAT_BREAK=$ngram"
        echo "    gate    $PYTHON tools/gate_profile.py --profile $profile --thinking on --url $URL"
        return 0
    fi

    say "$label  ($(date '+%Y-%m-%d %H:%M:%S'))"
    ./stop.sh >>"$LOG" 2>&1 || true
    # give the arena back before the budget model looks at MemAvailable
    for _ in $(seq 1 30); do
        avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 999)
        [ "$avail" -ge 90 ] && break; sleep 2
    done
    penv="$(profile_env "$profile")" \
        || err "tune.py says $profile at keep $KEEP / max_seq $MAX_SEQ does not fit this box (after stop)"

    # One subshell per run: the keep-set and the two controls leave with it, so
    # run 3 cannot inherit run 1's environment.
    (
        set -a
        eval "$penv"
        DSV41_THINK_BUDGET="$BUDGET"
        DSV41_THINK_REPEAT_BREAK="$ngram"
        set +a
        {
            echo "    keep-set  PRUNE_KEEP=$PRUNE_KEEP ARENA_GB=$ARENA_GB MAX_SEQ=$MAX_SEQ" \
                 "rank=$DSV41_PRUNE_RANK source=$DSV41_PRUNE_SOURCE"
            echo "    topics    ${EXPERT_TOPICS:-<none>}"
            echo "    controls  DSV41_THINK_BUDGET=$DSV41_THINK_BUDGET" \
                 "DSV41_THINK_REPEAT_BREAK=$DSV41_THINK_REPEAT_BREAK"
        } | tee -a "$LOG"

        ./start.sh >>"$LOG" 2>&1 \
            || err "the server did not come up for $label (see logs/server.log)"
        curl -sf "http://127.0.0.1:$PORT/health" >>"$LOG" 2>&1 || true
        echo >> "$LOG"

        # The gate exits nonzero whenever any prompt failed, which is the normal
        # outcome here, so its status is recorded and not acted on.
        local runlog
        runlog="$(mktemp)"
        (
            set -o pipefail
            "$PYTHON" tools/gate_profile.py --profile "$profile" --thinking on \
                --url "$URL" --out "$out" \
                ${EFFORT:+--effort $EFFORT} ${MAX_TOKENS:+--max-tokens $MAX_TOKENS} \
                2>&1 | tee "$runlog" | tail -30
        ) || rc=$?
        cat "$runlog" >> "$LOG"
        {
            echo "--- $label"
            echo "    gate exit $rc (nonzero = at least one prompt failed; read the counts)"
            grep -E "^[0-9]+ of [0-9]+ runs passed|finished a correct answer" "$runlog" | tail -2 || true
            echo "    appended to $out"
            echo
        } >> "$LOG"
        rm -f "$runlog"
    )
    ./stop.sh >>"$LOG" 2>&1 || true
}

if ! $LIST; then
    {
        echo
        echo "############################################################"
        echo "# reasoning-span controls: keep $KEEP, max_seq $MAX_SEQ, budget $BUDGET, n $NGRAM"
        echo "# started $(date '+%Y-%m-%d %H:%M:%S'), runs ${WANT[*]}"
        echo "############################################################"
    } >> "$LOG"
fi

for r in "${WANT[@]}"; do
    one_run "$r"
done

if ! $LIST; then
    say "done $(date '+%Y-%m-%d %H:%M:%S'); full output and summary in $LOG"
    echo
    grep -E "^--- run |^    gate exit|of [0-9]+ runs passed|finished a correct answer" "$LOG" \
        | tail -40 || true
fi
