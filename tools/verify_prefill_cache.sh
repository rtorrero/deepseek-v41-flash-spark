#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_prefill_cache.sh -- does the prefill unpack cache pay for itself?
#
#   ./tools/verify_prefill_cache.sh                 # 6 GB cache, Frontend, keep 0.36
#   CACHE_GB=4 ./tools/verify_prefill_cache.sh      # a different budget
#   TOKENS=12000 ./tools/verify_prefill_cache.sh    # a longer prompt (more chunks)
#   NO_GATE=1 ./tools/verify_prefill_cache.sh       # skip the generation gate
#
# A CB3 expert has to be unpacked to packed FP4 before a prefill-sized kernel can read it, and
# that unpack is thrown away at the end of the call -- so chunk 1 of a prompt unpacks exactly the
# experts chunk 0 already unpacked. DSV41_PREFILL_UNPACK_CACHE_GB buys a region of the FP4 scratch
# arena that survives between chunks (tools/unpack_cache.py).
#
# The experiment, both halves on ONE box, one variable:
#
#   load A   the shipped configuration, cache OFF    -> one long prompt
#   load B   the same configuration, cache ON        -> the SAME prompt, byte for byte
#            and, while load B is still up, the generation gate, because a cache that
#            changes what the model writes is a bug and not a speedup
#
# Load B is second on purpose: the gate needs a server that is already warm, and running it on the
# cache-on load is what proves the cache is invisible to the output.
#
# The engine is restarted between the two, because the budget is read once at start-up and the
# arena is sized from what is free at that moment. Each restart is a fresh warm start -- about
# three minutes on this recipe -- so the whole script takes the better part of an hour with the
# gate on the end.
#
# Writes /tmp/prefill_cache.log and appends a record under results/prefill/.
# It does NOT touch a profile's GATE.md: that file is the record of keep-set quality, and this is
# a memory/throughput experiment that happens to run the gate as a control.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

CACHE_GB="${CACHE_GB:-6}"
PROFILE="${PROFILE:-Frontend}"
KEEP="${KEEP:-0.36}"
TOKENS="${TOKENS:-7000}"
URL="${DSV41_URL:-http://127.0.0.1:8000/v1}"
LOG="${LOG:-/tmp/prefill_cache.log}"
REC_DIR="results/prefill"
STAMP="$(date +%Y%m%d-%H%M)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

say() { printf '%s\n' "$*" | tee -a "$LOG"; }
hdr() { say ""; say "=== $* ==="; }

: > "$LOG"
say "prefill unpack cache A/B -- $(date '+%Y-%m-%dT%H:%M:%S%z')"
say "profile $PROFILE, keep $KEEP, cache $CACHE_GB GB, prompt ~$TOKENS tokens"

# --- preconditions --------------------------------------------------------
[[ -f start.sh && -f stop.sh ]] || { echo "run this from a full checkout" >&2; exit 1; }
[[ -f .env ]] || { echo "no .env; copy env.example and set MODEL_DIR" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
if [[ ! -r /proc/meminfo ]]; then
    echo "this script measures a GB10 box; /proc/meminfo is missing here" >&2
    exit 1
fi

# The profile's environment, straight from the planner, so nothing has to be typed twice and .env
# is left exactly as it is. Taken with NO cache budget in scope, because this is the shipped
# configuration both loads have to share -- the cache is the only variable.
unset DSV41_PREFILL_UNPACK_CACHE_GB
PROFILE_ENV="$WORK/profile.env"
if ! ./tune.sh --profile "$PROFILE" --keep "$KEEP" --print > "$PROFILE_ENV" 2>"$WORK/tune.err"; then
    cat "$WORK/tune.err" >&2
    echo "tune.sh refused $PROFILE at keep $KEEP" >&2
    exit 1
fi
say ""
say "configuration (tools/tune.py --profile $PROFILE --keep $KEEP):"
sed 's/^/  /' "$PROFILE_ENV" >> "$LOG"
sed 's/^/  /' "$PROFILE_ENV"

# What the cost model says before the box says anything. `tune.sh` LOWERS the keep fraction when a
# configuration does not fit, so the test is not "did it print something" -- it is whether the same
# keep survives the cache, and whether the verdict is still on the right side of the gate. A budget
# the model calls "over" is one the memory watchdog will find on the first request.
say ""
say "cost model with a ${CACHE_GB} GB cache on top (tools/budget.py):"
PLAN="$WORK/plan.env"
DSV41_PREFILL_UNPACK_CACHE_GB="$CACHE_GB" ./tune.sh --profile "$PROFILE" --keep "$KEEP" --print \
    > "$PLAN" 2>/dev/null || true
head -1 "$PLAN" | sed 's/^/  /' >> "$LOG"
head -1 "$PLAN" | sed 's/^/  /'
PLANNED_KEEP="$(awk -F= '/^PRUNE_KEEP=/ {print $2}' "$PLAN")"
if head -1 "$PLAN" | grep -q "will not load\|— over" || [[ "$PLANNED_KEEP" != "$KEEP" ]]; then
    say "REFUSING: a $CACHE_GB GB cache does not fit $PROFILE at keep $KEEP"
    say "  (the planner came back with keep ${PLANNED_KEEP:-none}). Lower CACHE_GB."
    exit 1
fi

# --- the prompt, built once and reused ------------------------------------
PROMPT="$WORK/prompt.txt"
say "$(python3 tools/longprefill.py --tokens "$TOKENS" --print-only --prompt-out "$PROMPT" 2>&1 |
    sed 's/^/  /')"
PROMPT_SUM="$(cksum < "$PROMPT" | awk '{print $1}')"
say "  prompt checksum $PROMPT_SUM (identical for both loads by construction)"

# --- helpers ---------------------------------------------------------------
wait_for_memory() {
    # The kernel releases an 80 GB arena lazily. Starting the next load before it has is how this
    # box gets wedged -- it answers ping and accepts TCP while sshd can no longer fork.
    # Plain `if`, not `cond && action`: an AND-list whose left side is false returns 1, and under
    # `set -e` that would abort the script on the first loop that has to wait.
    local want="${1:-90}" waited=0 avail
    while :; do
        avail=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
        if [[ "$avail" -ge "$want" ]]; then
            say "  MemAvailable ${avail} GiB, going ahead"
            return 0
        fi
        if [[ "$waited" -ge 600 ]]; then
            say "  MemAvailable stuck at ${avail} GiB after ${waited}s -- not starting on top of it"
            return 1
        fi
        sleep 15
        waited=$((waited + 15))
    done
}

run_load() {           # run_load <label> <cache_gb> <json-out>
    local label="$1" cache="$2" out="$3"
    hdr "load $label: cache ${cache} GB"
    ./stop.sh >/dev/null 2>&1 || true
    wait_for_memory 90 || return 1
    set -a
    # shellcheck disable=SC1090
    . "$PROFILE_ENV"
    set +a
    export DSV41_PREFILL_UNPACK_CACHE_GB="$cache"
    say "  starting (warm start is minutes, not seconds)"
    if ! ./start.sh >>"$LOG" 2>&1; then
        say "  start.sh failed; see $LOG and logs/server.log"
        return 1
    fi
    say "$(grep -E "prefill unpack cache|arena .* GB =|refusing to start" logs/server.log |
        tail -4 | sed 's/^/  /')"
    say "  sending the prompt"
    say "$(python3 tools/longprefill.py --tokens "$TOKENS" --url "$URL" --json "$out" 2>&1)"
}

# --- load A: the cache off, the shipped behaviour --------------------------
run_load "A (off)" 0 "$WORK/off.json"

# --- load B: the cache on --------------------------------------------------
run_load "B (on)" "$CACHE_GB" "$WORK/on.json"

# --- the generation gate, on the cache-on load -----------------------------
GATE_STATUS="skipped"
if [[ "${NO_GATE:-0}" != "1" ]]; then
    hdr "generation gate, cache on"
    say "  python3 tools/gate_profile.py --profile $PROFILE --thinking on"
    GATE_OUT="$WORK/gate.txt"
    if python3 tools/gate_profile.py --profile "$PROFILE" --thinking on > "$GATE_OUT" 2>&1; then
        GATE_STATUS="PASS"
    else
        GATE_STATUS="FAIL"
    fi
    cat "$GATE_OUT" >> "$LOG"
    say "  gate: $GATE_STATUS (the full report is in results/keepsets/*/GATE.md and $LOG)"
    say "$(grep -E "Verdict" "$GATE_OUT" | tail -1 | sed 's/^/  /')"
fi

./stop.sh >/dev/null 2>&1 || true

# --- the record ------------------------------------------------------------
mkdir -p "$REC_DIR"
REC="$REC_DIR/unpack-cache-$STAMP.md"
CACHE_GB="$CACHE_GB" PROFILE="$PROFILE" KEEP="$KEEP" TOKENS="$TOKENS" \
GATE_STATUS="$GATE_STATUS" PROMPT_SUM="$PROMPT_SUM" REC="$REC" \
python3 - "$WORK/off.json" "$WORK/on.json" <<'PY' | tee -a "$LOG"
import json, os, sys, datetime

def load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}

off, on = load(sys.argv[1]), load(sys.argv[2])

def g(r, *keys, default=None):
    d = r
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d

rows = [
    ("prompt tokens", g(off, "usage", "prompt_tokens"), g(on, "usage", "prompt_tokens")),
    ("time to first token, client (s)", off.get("ttft_s"), on.get("ttft_s")),
    ("prefill_s (server)", g(off, "stats", "prefill_s"), g(on, "stats", "prefill_s")),
    ("prefill_tok_s (server)", g(off, "stats", "prefill_tok_s"), g(on, "stats", "prefill_tok_s")),
    ("decode_tok_s", g(off, "stats", "decode_tok_s"), g(on, "stats", "decode_tok_s")),
    ("unpack hits", g(off, "stats", "unpack_hits", default="-"), g(on, "stats", "unpack_hits", default="-")),
    ("unpack misses", g(off, "stats", "unpack_misses", default="-"), g(on, "stats", "unpack_misses", default="-")),
    ("unpack GB saved", g(off, "stats", "unpack_gb_saved", default="-"), g(on, "stats", "unpack_gb_saved", default="-")),
    ("unpack ms saved (estimate)", g(off, "stats", "unpack_ms_saved", default="-"), g(on, "stats", "unpack_ms_saved", default="-")),
    ("cache slots / used", "-",
     f'{g(on, "stats", "unpack_cache_slots", default="-")} / {g(on, "stats", "unpack_cache_used", default="-")}'),
]

cache_gb = os.environ["CACHE_GB"]
out = [
    f"## {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} — prefill unpack cache, "
    f"{os.environ['PROFILE']} at keep {os.environ['KEEP']}",
    "",
    f"`DSV41_PREFILL_UNPACK_CACHE_GB={cache_gb}` against the same configuration with it off. One "
    f"engine load each, one prompt each, the same prompt both times (~{os.environ['TOKENS']} "
    f"tokens, cksum {os.environ['PROMPT_SUM']}, built by `tools/longprefill.py` from the "
    f"checkout's corpus). Run by `tools/verify_prefill_cache.sh`.",
    "",
    "| | cache off | cache on |",
    "|---|---|---|",
]
for name, a, b in rows:
    out.append(f"| {name} | {a if a is not None else '-'} | {b if b is not None else '-'} |")

try:
    a = float(g(off, "stats", "prefill_tok_s") or 0)
    b = float(g(on, "stats", "prefill_tok_s") or 0)
    if a > 0 and b > 0:
        out += ["", f"Prefill throughput {b / a:.3f}x with the cache on "
                    f"({a:.1f} -> {b:.1f} tok/s)."]
except (TypeError, ValueError):
    pass

hits = g(on, "stats", "unpack_hits")
miss = g(on, "stats", "unpack_misses")
if isinstance(hits, int) and isinstance(miss, int) and (hits + miss):
    out += [f"Hit rate {hits / (hits + miss) * 100:.1f} % of expert lookups "
            f"({hits} of {hits + miss})."]

out += ["", f"Generation gate (`tools/gate_profile.py --profile {os.environ['PROFILE']} "
            f"--thinking on`), run against the cache-on load: **{os.environ['GATE_STATUS']}**.", ""]

rec = os.environ["REC"]
with open(rec, "a") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
print(f"written to {rec}")
PY

say ""
say "log: $LOG"
say "record: $REC"
