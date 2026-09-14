#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_escape.sh -- does the escape hatch change what the model writes, and what does it cost?
#
#   ./tools/verify_escape.sh                    # the whole thing, ~1-3 h
#   ESCAPE_MARGIN=0.05 ./tools/verify_escape.sh # a looser margin
#   SKIP_GATE=1 ./tools/verify_escape.sh        # speed A/B only, ~20 min
#
# Run it on the box, from the checkout, with the interpreter the server uses (PYTHON in .env).
# It starts and stops the server itself -- one engine at a time, which this box requires.
#
# What it does, in order:
#   1. server up on the Frontend keep-set at PRUNE_KEEP=0.36 with DSV41_ESCAPE_K=1, then
#      tools/gate_profile.py --profile Frontend --thinking on, then --profile Backend
#   2. one short prompt through the server, hatch ON: tokens/s from `usage` and wall time
#   3. server restarted with the hatch OFF, same everything else, the same short prompt
#   4. a record appended to results/escape/RECORDS.md; the whole transcript in /tmp/escape.log
#
# It is a measurement, not a test: it never asserts, and it exits 0 whatever the numbers say.
# Read the record. The two things to look for are whether the gate count moved against the
# profile's own GATE.md, and what `escape_ms_each` says a fetch costs -- if that is tens of ms,
# the CB3 repack is the cost and the hatch needs the FP4 side-arena described in
# docs/keep-sets.md before it is worth anything at decode.
#
# The gate output goes to results/escape/, NOT to the profile's shipped GATE.md: those files are
# the record of the shipped configuration and an experimental engine has no business in them.
# ---------------------------------------------------------------------------
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

LOG="${ESCAPE_LOG:-/tmp/escape.log}"
OUT_DIR="results/escape"
RECORDS="$OUT_DIR/RECORDS.md"
mkdir -p "$OUT_DIR"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

# The configuration under test. Everything is overridable from the environment so a second run
# can move one thing at a time.
PROFILE="${PROFILE:-frontend}"
PRUNE_KEEP="${PRUNE_KEEP:-0.36}"
ARENA_GB="${ARENA_GB:-81}"
MAX_SEQ="${MAX_SEQ:-262144}"
EXPERT_FORMAT="${EXPERT_FORMAT:-cb3}"
TRANSIENT_SLOTS="${TRANSIENT_SLOTS:-8}"
KEEP_FREE_GB="${KEEP_FREE_GB:-6}"
PRUNE_RANK="${DSV41_PRUNE_RANK:-maxmin}"
PRUNE_SOURCE="${DSV41_PRUNE_SOURCE:-saliency}"
ESCAPE_K="${ESCAPE_K:-1}"
ESCAPE_MARGIN="${ESCAPE_MARGIN:-0.10}"   # engine/escape.py DEFAULT_MARGIN
URL="${DSV41_URL:-http://127.0.0.1:8000/v1}"
EFFORT="${EFFORT:-45}"
SKIP_GATE="${SKIP_GATE:-0}"
# One short prompt, thinking off, greedy: a decode-rate probe and nothing else. No sweep.
PROBE_PROMPT="${PROBE_PROMPT:-Write a Python function that merges two sorted lists. Code only.}"
PROBE_TOKENS="${PROBE_TOKENS:-200}"

# The interpreter the server runs under, from .env if it says.
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && -f .env ]]; then
    PYTHON="$(sed -n 's/^[[:space:]]*PYTHON=//p' .env | tail -1 | tr -d '"'"'"'')"
fi
PYTHON="${PYTHON:-python3}"
command -v -- "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: no interpreter '$PYTHON'"; exit 1; }

info() { echo; echo "=== $* ==="; }
info "verify_escape.sh  $(date -u '+%Y-%m-%d %H:%M:%SZ')  python $($PYTHON -V 2>&1)"
echo "profile=$PROFILE keep=$PRUNE_KEEP arena=$ARENA_GB format=$EXPERT_FORMAT rank=$PRUNE_RANK source=$PRUNE_SOURCE"
echo "hatch: DSV41_ESCAPE_K=$ESCAPE_K DSV41_ESCAPE_MARGIN=$ESCAPE_MARGIN  transient=$TRANSIENT_SLOTS"
echo "log: $LOG"

# --- the decode probe -----------------------------------------------------
# tokens/s two ways: the server's own decode_tok_s from x_engine_stats, and completion_tokens over
# the wall clock of the request. They differ by TTFT, which is why both are printed.
probe() {
    "$PYTHON" - "$URL" "$PROBE_PROMPT" "$PROBE_TOKENS" <<'PY'
import json, sys, time, urllib.request
base, prompt, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = {"model": "default", "messages": [{"role": "user", "content": prompt}], "stream": False,
        "chat_template_kwargs": {"thinking": False}, "max_tokens": ntok, "temperature": 0}
try:
    with urllib.request.urlopen(base + "/models", timeout=30) as r:
        body["model"] = json.loads(r.read())["data"][0]["id"]
except Exception:
    pass
req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=1800) as r:
    d = json.loads(r.read())
wall = time.perf_counter() - t0
u = d.get("usage") or {}
st = d.get("x_engine_stats") or {}
n = u.get("completion_tokens") or 0
row = {
    "completion_tokens": n,
    "wall_s": round(wall, 2),
    "tok_s_wall": round(n / wall, 2) if wall > 0 else None,
    "decode_tok_s": st.get("decode_tok_s"),
    "prefill_s": st.get("prefill_s"),
    "steps": st.get("steps"),
    "accept_len_mean": st.get("accept_len_mean"),
    "escape_k": st.get("escape_k", 0),
    "escapes": st.get("escapes"),
    "escapes_per_token": st.get("escapes_per_token"),
    "escape_gb": st.get("escape_gb"),
    "escape_ms": st.get("escape_ms"),
    "escape_ms_each": st.get("escape_ms_each"),
    "escape_peak_live": st.get("escape_peak_live"),
    "escape_evictions": st.get("escape_evictions"),
    "escape_blocked_margin": st.get("escape_blocked_margin"),
    "escape_blocked_budget": st.get("escape_blocked_budget"),
}
print("PROBE " + json.dumps(row))
PY
}

# --- start the server in one of the two configurations --------------------
start_engine() {
    local k="$1"
    ./stop.sh || true
    info "starting server with DSV41_ESCAPE_K=$k"
    EXPERT_PROFILE="$PROFILE" PRUNE_KEEP="$PRUNE_KEEP" ARENA_GB="$ARENA_GB" MAX_SEQ="$MAX_SEQ" \
    EXPERT_FORMAT="$EXPERT_FORMAT" TRANSIENT_SLOTS="$TRANSIENT_SLOTS" KEEP_FREE_GB="$KEEP_FREE_GB" \
    DSV41_PRUNE_RANK="$PRUNE_RANK" DSV41_PRUNE_SOURCE="$PRUNE_SOURCE" \
    DSV41_ESCAPE_K="$k" DSV41_ESCAPE_MARGIN="$ESCAPE_MARGIN" \
    DEFAULT_THINKING=off DEFAULT_EFFORT="$EFFORT" \
        ./start.sh
    local rc=$?
    if (( rc != 0 )); then
        echo "ERROR: the server did not come up with DSV41_ESCAPE_K=$k (rc $rc)"
        echo "--- last 40 lines of logs/server.log ---"
        tail -n 40 logs/server.log 2>/dev/null
        return 1
    fi
    # what the engine thinks it is, quoted with every number below
    curl -sf "${URL%/v1}/health" | "$PYTHON" -c \
        'import json,sys;c=json.load(sys.stdin).get("engine_config",{});print("engine_config:",json.dumps({k:c.get(k) for k in ("prune_keep","prune_mode","prune_source","expert_format","arena_gb","transient_slots","max_seq","escape_k","escape_margin","expert_topics")}))'
}

# --- 1 + 2: hatch ON ------------------------------------------------------
ON_PROBE="" ; OFF_PROBE="" ; GATE_SUMMARY=""
if start_engine "$ESCAPE_K"; then
    info "decode probe, hatch ON"
    ON_PROBE="$(probe | tail -1)"
    echo "$ON_PROBE"
    if [[ "$SKIP_GATE" != "1" ]]; then
        for p in Frontend Backend; do
            info "generation gate: $p, thinking on, hatch ON"
            "$PYTHON" tools/gate_profile.py --profile "$p" --thinking on --effort "$EFFORT" \
                --url "$URL" --out "$OUT_DIR/GATE-$(echo "$p" | tr 'A-Z' 'a-z')-escape.md"
            echo "gate $p exit $?"
            GATE_SUMMARY="$GATE_SUMMARY $p:$OUT_DIR/GATE-$(echo "$p" | tr 'A-Z' 'a-z')-escape.md"
        done
    else
        echo "SKIP_GATE=1, gates not run"
    fi
else
    echo "hatch ON phase failed; continuing to the OFF baseline so the record is not empty"
fi

# --- 3: hatch OFF, the same short prompt ----------------------------------
if start_engine 0; then
    info "decode probe, hatch OFF (the shipped engine)"
    OFF_PROBE="$(probe | tail -1)"
    echo "$OFF_PROBE"
fi
./stop.sh || true

# --- 4: the record --------------------------------------------------------
info "record -> $RECORDS"
{
    echo
    echo "## $(date -u '+%Y-%m-%d %H:%M')Z — escape hatch, $PROFILE at keep $PRUNE_KEEP"
    echo
    echo "\`DSV41_ESCAPE_K=$ESCAPE_K DSV41_ESCAPE_MARGIN=$ESCAPE_MARGIN\`, \`EXPERT_PROFILE=$PROFILE\`,"
    echo "\`PRUNE_KEEP=$PRUNE_KEEP ARENA_GB=$ARENA_GB EXPERT_FORMAT=$EXPERT_FORMAT TRANSIENT_SLOTS=$TRANSIENT_SLOTS\`,"
    echo "\`DSV41_PRUNE_RANK=$PRUNE_RANK DSV41_PRUNE_SOURCE=$PRUNE_SOURCE MAX_SEQ=$MAX_SEQ\`, thinking off for the"
    echo "probe and on for the gates, effort $EFFORT. One prompt for the speed probe, no sweep."
    echo
    echo "| | hatch on | hatch off |"
    echo "|---|---|---|"
    "$PYTHON" - "$ON_PROBE" "$OFF_PROBE" <<'PY'
import json, sys
def load(s):
    s = (s or "").strip()
    return json.loads(s[len("PROBE "):]) if s.startswith("PROBE ") else {}
on, off = load(sys.argv[1]), load(sys.argv[2])
rows = [("decode tok/s (engine)", "decode_tok_s"), ("tok/s over the wall clock", "tok_s_wall"),
        ("completion tokens", "completion_tokens"), ("wall s", "wall_s"), ("prefill s", "prefill_s"),
        ("decode steps", "steps"), ("DSpark acceptance", "accept_len_mean"),
        ("escapes", "escapes"), ("escapes per token", "escapes_per_token"),
        ("escape GB read", "escape_gb"), ("escape ms total", "escape_ms"),
        ("**ms per escape**", "escape_ms_each"), ("peak live escapes", "escape_peak_live"),
        ("escapes evicted from the ring", "escape_evictions"),
        ("candidates refused on the margin", "escape_blocked_margin"),
        ("candidates refused on the step budget", "escape_blocked_budget")]
for label, key in rows:
    a, b = on.get(key), off.get(key)
    if a is None and b is None:
        continue
    print(f"| {label} | {'-' if a is None else a} | {'-' if b is None else b} |")
if on.get("decode_tok_s") and off.get("decode_tok_s"):
    r = on["decode_tok_s"] / off["decode_tok_s"]
    print(f"| **hatch on / hatch off** | {r:.3f}x | 1.000x |")
PY
    echo
    if [[ -n "$GATE_SUMMARY" ]]; then
        echo "Gate output:$GATE_SUMMARY — compare each against the profile's own"
        echo "\`results/keepsets/<profile>/GATE.md\` for the shipped engine on the same prompts."
    else
        echo "No gate was run (SKIP_GATE=1 or the armed engine did not start)."
    fi
    echo
    echo "Transcript: \`$LOG\`."
} >> "$RECORDS"

tail -n 40 "$RECORDS"
info "done. record appended to $RECORDS, transcript in $LOG"
exit 0
