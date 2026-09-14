#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_mtp.sh -- serve a fine-tuned DSpark head and find out whether it helped.
#
#   tools/verify_mtp.sh weights/mtp_finetuned.safetensors
#   tools/verify_mtp.sh                      # the shipped head: the baseline row
#   PROFILE=writing tools/verify_mtp.sh weights/mtp_finetuned.safetensors
#
# The trainer reports an acceptance PROXY -- top-1 agreement on held-out recorded data. This script
# reports the number that decides anything: `accept_len_mean` and tok/s out of the engine's own
# stats, on the working point the recipe ships (the Frontend keep-set at PRUNE_KEEP=0.36), for one
# prose prompt and one markup prompt. Those two are the whole question: prose is what the fine-tune
# is FOR (2.46-2.64 accepted tokens a step, RESULTS.md 4.3) and markup is what it must not lose
# (5.07). A head that lifts the first and drops the second is not an improvement.
#
# Then the two generation gates, because acceptance is not quality: a drafter is verified by the
# target, so a worse drafter cannot corrupt the output -- but a head that has drifted can still make
# the server slower than the one it replaced, and the gate is where a regression in what the server
# WRITES would show up. Frontend is the profile the server is served on; Writing is the prose-heavy
# one the fine-tune is aimed at.
#
# Everything lands in results/mtp/<stamp>/ and the server log in /tmp/mtp.log.
#
# This stops and restarts the server. It cannot run beside anything else that holds the unified
# memory pool -- see the guard in start.sh.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

WEIGHTS="${1:-}"
PROFILE="${PROFILE:-frontend}"
KEEP="${PRUNE_KEEP:-0.36}"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}/v1"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/mtp/$STAMP"
LOG="/tmp/mtp.log"

if [[ -n "$WEIGHTS" && ! -f "$WEIGHTS" ]]; then
    echo "ERROR: no such weights file: $WEIGHTS" >&2
    exit 1
fi

mkdir -p "$OUT"
echo "--- head:    ${WEIGHTS:-<shipped>}"
echo "--- profile: $PROFILE, keep $KEEP"
echo "--- out:     $OUT"
echo "--- log:     $LOG"

# --- serve -----------------------------------------------------------------
./stop.sh || true
: > "$LOG"
env_args=(EXPERT_PROFILE="$PROFILE" PRUNE_KEEP="$KEEP")
[[ -n "$WEIGHTS" ]] && env_args+=(DSV41_MTP_WEIGHTS="$WEIGHTS")
echo "--- starting: ${env_args[*]} ./start.sh"
env "${env_args[@]}" ./start.sh 2>&1 | tee -a "$LOG"

# --- the two prompts -------------------------------------------------------
# One prose, one markup, both with thinking off so the measurement is the drafter and not the
# length of a think block. Temperature 0: acceptance at temperature 0 is the greedy verify path,
# which is the one the lean decode step takes and the one every number in RESULTS.md 4.3 was taken
# on.
ask() {   # ask <name> <max_tokens> <prompt>
    local name="$1" maxtok="$2" prompt="$3"
    echo "--- $name"
    python3 - "$URL" "$name" "$maxtok" "$OUT" "$prompt" <<'PY' | tee -a "$LOG"
import json, sys, time, urllib.request
url, name, maxtok, out, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}],
        "stream": False, "chat_template_kwargs": {"thinking": False},
        "max_tokens": maxtok, "temperature": 0.0}
req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=3600) as r:
    d = json.loads(r.read())
st = d.get("x_engine_stats") or {}
usage = d.get("usage") or {}
row = {"prompt": name, "accept_len_mean": st.get("accept_len_mean"),
       "decode_tok_s": st.get("decode_tok_s"), "prefill_tok_s": st.get("prefill_tok_s"),
       "completion_tokens": usage.get("completion_tokens"), "steps": st.get("steps"),
       "expert_hit_rate": st.get("expert_hit_rate"), "wall_s": round(time.perf_counter() - t0, 1)}
with open(f"{out}/{name}.json", "w") as f:
    json.dump({"row": row, "stats": st, "answer": (d.get("choices") or [{}])[0].get("message", {})}, f, indent=1)
print(f"{name:<8} accept_len_mean {row['accept_len_mean']}  {row['decode_tok_s']} tok/s  "
      f"{row['completion_tokens']} tokens in {row['wall_s']}s")
PY
}

ask prose 400 "Write four paragraphs of a quiet short story about a night train arriving in a town the narrator left as a child. No headings, no dialogue tags in the first paragraph."
ask markup 700 "Write a complete single-file HTML page with inline CSS and JavaScript: a tic-tac-toe board, a status line, a reset button, and a win check. Reply with the file only."

# --- the gates -------------------------------------------------------------
# Long: each is a suite of free generations with thinking on, and an hour a profile is normal.
for p in "$PROFILE" writing; do
    echo "--- gate: $p (thinking on)"
    python3 tools/gate_profile.py --profile "$p" --thinking on --url "$URL" \
        --out "$OUT/GATE-$p.md" 2>&1 | tee -a "$LOG" || true
done

# --- the record ------------------------------------------------------------
{
    echo "# DSpark head verification -- $STAMP"
    echo
    echo "head: ${WEIGHTS:-shipped (checkpoint mtp.*)}"
    echo "serving: EXPERT_PROFILE=$PROFILE PRUNE_KEEP=$KEEP, thinking off for the two prompts,"
    echo "thinking on for the gates."
    echo
    echo "| prompt | accept_len_mean | tok/s | tokens |"
    echo "|---|---|---|---|"
    for n in prose markup; do
        python3 - "$OUT/$n.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["row"]
print(f"| {r['prompt']} | {r['accept_len_mean']} | {r['decode_tok_s']} | {r['completion_tokens']} |")
PY
    done
    echo
    echo "Gates: see GATE-*.md beside this file."
    [[ -n "$WEIGHTS" && -f "${WEIGHTS%.safetensors}.json" ]] && {
        echo
        echo "Training manifest:"
        echo '```json'
        cat "${WEIGHTS%.safetensors}.json"
        echo '```'
    }
} > "$OUT/RUN.md"

echo
cat "$OUT/RUN.md"
echo "--- recorded in $OUT"
