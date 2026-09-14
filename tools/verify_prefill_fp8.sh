#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_prefill_fp8.sh -- does the fp8 prefill gather actually pay, and does a
# 4,096-token chunk hold?
#
# Three runs, one identical long prompt each, the shipped .env untouched except
# for the two variables under test:
#
#   (a) baseline     DSV41_PREFILL_KV_FP8=0  DSV41_PREFILL_CHUNK=2048
#   (b) fp8          DSV41_PREFILL_KV_FP8=1  DSV41_PREFILL_CHUNK=2048
#   (c) fp8 + chunk  DSV41_PREFILL_KV_FP8=1  DSV41_PREFILL_CHUNK=4096
#
# (a) vs (b) is the memory question with nothing else moving. (b) vs (c) is the
# speed question: a prefill chunk touches every resident expert of every layer
# once, so halving the number of chunks halves the unpack passes -- and that is
# only reachable if (b) freed enough room for the bigger chunk to fit.
#
# What is recorded per run: prefill tok/s and prefill seconds from
# `x_engine_stats`, TTFT measured at the client as the wall clock to the first
# streamed content byte, and the LOW-WATER MARK of MemAvailable, sampled from
# /proc/meminfo five times a second for the whole request. That last one is the
# point of the exercise: the prefill reserve in tools/budget.py is a fit to
# exactly this measurement, and this is how it gets replaced by a better one.
#
# Then, once, on (c): tools/gate_profile.py --profile Frontend --thinking on.
# Memory is not the only thing fp8 changes -- the rounding lands in an attention
# score and from there in the residual stream, which is where the window KV the
# decoder layers write, and decode then reads, comes from. The gate is the only
# instrument here that can see that, and it is free generation rather than a
# perplexity number for the reasons in tools/gate_profile.py's own docstring.
# Its verdict is appended under results/prefill/, NOT to the keep-set's own
# GATE.md: that file is the record of the shipped configuration, and this is not
# the shipped configuration.
#
#   ./tools/verify_prefill_fp8.sh              # all three runs, then the gate
#   RUNS=a,b ./tools/verify_prefill_fp8.sh     # a subset
#   GATE=0 ./tools/verify_prefill_fp8.sh       # skip the gate
#
#   PROMPT_CHARS    characters of corpus text in the prompt (28000, ~7k tokens)
#   EXPERT_PROFILE  keep-set                                (frontend)
#   PRUNE_KEEP      keep fraction                           (0.36)
#   MAX_TOKENS      generated per run, enough to time TTFT  (48)
#   LOG             transcript                              (/tmp/prefill_fp8.log)
#   OUT             where the JSON lands                    (results/prefill)
#
# Runs on the serving box only: it starts and stops the server three times, each
# start being a full warm start. Budget about an hour, plus the gate.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

LOG="${LOG:-/tmp/prefill_fp8.log}"
OUT="${OUT:-$ROOT/results/prefill}"
RUNS="${RUNS:-a,b,c}"
GATE="${GATE:-1}"
PROMPT_CHARS="${PROMPT_CHARS:-28000}"
MAX_TOKENS="${MAX_TOKENS:-48}"
STAMP="$(date +%Y%m%d-%H%M%S)"
PROMPT_FILE=""
CLIENT=""
trap 'rm -f "$PROMPT_FILE" "$CLIENT" 2>/dev/null || true' EXIT

# The shipped .env is the baseline; only the keep-set and the two variables
# under test are forced, so anything else a box has tuned still applies.
[[ -f .env ]] && { set -a; . ./.env; set +a; }
export EXPERT_PROFILE="${EXPERT_PROFILE:-frontend}"
export PRUNE_KEEP="${PRUNE_KEEP:-0.36}"
PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MAX_SEQ="${MAX_SEQ:-32768}"
URL="http://$HOST:$PORT/v1"

mkdir -p "$OUT"
: > "$LOG"
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

[[ -r /proc/meminfo ]] || { echo "this script measures /proc/meminfo; run it on the serving box" >&2; exit 2; }

# --- the prompt -------------------------------------------------------------
# One fixed slice of text that is already in the checkout, so all three runs
# prefill the same tokens and a tok/s comparison means something. The exact
# token count is whatever the tokenizer makes of it, reported per run from
# `usage` rather than assumed here.
PROMPT_FILE="$(mktemp /tmp/prefill_prompt.XXXXXX)"
head -c "$PROMPT_CHARS" corpus/sources/dsv41_techreport.txt > "$PROMPT_FILE"
[[ -s "$PROMPT_FILE" ]] || { echo "no corpus text at corpus/sources/dsv41_techreport.txt" >&2; exit 2; }

# --- the client -------------------------------------------------------------
# Standard library only, like server/app.py and bench/bench.py. It streams, so
# TTFT is the wall clock to the first content byte and not to the whole
# response, and it reads x_engine_stats out of the final chunk.
CLIENT="$(mktemp /tmp/prefill_client.XXXXXX.py)"
cat > "$CLIENT" <<'PY'
import json, sys, time, urllib.request

url, prompt_file, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = open(prompt_file, encoding="utf-8", errors="replace").read()
body = json.dumps({
    "model": "default",
    "messages": [{"role": "user",
                  "content": text + "\n\nIn one sentence: what is this document about?"}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}).encode()
req = urllib.request.Request(url + "/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
ttft, stats, usage = None, {}, {}
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        stats = d.get("x_engine_stats") or stats
        usage = d.get("usage") or usage
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = time.perf_counter() - t0
print(json.dumps({"ttft_s": round(ttft, 3) if ttft else None,
                  "wall_s": round(time.perf_counter() - t0, 3),
                  "usage": usage, "stats": stats}))
PY

# --- MemAvailable low-water mark -------------------------------------------
# Five samples a second for the whole request. The engine's own watchdog reads
# the same field and kills the process below 2.5 GB, so this is the margin that
# decides whether a configuration serves at all.
poll_mem() {
    while :; do
        awk '/^MemAvailable:/ {print $2}' /proc/meminfo >> "$1"
        sleep 0.2
    done
}

wait_health() {
    local deadline=$(( SECONDS + ${HEALTH_TIMEOUT_S:-1800} ))
    while (( SECONDS < deadline )); do
        curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 5
    done
    return 1
}

run_one() {
    local tag="$1" fp8="$2" chunk="$3"
    say "--- run $tag: DSV41_PREFILL_KV_FP8=$fp8 DSV41_PREFILL_CHUNK=$chunk"

    ./stop.sh >>"$LOG" 2>&1 || true
    # The ring must stay longer than window_size + chunk; engine/model.py
    # derives its default from the chunk, so nothing is pinned here on purpose.
    unset DSV41_RING || true
    export DSV41_PREFILL_KV_FP8="$fp8" DSV41_PREFILL_CHUNK="$chunk"

    if ! ./start.sh >>"$LOG" 2>&1; then
        say "    start.sh failed -- see $LOG"
        printf '{"run":"%s","prefill_kv_fp8":%s,"prefill_chunk":%s,"error":"start failed"}\n' \
            "$tag" "$([[ $fp8 == 1 ]] && echo true || echo false)" "$chunk" > "$OUT/$STAMP-$tag.json"
        return 1
    fi
    wait_health || { say "    /health never came up"; return 1; }
    grep -E "prefill chunk .* gathered KV|arena .* expert slots" logs/server.log | tail -3 \
        | tee -a "$LOG" || true

    local memfile
    memfile="$(mktemp /tmp/prefill_mem.XXXXXX)"
    poll_mem "$memfile" &
    local poller=$!
    local result
    result="$("$PYTHON" "$CLIENT" "$URL" "$PROMPT_FILE" "$MAX_TOKENS")" || result='{"error":"request failed"}'
    kill "$poller" 2>/dev/null || true
    wait "$poller" 2>/dev/null || true

    local minmem
    minmem="$(sort -n "$memfile" | head -1)"
    rm -f "$memfile"

    "$PYTHON" - "$OUT/$STAMP-$tag.json" "$tag" "$fp8" "$chunk" "${minmem:-0}" "$result" <<'PY' | tee -a "$LOG"
import json, os, sys
path, tag, fp8, chunk, minmem, result = sys.argv[1:7]
r = json.loads(result)
st, us = r.get("stats") or {}, r.get("usage") or {}
out = {
    "run": tag,
    "prefill_kv_fp8": fp8 == "1",
    "prefill_chunk": int(chunk),
    "prompt_tokens": us.get("prompt_tokens") or st.get("prompt_tokens"),
    "prefill_s": st.get("prefill_s"),
    "prefill_tok_s": st.get("prefill_tok_s"),
    "ttft_s": r.get("ttft_s"),
    "decode_tok_s": st.get("decode_tok_s"),
    "mem_available_min_gb": round(int(minmem) * 1024 / 1e9, 2) if minmem not in ("0", "") else None,
    "nvme_gb": st.get("nvme_gb"),
    "expert_hit_rate": st.get("expert_hit_rate"),
    # read back from /health rather than trusted from the environment: a
    # variable that did not reach the engine is exactly the failure this
    # comparison would otherwise report as "fp8 changes nothing"
    "engine_prefill_chunk": st.get("prefill_chunk"),
    "engine_prefill_kv_fp8": st.get("prefill_kv_fp8"),
    "engine_ring": st.get("ring"),
    "error": r.get("error"),
}
# what the cost model said this configuration would need, so the row carries the
# prediction next to the measurement instead of leaving them in two files
try:
    sys.path.insert(0, "tools")
    import budget as B
    out["predicted_reserve_gb"] = round(
        B.prefill_bytes(int(os.environ.get("MAX_SEQ", 32768)), int(chunk), fp8 == "1") / B.GB, 2)
except Exception:
    out["predicted_reserve_gb"] = None
json.dump(out, open(path, "w"), indent=2)
print(f"    prompt {out['prompt_tokens']} tok | prefill {out['prefill_s']} s = "
      f"{out['prefill_tok_s']} tok/s | TTFT {out['ttft_s']} s | "
      f"MemAvailable min {out['mem_available_min_gb']} GB "
      f"(model said {out['predicted_reserve_gb']} GB would be needed) | "
      f"engine says chunk={out['engine_prefill_chunk']} fp8={out['engine_prefill_kv_fp8']} "
      f"ring={out['engine_ring']}")
PY
}

# --- the three runs ---------------------------------------------------------
say "logging to $LOG, results to $OUT/$STAMP-*.json"
say "profile $EXPERT_PROFILE, keep $PRUNE_KEEP, MAX_SEQ $MAX_SEQ"
say "prompt: $(wc -c < "$PROMPT_FILE") characters of corpus text"
LAST=""
case ",$RUNS," in *",a,"*) { run_one a 0 2048 && LAST=a; } || true ;; esac
case ",$RUNS," in *",b,"*) { run_one b 1 2048 && LAST=b; } || true ;; esac
case ",$RUNS," in *",c,"*) { run_one c 1 4096 && LAST=c; } || true ;; esac

# --- the comparison ---------------------------------------------------------
"$PYTHON" - "$OUT" "$STAMP" <<'PY' | tee -a "$LOG"
import json, os, sys
out, stamp = sys.argv[1], sys.argv[2]
rows = []
for tag in ("a", "b", "c"):
    p = os.path.join(out, f"{stamp}-{tag}.json")
    if os.path.exists(p):
        rows.append(json.load(open(p)))
if not rows:
    sys.exit(0)
hdr = ("run", "fp8", "chunk", "prompt", "prefill s", "tok/s", "TTFT s", "MemAvail min GB")
print()
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for r in rows:
    print("| " + " | ".join(str(r.get(k)) for k in (
        "run", "prefill_kv_fp8", "prefill_chunk", "prompt_tokens",
        "prefill_s", "prefill_tok_s", "ttft_s", "mem_available_min_gb")) + " |")
base = next((r for r in rows if r["run"] == "a"), None)
for r in rows:
    if base and r is not base and r.get("prefill_tok_s") and base.get("prefill_tok_s"):
        dm = (r.get("mem_available_min_gb") or 0) - (base.get("mem_available_min_gb") or 0)
        print(f"  {r['run']} vs a: prefill {r['prefill_tok_s'] / base['prefill_tok_s']:.2f}x, "
              f"MemAvailable floor {dm:+.2f} GB")
json.dump(rows, open(os.path.join(out, f"{stamp}-summary.json"), "w"), indent=2)
print(f"\nwritten: {os.path.join(out, stamp + '-summary.json')}")
PY

# --- the quality gate, once, on the last configuration ----------------------
if [[ "$GATE" == "1" && "$LAST" == "c" ]]; then
    say "--- gate: profile Frontend, thinking on, on run c (its server is still up)"
    "$PYTHON" tools/gate_profile.py --profile Frontend --thinking on --url "$URL" \
        --out "$OUT/$STAMP-GATE.md" 2>&1 | tee -a "$LOG" | tail -40
    say "gate recorded in $OUT/$STAMP-GATE.md"
elif [[ "$GATE" == "1" ]]; then
    say "gate skipped: run c did not complete, and the gate belongs to the configuration it judges"
fi

say "done. transcript $LOG, results $OUT"
