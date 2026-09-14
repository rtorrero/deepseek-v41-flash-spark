#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_prefill_topk.sh -- what fewer routed experts in prefill actually buy,
# and what they cost.
#
# `tools/prefill_bound_test.py` already answered the first half in isolation:
# one layer's routed MoE is 54.1 ms at k=6, 38.3 at k=4 and 30.9 at k=3, so the
# prefill MoE is compute-bound and cutting k buys time. What it could not
# answer is what the SERVER does with that -- the MoE kernels are ~893 ms of a
# ~3.7 s chunk, so a 0.57x on them is ~10 % on the chunk, and TTFT has a
# tokenizer, an attention half and a head in it too. And it could not answer
# the only question that decides whether this ships: whether the model still
# writes.
#
# Three runs, one identical ~7,000-token prompt each, the shipped .env
# untouched except for the two variables under test:
#
#   (a) baseline   DSV41_PREFILL_TOPK unset            -- six routed experts
#   (c) three      DSV41_PREFILL_TOPK=3  FOLD=exfold   -- the paper's P3 budget
#   (b) four       DSV41_PREFILL_TOPK=4  FOLD=exfold   -- the middle
#
# Run in the order a, c, b so that (b) is the configuration left standing when
# the gate runs: the gate belongs to the configuration it judges, and a second
# warm start to get back to it costs more than the reordering does.
#
# Then, once, on (b): tools/gate_profile.py --profile Frontend --thinking on.
# This is THE measurement. Prefill precision does not stay in prefill -- the
# window KV the decoder layers read is what the prompt wrote, so a routing
# approximation taken during the prompt is still being read hundreds of tokens
# into the answer. Perplexity cannot see the failure mode that matters here
# (NOTES: an NLL number cannot see degeneration), and free generation can, so
# the gate is free generation. Its verdict is appended under results/prefill/,
# NOT to the keep-set's own GATE.md: that file is the record of the shipped
# configuration, and this is not the shipped configuration.
#
# The tables have to exist first. Once, with the server stopped:
#
#     ./stop.sh && python3 tools/exfold_prepare.py
#
#   ./tools/verify_prefill_topk.sh              # three runs, then the gate
#   RUNS=a,b ./tools/verify_prefill_topk.sh     # a subset
#   GATE=0 ./tools/verify_prefill_topk.sh       # skip the gate
#   FOLD=none ./tools/verify_prefill_topk.sh    # the control arm: drop, do not fold
#
#   PROMPT_CHARS    characters of corpus text in the prompt (28000, ~7k tokens)
#   EXPERT_PROFILE  keep-set                                (frontend)
#   PRUNE_KEEP      keep fraction                           (0.36)
#   MAX_TOKENS      generated per run, enough to time TTFT  (48)
#   FOLD            DSV41_PREFILL_FOLD for runs b and c     (exfold)
#   LOG             transcript                              (/tmp/prefill_topk.log)
#   OUT             where the JSON lands                    (results/prefill)
#
# Runs on the serving box only: it starts and stops the server three times,
# each start being a full warm start. Budget about an hour, plus the gate.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

LOG="${LOG:-/tmp/prefill_topk.log}"
OUT="${OUT:-$ROOT/results/prefill}"
RUNS="${RUNS:-a,c,b}"
GATE="${GATE:-1}"
FOLD="${FOLD:-exfold}"
PROMPT_CHARS="${PROMPT_CHARS:-28000}"
MAX_TOKENS="${MAX_TOKENS:-48}"
STAMP="$(date +%Y%m%d-%H%M%S)"
PROMPT_FILE=""
CLIENT=""
trap 'rm -f "$PROMPT_FILE" "$CLIENT" 2>/dev/null || true' EXIT

# The shipped .env is the baseline; only the keep-set and the variables under
# test are forced, so anything else a box has tuned still applies.
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

[[ -r /proc/meminfo ]] || { echo "this script runs on the serving box" >&2; exit 2; }

# The fold arm cannot run without its tables, and finding that out after the
# first warm start wastes twenty minutes.
TABLE="${DSV41_EXFOLD_TABLE:-${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash}/exfold_table.npz}"
if [[ "$FOLD" == "exfold" && ! -f "$TABLE" ]]; then
    echo "no ExFold tables at $TABLE" >&2
    echo "run, once, with the server stopped:  ./stop.sh && $PYTHON tools/exfold_prepare.py" >&2
    echo "or set FOLD=none to measure the drop arm instead" >&2
    exit 2
fi

# --- the prompt -------------------------------------------------------------
# One fixed slice of text that is already in the checkout, so all three runs
# prefill the same tokens and a tok/s comparison means something.
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
ttft, stats, usage, text_out = None, {}, {}, []
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
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                text_out.append(piece)
                if ttft is None:
                    ttft = time.perf_counter() - t0
print(json.dumps({"ttft_s": round(ttft, 3) if ttft else None,
                  "wall_s": round(time.perf_counter() - t0, 3),
                  "head": "".join(text_out)[:240],
                  "usage": usage, "stats": stats}))
PY

wait_health() {
    local deadline=$(( SECONDS + ${HEALTH_TIMEOUT_S:-1800} ))
    while (( SECONDS < deadline )); do
        curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 5
    done
    return 1
}

run_one() {
    local tag="$1" k="$2"
    if [[ "$k" == "off" ]]; then
        say "--- run $tag: DSV41_PREFILL_TOPK unset (the shipped six)"
        unset DSV41_PREFILL_TOPK DSV41_PREFILL_FOLD || true
    else
        say "--- run $tag: DSV41_PREFILL_TOPK=$k DSV41_PREFILL_FOLD=$FOLD"
        export DSV41_PREFILL_TOPK="$k" DSV41_PREFILL_FOLD="$FOLD"
    fi

    ./stop.sh >>"$LOG" 2>&1 || true
    if ! ./start.sh >>"$LOG" 2>&1; then
        say "    start.sh failed -- see $LOG"
        printf '{"run":"%s","prefill_topk":"%s","error":"start failed"}\n' "$tag" "$k" \
            > "$OUT/$STAMP-topk-$tag.json"
        return 1
    fi
    wait_health || { say "    /health never came up"; return 1; }
    # The line the engine prints when it loads the tables, so the record says how
    # much of the table the keep-set actually covered on the day of the run.
    grep -E "ExFold|prefill top-" logs/server.log | tail -3 | tee -a "$LOG" || true

    local result
    result="$("$PYTHON" "$CLIENT" "$URL" "$PROMPT_FILE" "$MAX_TOKENS")" \
        || result='{"error":"request failed"}'

    "$PYTHON" - "$OUT/$STAMP-topk-$tag.json" "$tag" "$k" "$FOLD" "$result" <<'PY' | tee -a "$LOG"
import json, sys
path, tag, k, fold, result = sys.argv[1:6]
r = json.loads(result)
st, us = r.get("stats") or {}, r.get("usage") or {}
out = {
    "run": tag,
    "prefill_topk": None if k == "off" else int(k),
    "prefill_fold": "off" if k == "off" else fold,
    "prompt_tokens": us.get("prompt_tokens") or st.get("prompt_tokens"),
    "prefill_s": st.get("prefill_s"),
    "prefill_tok_s": st.get("prefill_tok_s"),
    "ttft_s": r.get("ttft_s"),
    "decode_tok_s": st.get("decode_tok_s"),
    # read back from the engine rather than trusted from the environment: a
    # variable that did not reach the engine is exactly the failure this
    # comparison would otherwise report as "fewer experts change nothing"
    "engine_prefill_topk": st.get("prefill_topk"),
    "engine_prefill_fold": st.get("prefill_fold"),
    "engine_routed_topk": st.get("routed_topk"),
    # the first words of the answer, so a run that went to pieces is visible in
    # the table rather than only in the gate an hour later
    "head": r.get("head"),
    "error": r.get("error"),
}
json.dump(out, open(path, "w"), indent=2)
print(f"    prompt {out['prompt_tokens']} tok | prefill {out['prefill_s']} s = "
      f"{out['prefill_tok_s']} tok/s | TTFT {out['ttft_s']} s | "
      f"decode {out['decode_tok_s']} tok/s | engine says "
      f"topk={out['engine_prefill_topk']} fold={out['engine_prefill_fold']}")
print(f"    answer: {(out['head'] or '').strip()[:120]!r}")
PY
}

# --- the three runs ---------------------------------------------------------
say "logging to $LOG, results to $OUT/$STAMP-topk-*.json"
say "profile $EXPERT_PROFILE, keep $PRUNE_KEEP, MAX_SEQ $MAX_SEQ, fold $FOLD"
say "prompt: $(wc -c < "$PROMPT_FILE") characters of corpus text"
[[ "$FOLD" == "exfold" ]] && say "tables: $TABLE"
LAST=""
case ",$RUNS," in *",a,"*) { run_one a off && LAST=a; } || true ;; esac
case ",$RUNS," in *",c,"*) { run_one c 3   && LAST=c; } || true ;; esac
case ",$RUNS," in *",b,"*) { run_one b 4   && LAST=b; } || true ;; esac

# --- the comparison ---------------------------------------------------------
"$PYTHON" - "$OUT" "$STAMP" <<'PY' | tee -a "$LOG"
import json, os, sys
out, stamp = sys.argv[1], sys.argv[2]
rows = []
for tag in ("a", "b", "c"):
    p = os.path.join(out, f"{stamp}-topk-{tag}.json")
    if os.path.exists(p):
        rows.append(json.load(open(p)))
if not rows:
    sys.exit(0)
rows.sort(key=lambda r: -(r.get("prefill_topk") or 99))
hdr = ("run", "prefill k", "fold", "prompt", "prefill s", "prefill tok/s", "TTFT s", "decode tok/s")
print()
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for r in rows:
    print("| " + " | ".join(str(r.get(k)) for k in (
        "run", "prefill_topk", "prefill_fold", "prompt_tokens",
        "prefill_s", "prefill_tok_s", "ttft_s", "decode_tok_s")) + " |")
base = next((r for r in rows if r["run"] == "a"), None)
for r in rows:
    if base and r is not base and r.get("prefill_tok_s") and base.get("prefill_tok_s"):
        line = f"  {r['run']} (k={r['prefill_topk']}) vs a: prefill {r['prefill_tok_s'] / base['prefill_tok_s']:.3f}x"
        if r.get("ttft_s") and base.get("ttft_s"):
            line += f", TTFT {base['ttft_s'] / r['ttft_s']:.3f}x"
        print(line)
# What the bound test predicts, so the measurement lands next to the prediction
# instead of in a different file. The routed MoE is ~893 ms of a ~3.7 s chunk
# (24 %) and scales 1.00 / 0.708 / 0.571 at k = 6 / 4 / 3.
share, scale = 0.24, {6: 1.0, 4: 0.708, 3: 0.571}
print("\n  predicted from tools/prefill_bound_test.py (MoE is ~24 % of a chunk):")
for k, s in scale.items():
    print(f"    k={k}: prefill {1 / (1 - share * (1 - s)):.3f}x")
json.dump(rows, open(os.path.join(out, f"{stamp}-topk-summary.json"), "w"), indent=2)
print(f"\nwritten: {os.path.join(out, stamp + '-topk-summary.json')}")
PY

# --- the quality gate, once, on k=4 -----------------------------------------
if [[ "$GATE" == "1" && "$LAST" == "b" ]]; then
    say "--- gate: profile Frontend, thinking on, on run b (k=4, its server is still up)"
    "$PYTHON" tools/gate_profile.py --profile Frontend --thinking on --url "$URL" \
        --out "$OUT/$STAMP-topk-GATE.md" 2>&1 | tee -a "$LOG" | tail -40
    say "gate recorded in $OUT/$STAMP-topk-GATE.md"
elif [[ "$GATE" == "1" ]]; then
    say "gate skipped: run b did not complete, and the gate belongs to the configuration it judges"
fi

say "done. transcript $LOG, results $OUT"
