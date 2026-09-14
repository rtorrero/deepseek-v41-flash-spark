#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_prefill_hc.sh -- what do prefill's two attention GEMMs cost, and what does moving them
# onto the tensor cores buy and break?
#
#   ./tools/verify_prefill_hc.sh                 # the whole thing, ~1-2 h
#   SKIP_GATE=1 ./tools/verify_prefill_hc.sh     # the three speed runs + the audit only, ~25 min
#   SKIP_AUDIT=1 ./tools/verify_prefill_hc.sh    # no profiler run
#   GATE_MODE=bf16 ./tools/verify_prefill_hc.sh  # gate a mode other than the recommended one
#
# Run it on the box, from the checkout, with the interpreter the server uses (PYTHON in .env).
# It starts and stops the server itself -- one engine at a time, which this box requires. It never
# runs two engines at once and never leaves one running.
#
# The finding it exists to settle (docs/gemm-dispatch.md, "The two fp32 attention GEMMs"): one
# 2,048-token prefill chunk spends 559.5 ms in `cutlass_80_simt_sgemm_128x32_8x5_tn_align1` over
# 672 calls and 295.7 ms in the `_nn_` variant over another 672 -- 855 ms, more than both FP4 MoE
# kernels together (483 + 410). Those 1,344 calls are the score product and the PV product of
# `engine/model.py::Model._softmax_attn`: 2,048 / ATTN_TILE(64) = 32 tiles, two products a tile,
# 21 encoder layers, 32 x 21 = 672 of each. SIMT sgemm means fp32, and fp32 on this hardware never
# reaches a tensor core.
#
# What it does, in order:
#   1. one identical ~7,000-token prompt through the server in each of the three modes
#      (DSV41_PREFILL_ATTN_GEMM=fp32 | tf32 | bf16), reading prefill_tok_s and TTFT out of
#      x_engine_stats -- fp32 first, because it is the baseline every other number is a ratio of
#   2. tools/audit_gemm_dispatch.py --phase prefill with the server STOPPED, in fp32 and in tf32:
#      the sgemm rows must shrink and tensor-core kernel names must appear, or the flag never took
#   3. tools/gate_profile.py --profile Frontend --thinking on on GATE_MODE -- because a speed
#      number cannot accept either non-default mode on its own. Both of them are far above the
#      1e-7 jitter this engine's chunk-invariance argument is built on, and the thing that goes
#      wrong is a flipped router top-k in a long prompt, which only generation shows.
#   4. a record appended to results/prefill/RECORDS.md; the whole transcript in /tmp/prefill_hc.log
#
# It is a measurement, not a test: it never asserts, and it exits 0 whatever the numbers say.
# Read the record. The gate output goes to results/prefill/, NOT to the profile's shipped GATE.md:
# those files are the record of the shipped configuration and an experimental math type has no
# business in them.
# ---------------------------------------------------------------------------
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

LOG="${PREFILL_HC_LOG:-/tmp/prefill_hc.log}"
OUT_DIR="results/prefill"
RECORDS="$OUT_DIR/RECORDS.md"
mkdir -p "$OUT_DIR"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

# The shipped configuration, and nothing else moves between the three runs.
PROFILE="${PROFILE:-frontend}"
PRUNE_KEEP="${PRUNE_KEEP:-0.36}"
ARENA_GB="${ARENA_GB:-81}"
MAX_SEQ="${MAX_SEQ:-262144}"
EXPERT_FORMAT="${EXPERT_FORMAT:-cb3}"
TRANSIENT_SLOTS="${TRANSIENT_SLOTS:-8}"
KEEP_FREE_GB="${KEEP_FREE_GB:-6}"
PRUNE_RANK="${DSV41_PRUNE_RANK:-maxmin}"
PRUNE_SOURCE="${DSV41_PRUNE_SOURCE:-saliency}"
PREFILL_CHUNK="${DSV41_PREFILL_CHUNK:-2048}"
URL="${DSV41_URL:-http://127.0.0.1:8000/v1}"
EFFORT="${EFFORT:-45}"
MODES="${MODES:-fp32 tf32 bf16}"
GATE_MODE="${GATE_MODE:-tf32}"
SKIP_GATE="${SKIP_GATE:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
PROMPT_CHARS="${PROMPT_CHARS:-28000}"   # ~7,000 tokens of the tech report's prose
PROBE_TOKENS="${PROBE_TOKENS:-64}"      # the prompt is the measurement; the answer only has to start
PROMPT_FILE="$OUT_DIR/prompt.txt"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && -f .env ]]; then
    PYTHON="$(sed -n 's/^[[:space:]]*PYTHON=//p' .env | tail -1 | tr -d '"'"'"'')"
fi
PYTHON="${PYTHON:-python3}"
command -v -- "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: no interpreter '$PYTHON'"; exit 1; }

info() { echo; echo "=== $* ==="; }
info "verify_prefill_hc.sh  $(date -u '+%Y-%m-%d %H:%M:%SZ')  python $($PYTHON -V 2>&1)"
echo "profile=$PROFILE keep=$PRUNE_KEEP arena=$ARENA_GB format=$EXPERT_FORMAT chunk=$PREFILL_CHUNK"
echo "modes=$MODES  gate mode=$GATE_MODE  log=$LOG"

# --- the prompt -----------------------------------------------------------
# One file, written once and reused by all three runs, so "identical prompt" is a property of the
# bytes and not of a generator. Prose from the corpus the keep-sets were ranked on: the router's
# choices decide how many experts a chunk touches, and that is most of what a prefill chunk costs,
# so a prompt of repeated filler would not exercise the same path.
build_prompt() {
    "$PYTHON" - "$PROMPT_FILE" "$PROMPT_CHARS" <<'PY'
import sys
out, budget = sys.argv[1], int(sys.argv[2])
src = ["corpus/sources/dsv41_techreport.txt", "corpus/sources/reasoning_design.txt",
       "corpus/sources/dsv41_readme.md"]
buf = []
n = 0
for p in src:
    try:
        t = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    buf.append(t)
    n += len(t)
    if n >= budget:
        break
body = "".join(buf)[:budget]
head = ("Read the following material carefully, then answer in one short paragraph: what is the "
        "single largest cost of processing a long prompt on a memory-bound single-GPU box, and "
        "why?\n\n---\n\n")
open(out, "w", encoding="utf-8").write(head + body)
print(f"prompt: {out}  {len(head) + len(body)} chars")
PY
}
build_prompt

# --- one prompt through the server, prefill numbers out of x_engine_stats --
probe() {
    "$PYTHON" - "$URL" "$PROMPT_FILE" "$PROBE_TOKENS" <<'PY'
import json, sys, time, urllib.request
base, path, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompt = open(path, encoding="utf-8").read()
body = {"model": "default", "messages": [{"role": "user", "content": prompt}], "stream": True,
        "chat_template_kwargs": {"thinking": False}, "max_tokens": ntok, "temperature": 0,
        "stream_options": {"include_usage": True}}
try:
    with urllib.request.urlopen(base + "/models", timeout=30) as r:
        body["model"] = json.loads(r.read())["data"][0]["id"]
except Exception:
    pass
req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
ttft = None
usage, stats = {}, {}
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except ValueError:
            continue
        ch = (d.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
            ttft = time.perf_counter() - t0          # first token out of the door, TTFT as a client sees it
        usage = d.get("usage") or usage
        stats = d.get("x_engine_stats") or stats
wall = time.perf_counter() - t0
print("PROBE " + json.dumps({
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "ttft_s": round(ttft, 3) if ttft else None,
    "wall_s": round(wall, 2),
    "prefill_s": stats.get("prefill_s"),
    "prefill_tok_s": stats.get("prefill_tok_s"),
    "decode_tok_s": stats.get("decode_tok_s"),
    "accept_len_mean": stats.get("accept_len_mean"),
}))
PY
}

start_engine() {
    local mode="$1"
    ./stop.sh || true
    info "starting server with DSV41_PREFILL_ATTN_GEMM=$mode"
    EXPERT_PROFILE="$PROFILE" PRUNE_KEEP="$PRUNE_KEEP" ARENA_GB="$ARENA_GB" MAX_SEQ="$MAX_SEQ" \
    EXPERT_FORMAT="$EXPERT_FORMAT" TRANSIENT_SLOTS="$TRANSIENT_SLOTS" KEEP_FREE_GB="$KEEP_FREE_GB" \
    DSV41_PRUNE_RANK="$PRUNE_RANK" DSV41_PRUNE_SOURCE="$PRUNE_SOURCE" \
    DSV41_PREFILL_CHUNK="$PREFILL_CHUNK" DSV41_PREFILL_ATTN_GEMM="$mode" \
    DEFAULT_THINKING=off DEFAULT_EFFORT="$EFFORT" \
        ./start.sh
    local rc=$?
    if (( rc != 0 )); then
        echo "ERROR: the server did not come up with DSV41_PREFILL_ATTN_GEMM=$mode (rc $rc)"
        echo "--- last 40 lines of logs/server.log ---"
        tail -n 40 logs/server.log 2>/dev/null
        return 1
    fi
    curl -sf "${URL%/v1}/health" | "$PYTHON" -c \
        'import json,sys;c=json.load(sys.stdin).get("engine_config",{});print("engine_config:",json.dumps({k:c.get(k) for k in ("prune_keep","expert_format","arena_gb","max_seq","prefill_chunk","swa_replay","expert_topics")}))'
}

# --- 1: the three speed runs ---------------------------------------------
declare -A PROBES
for mode in $MODES; do
    if start_engine "$mode"; then
        info "prefill probe, mode $mode"
        PROBES[$mode]="$(probe | tail -1)"
        echo "${PROBES[$mode]}"
    else
        echo "mode $mode did not start; continuing so the record is not empty"
        PROBES[$mode]=""
    fi
done
./stop.sh || true

# --- 2: the profiler, server stopped -------------------------------------
# The audit builds its own engine, so nothing else may hold the box: two ~89 GB engine loads at
# once wedge it (docs/gotchas.md). Hence ./stop.sh above and the check here.
AUDIT_FILES=""
if [[ "$SKIP_AUDIT" != "1" ]]; then
    for mode in fp32 tf32; do
        info "kernel audit, prefill only, mode $mode (server stopped)"
        out="$OUT_DIR/gemm_dispatch-$mode.json"
        EXPERT_PROFILE="$PROFILE" PRUNE_KEEP="$PRUNE_KEEP" ARENA_GB="$ARENA_GB" MAX_SEQ="$MAX_SEQ" \
        EXPERT_FORMAT="$EXPERT_FORMAT" TRANSIENT_SLOTS="$TRANSIENT_SLOTS" KEEP_FREE_GB="$KEEP_FREE_GB" \
        DSV41_PREFILL_ATTN_GEMM="$mode" \
            "$PYTHON" tools/audit_gemm_dispatch.py --phase prefill --chunk "$PREFILL_CHUNK" --json "$out"
        echo "audit $mode exit $?"
        AUDIT_FILES="$AUDIT_FILES $out"
    done
    info "what the two audits say about the sgemm rows"
    "$PYTHON" - $AUDIT_FILES <<'PY'
import json, sys
def rows(p):
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"  {p}: unreadable ({e})")
        return []
    ph = d.get("prefill") or d.get("phases", {}).get("prefill") or d
    return ph.get("kernels", [])
for p in sys.argv[1:]:
    ks = rows(p)
    simt = [k for k in ks if "simt" in k["name"].lower() or "sgemm" in k["name"].lower()]
    tc = [k for k in ks if any(s in k["name"].lower() for s in ("tensorop", "s1688", "s16816", "hmma", "bf16"))]
    print(f"\n  {p}")
    print(f"    SIMT / sgemm rows: {len(simt)}, {sum(k['ms'] for k in simt):.1f} ms, "
          f"{sum(k['calls'] for k in simt)} calls")
    for k in simt[:4]:
        print(f"      {k['ms']:9.2f} ms  {k['calls']:7d}  {k['name']}")
    print(f"    tensor-core rows:  {len(tc)}, {sum(k['ms'] for k in tc):.1f} ms, "
          f"{sum(k['calls'] for k in tc)} calls")
    for k in tc[:4]:
        print(f"      {k['ms']:9.2f} ms  {k['calls']:7d}  {k['name']}")
print("\n  Reading: in fp32 the two 672-call sgemm rows must be there. In tf32 they must shrink or")
print("  disappear and a tensor-core name must appear with roughly those call counts. If the fp32")
print("  and tf32 tables are the same, the flag never reached cuBLAS and every speed number above")
print("  is fp32 measured three times.")
PY
else
    echo "SKIP_AUDIT=1, the profiler was not run"
fi

# --- 3: the generation gate on the recommended mode -----------------------
GATE_OUT=""
if [[ "$SKIP_GATE" != "1" ]]; then
    if start_engine "$GATE_MODE"; then
        GATE_OUT="$OUT_DIR/GATE-frontend-$GATE_MODE.md"
        info "generation gate: Frontend, thinking on, mode $GATE_MODE"
        "$PYTHON" tools/gate_profile.py --profile Frontend --thinking on --effort "$EFFORT" \
            --url "$URL" --out "$GATE_OUT"
        echo "gate exit $?"
    fi
    ./stop.sh || true
else
    echo "SKIP_GATE=1, no gate was run -- so nothing here accepts a non-default mode"
fi

# --- 4: the record --------------------------------------------------------
info "record -> $RECORDS"
{
    echo
    echo "## $(date -u '+%Y-%m-%d %H:%M')Z — prefill attention GEMM math type, $PROFILE at keep $PRUNE_KEEP"
    echo
    echo "\`DSV41_PREFILL_ATTN_GEMM\` over \`$MODES\`, one identical prompt (\`$PROMPT_FILE\`,"
    echo "$PROMPT_CHARS chars), \`EXPERT_PROFILE=$PROFILE PRUNE_KEEP=$PRUNE_KEEP ARENA_GB=$ARENA_GB\`,"
    echo "\`EXPERT_FORMAT=$EXPERT_FORMAT DSV41_PREFILL_CHUNK=$PREFILL_CHUNK MAX_SEQ=$MAX_SEQ\`,"
    echo "thinking off for the probes and on for the gate, effort $EFFORT. One prompt per mode, no sweep."
    echo
    "$PYTHON" - "${PROBES[fp32]:-}" "${PROBES[tf32]:-}" "${PROBES[bf16]:-}" <<'PY'
import json, sys
def load(s):
    s = (s or "").strip()
    return json.loads(s[len("PROBE "):]) if s.startswith("PROBE ") else {}
fp32, tf32, bf16 = (load(a) for a in sys.argv[1:4])
print("| | fp32 (shipped) | tf32 | bf16 |")
print("|---|---|---|---|")
rows = [("prompt tokens", "prompt_tokens"), ("**prefill tok/s**", "prefill_tok_s"),
        ("prefill s", "prefill_s"), ("**TTFT s**", "ttft_s"), ("wall s", "wall_s"),
        ("decode tok/s", "decode_tok_s"), ("DSpark acceptance", "accept_len_mean")]
for label, key in rows:
    vals = [d.get(key) for d in (fp32, tf32, bf16)]
    if all(v is None for v in vals):
        continue
    print("| " + label + " | " + " | ".join("-" if v is None else str(v) for v in vals) + " |")
base = fp32.get("prefill_tok_s")
if base:
    r = [("1.000x" if d is fp32 else (f"{d['prefill_tok_s'] / base:.3f}x"
                                      if d.get("prefill_tok_s") else "-"))
         for d in (fp32, tf32, bf16)]
    print("| **prefill tok/s against fp32** | " + " | ".join(r) + " |")
if fp32.get("prompt_tokens") and tf32.get("prompt_tokens") and fp32["prompt_tokens"] != tf32["prompt_tokens"]:
    print("\n**The prompt was not identical across the runs** — the token counts differ, so the")
    print("rows above are not comparable. Rerun.")
PY
    echo
    if [[ -n "$AUDIT_FILES" ]]; then
        echo "Kernel audits:$AUDIT_FILES — the fp32 table must show the two 672-call"
        echo "\`cutlass_*_simt_sgemm_*_{tn,nn}_align1\` rows; the tf32 table must not."
    else
        echo "No kernel audit was run (SKIP_AUDIT=1)."
    fi
    echo
    if [[ -n "$GATE_OUT" ]]; then
        echo "Generation gate on \`$GATE_MODE\`: \`$GATE_OUT\` — compare against"
        echo "\`results/keepsets/frontend/GATE.md\` for the shipped engine on the same prompts. A"
        echo "non-default mode is accepted by that comparison, not by the tok/s above."
    else
        echo "**No gate was run, so no non-default mode is accepted by this record.** The speed"
        echo "numbers stand on their own and mean nothing about what the model writes."
    fi
    echo
    echo "Transcript: \`$LOG\`."
} >> "$RECORDS"

tail -n 50 "$RECORDS"
info "done. record appended to $RECORDS, transcript in $LOG"
exit 0
