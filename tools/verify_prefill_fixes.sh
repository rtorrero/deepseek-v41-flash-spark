#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_prefill_fixes.sh -- do the two prefill fixes of docs/gemm-dispatch.md
# move prefill on this box, and by how much?
#
# Four runs on the SHIPPED .env (Frontend, keep 0.36), one identical ~7,000-token
# prompt each, nothing else changed between them:
#
#   1  both off                                                  the baseline
#   2  Fix A   DSV41_PREFILL_FP8_DEQUANT=fused
#   3  Fix B   DSV41_PREFILL_FUSED_SINKHORN=1
#   4  both    DSV41_PREFILL_FP8_DEQUANT=fused DSV41_PREFILL_FUSED_SINKHORN=1
#
# Fix A replaces FP8Weight.dequant() -- a full [N, K] fp32 scale table built by
# two repeat_interleaves plus a full fp32 weight, ten transient bytes per weight
# per layer per chunk -- with one Triton kernel that reads the fp8 codes and
# writes bf16. Same bits, two transient bytes. Fix B runs prefill's 20-iteration
# Sinkhorn on the fused kernel decode has always used instead of the torch port
# in 16-row tiles: 698,880 launches per 2,048-token chunk become 42. Both are
# off by default and off is the engine that shipped.
#
# What each run reports, from the server's own numbers: TTFT (client-side, to
# the first streamed token) and `x_engine_stats.prefill_tok_s` / `prefill_s`.
# Three timed requests after a discarded warm one -- the first request of a load
# compiles the Triton kernels -- and the median is the row.
#
# Runs 1 and 4 additionally get tools/audit_gemm_dispatch.py, with the server
# STOPPED: it builds an engine of its own, and two ~89 GB arenas at once wedge
# this box past the point where sshd can fork (docs/gotchas.md). That is where
# the launch count and the GPU-busy fraction come from -- the two numbers a
# top-20-by-GPU-time kernel table cannot show, and the only ones that can
# confirm Fix B did anything at all.
#
# Run 4 then takes tools/gate_profile.py --profile Frontend --thinking on once,
# so the run that would ship has a generation record and not only a stopwatch.
#
#   ./tools/verify_prefill_fixes.sh            # all four, in order, then the gate
#   ./tools/verify_prefill_fixes.sh 1 4        # only those runs (resume)
#   ./tools/verify_prefill_fixes.sh --no-gate  # skip the generation gate
#   ./tools/verify_prefill_fixes.sh --list     # what it would do, no HTTP
#
# Summary and full output in /tmp/prefill_fixes.log; the record, including the
# prompt itself and the raw per-request JSON, goes to results/prefill/.
#
# Budget a warm start per run (the arena is streamed from NVMe), a few minutes
# of requests, and an hour or two for the gate. Hold the box: it serves one
# engine at a time.
#
# Honours: PYTHON (default ./.venv/bin/python, else python3), PORT (from .env,
# else 8000), PROMPT_TOKENS (7000), REPEATS (3), MAX_TOKENS_PROBE (16), and
# EFFORT / MAX_TOKENS for the gate (gate_profile's own defaults when unset).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=/tmp/prefill_fixes.log
OUTDIR=results/prefill
PROMPT="$OUTDIR/prompt.txt"
PROMPT_TOKENS="${PROMPT_TOKENS:-7000}"
REPEATS="${REPEATS:-3}"
MAX_TOKENS_PROBE="${MAX_TOKENS_PROBE:-16}"

err() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "=== $*"; echo "=== $*" >> "$LOG"; }

[[ -f tools/audit_gemm_dispatch.py ]] || err "run this from the checkout (tools/audit_gemm_dispatch.py not found)"

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

# A value for either switch in .env would beat the exported one: start.sh sources
# .env after capturing its own whitelist, and neither of these is on it.
if [[ -f .env ]] && grep -qE '^[[:space:]]*DSV41_PREFILL_(FP8_DEQUANT|FUSED_SINKHORN)=' .env; then
    err ".env sets DSV41_PREFILL_FP8_DEQUANT or DSV41_PREFILL_FUSED_SINKHORN and would override \
this script; comment those lines out and run again"
fi

#            1      2        3      4
RUN_NAME=(-  "both off" "Fix A (fused fp8 dequant)" "Fix B (fused Sinkhorn)" "both on")
RUN_FP8=(-   off    fused    off    fused)
RUN_HC=(-    0      0        1      1)
RUN_AUDIT=(- 1      0        0      1)
RUN_GATE=(-  0      0        0      1)

LIST=false
GATE=true
WANT=()
for a in "$@"; do
    case "$a" in
        --list)    LIST=true ;;
        --no-gate) GATE=false ;;
        [1-4])     WANT+=("$a") ;;
        *)         err "arguments are run numbers 1-4, --no-gate, or --list (not '$a')" ;;
    esac
done
[[ ${#WANT[@]} -eq 0 ]] && WANT=(1 2 3 4)

mkdir -p "$OUTDIR"

# --------------------------------------------------------------- the prompt
# Written once and reused by every run, so "identical prompt" is a fact about a
# file and not a claim about a generator. Real prose: the router's choices decide
# how many distinct experts a chunk touches, and that is what the MoE path costs.
make_prompt() {
    [[ -s "$PROMPT" ]] && return 0
    MODEL_DIR="${MODEL_DIR:-}" TARGET="$PROMPT_TOKENS" OUT="$PROMPT" "$PYTHON" - <<'PY'
import os, re
para = (
    "The engine streams three-bit routed experts out of an arena on the device, unpacks the ones a "
    "chunk touches back into packed FP4, and runs the grouped kernel over them. Attention keeps a "
    "sliding window of 128 positions plus every completed compressed group, selected by an indexer "
    "that scores in fixed blocks along the compressed-key axis. The hyper-connection coefficients "
    "are balanced by a Sinkhorn iteration over a four-by-four matrix per token, and the dense "
    "projections are read in their stored eight-bit format rather than dequantised into memory. "
    "A prefill chunk therefore costs bytes of format conversion that a decode step does not, and "
    "the question this prompt exists to ask is how much of that cost is arithmetic and how much of "
    "it is the shape of the loop around it. "
)
target = int(os.environ["TARGET"])
md = os.environ.get("MODEL_DIR") or ""
enc = None
for cand in (md, os.path.expanduser("~/models/DeepSeek-V4.1-Flash"), "./models/DeepSeek-V4.1-Flash"):
    p = os.path.join(cand, "tokenizer.json") if cand else ""
    if p and os.path.exists(p):
        try:
            from tokenizers import Tokenizer
            enc = Tokenizer.from_file(p)
        except Exception:
            enc = None
        break

def ntok(s):
    if enc is not None:
        return len(enc.encode(s, add_special_tokens=False).ids)
    return max(1, len(s) // 4)          # the usual ratio; the server reports the true count anyway

text = ""
while ntok(text) < target:
    text += para
# trim back to the target on sentence boundaries, so the prompt ends as prose
parts = re.split(r"(?<=\. )", text)
while len(parts) > 1 and ntok("".join(parts)) > target:
    parts.pop()
text = "".join(parts).strip()
open(os.environ["OUT"], "w").write(text + "\n")
print(f"wrote {os.environ['OUT']}: {ntok(text)} tokens "
      f"({'checkpoint tokenizer' if enc is not None else 'estimated at 4 chars/token'}), "
      f"{len(text)} characters")
PY
}

# ------------------------------------------------------- one run's requests
# Streamed, so TTFT is the time to the first token and not the time to the last;
# x_engine_stats and usage ride the final chunk (bench/bench.py says the same).
probe() {
    local tag="$1" json="$2"
    URLV="$URL" PROMPT="$PROMPT" TAG="$tag" JSON="$json" REPEATS="$REPEATS" \
    MAXTOK="$MAX_TOKENS_PROBE" APIKEY="${DSV41_API_KEY:-}" "$PYTHON" - <<'PY'
import json as J, os, statistics, time, urllib.request

url = os.environ["URLV"].rstrip("/") + "/chat/completions"
prompt = open(os.environ["PROMPT"]).read()
reps = int(os.environ["REPEATS"])
rows = []


def one():
    body = J.dumps({"model": "default", "stream": True,
                    "max_tokens": int(os.environ["MAXTOK"]), "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}]}).encode()
    hdr = {"Content-Type": "application/json"}
    if os.environ.get("APIKEY"):
        hdr["Authorization"] = "Bearer " + os.environ["APIKEY"]
    t0 = time.perf_counter()
    ttft = None
    stats = usage = None
    with urllib.request.urlopen(urllib.request.Request(url, body, hdr), timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = J.loads(line[6:])
            if obj.get("x_engine_stats"):
                stats = obj["x_engine_stats"]
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if ttft is None and (d.get("content") or d.get("reasoning_content")):
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {"ttft_s": ttft if ttft is not None else total, "total_s": total,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "prefill_s": (stats or {}).get("prefill_s"),
            "prefill_tok_s": (stats or {}).get("prefill_tok_s"),
            "x_engine_stats": stats}


warm = one()
print(f"    warm-up (discarded: the first request of a load compiles the Triton kernels)  "
      f"ttft {warm['ttft_s']:.2f} s  prefill {warm.get('prefill_tok_s')} tok/s")
for i in range(reps):
    r = one()
    rows.append(r)
    print(f"    request {i + 1}  prompt {r['prompt_tokens']} tok  ttft {r['ttft_s']:8.2f} s  "
          f"prefill {r['prefill_s']} s = {r['prefill_tok_s']} tok/s")


def med(k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return round(statistics.median(v), 3) if v else None


out = {"tag": os.environ["TAG"], "warm_up": warm, "runs": rows,
       "median": {"ttft_s": med("ttft_s"), "prefill_s": med("prefill_s"),
                  "prefill_tok_s": med("prefill_tok_s")},
       "prompt_tokens": rows[0]["prompt_tokens"] if rows else None,
       "prompt_chars": len(prompt)}
J.dump(out, open(os.environ["JSON"], "w"), indent=1)
m = out["median"]
print(f"    MEDIAN  ttft {m['ttft_s']} s   prefill {m['prefill_s']} s = {m['prefill_tok_s']} tok/s")
PY
}

wait_for_memory() {
    for _ in $(seq 1 60); do
        avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 999)
        [ "$avail" -ge 90 ] && return 0
        sleep 2
    done
}

one_run() {
    local id="$1"
    local name="${RUN_NAME[$id]}" fp8="${RUN_FP8[$id]}" hc="${RUN_HC[$id]}"
    local label="run $id: $name  (DSV41_PREFILL_FP8_DEQUANT=$fp8 DSV41_PREFILL_FUSED_SINKHORN=$hc)"
    local probe_json="$OUTDIR/run$id-probe.json"
    local audit_json="$OUTDIR/run$id-audit.json"

    if $LIST; then
        echo "$label"
        echo "    prompt   $PROMPT (~$PROMPT_TOKENS tokens, written once, shared by every run)"
        echo "    serve    ./start.sh with the shipped .env plus those two variables"
        echo "    probe    $REPEATS streamed requests at max_tokens=$MAX_TOKENS_PROBE -> $probe_json"
        [ "${RUN_AUDIT[$id]}" = 1 ] && \
            echo "    audit    (server stopped) tools/audit_gemm_dispatch.py --phase prefill -> $audit_json"
        [ "${RUN_GATE[$id]}" = 1 ] && $GATE && \
            echo "    gate     tools/gate_profile.py --profile Frontend --thinking on"
        return 0
    fi

    say "$label  ($(date '+%Y-%m-%d %H:%M:%S'))"
    ./stop.sh >>"$LOG" 2>&1 || true
    wait_for_memory

    (
        export DSV41_PREFILL_FP8_DEQUANT="$fp8"
        export DSV41_PREFILL_FUSED_SINKHORN="$hc"

        ./start.sh >>"$LOG" 2>&1 \
            || err "the server did not come up for $label (see logs/server.log)"
        curl -sf "http://127.0.0.1:$PORT/health" >>"$LOG" 2>&1 || true
        echo >> "$LOG"

        probe "$name" "$probe_json" 2>&1 | tee -a "$LOG"

        if [ "${RUN_GATE[$id]}" = 1 ] && $GATE; then
            say "run $id: generation gate, Frontend, thinking on"
            # The gate exits nonzero whenever any prompt failed, which is a normal
            # outcome here, so its status is recorded and not acted on.
            local rc=0
            "$PYTHON" tools/gate_profile.py --profile Frontend --thinking on --url "$URL" \
                --out results/keepsets/frontend/GATE.md \
                ${EFFORT:+--effort $EFFORT} ${MAX_TOKENS:+--max-tokens $MAX_TOKENS} \
                2>&1 | tee -a "$LOG" | tail -30 || rc=$?
            echo "    gate exit $rc (nonzero = at least one prompt failed; read the counts)" | tee -a "$LOG"
        fi
    )

    ./stop.sh >>"$LOG" 2>&1 || true

    if [ "${RUN_AUDIT[$id]}" = 1 ]; then
        wait_for_memory
        say "run $id: kernel/launch audit, server stopped (it builds an engine of its own)"
        (
            # The audit builds its own engine, so it needs what start.sh would have
            # read: the keep-set, the arena, the topics. Sourced first, then the two
            # switches exported over it, so this script decides them and .env does not.
            [ -f .env ] && { set -a; . ./.env; set +a; } || true
            export DSV41_PREFILL_FP8_DEQUANT="$fp8"
            export DSV41_PREFILL_FUSED_SINKHORN="$hc"
            "$PYTHON" tools/audit_gemm_dispatch.py --phase prefill --json "$audit_json" \
                2>&1 | tee -a "$LOG" | tail -40
        ) || say "run $id: the audit failed; the probe numbers above still stand"
        wait_for_memory
    fi
}

if ! $LIST; then
    make_prompt | tee -a "$LOG"
    {
        echo
        echo "############################################################"
        echo "# prefill fixes: gemm-dispatch Fix A (fp8 dequant) and Fix B (fused Sinkhorn)"
        echo "# shipped .env, ~$PROMPT_TOKENS-token prompt, $REPEATS timed requests per run"
        echo "# started $(date '+%Y-%m-%d %H:%M:%S'), runs ${WANT[*]}"
        echo "############################################################"
    } >> "$LOG"
else
    echo "prompt: $PROMPT (written on the first real run)"
fi

for r in "${WANT[@]}"; do
    one_run "$r"
done

if ! $LIST; then
    say "done $(date '+%Y-%m-%d %H:%M:%S'); full output in $LOG, record in $OUTDIR"
    echo
    # One table out of the four probe files, plus the two audits.
    OUTDIR="$OUTDIR" "$PYTHON" - <<'PY' | tee -a "$LOG"
import json, os
d = os.environ["OUTDIR"]
NAMES = {1: "1 both off", 2: "2 Fix A", 3: "3 Fix B", 4: "4 both on"}
print(f"{'run':12s} {'prompt tok':>10s} {'TTFT s':>9s} {'prefill s':>10s} {'prefill tok/s':>14s}")
base = None
for i, nm in NAMES.items():
    p = os.path.join(d, f"run{i}-probe.json")
    if not os.path.exists(p):
        print(f"{nm:12s} {'(not run)':>10s}")
        continue
    m = json.load(open(p))
    med = m["median"]
    tps = med.get("prefill_tok_s")
    if i == 1:
        base = tps
    delta = f"  ({tps / base:.2f}x)" if (base and tps) else ""
    print(f"{nm:12s} {str(m.get('prompt_tokens')):>10s} {med.get('ttft_s'):>9} "
          f"{med.get('prefill_s'):>10} {str(tps):>14s}{delta}")
print()
for i in (1, 4):
    p = os.path.join(d, f"run{i}-audit.json")
    if not os.path.exists(p):
        continue
    a = json.load(open(p)).get("prefill", {})
    print(f"{NAMES[i]:12s} audit: {a.get('launches')} launches, "
          f"{a.get('gpu_busy_frac')} GPU-busy, {a.get('gpu_ms')} ms of kernels "
          f"in {round(a.get('wall_s', 0) * 1e3, 1)} ms of wall")
PY
fi
