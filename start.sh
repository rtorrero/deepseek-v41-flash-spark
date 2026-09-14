#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh -- serve DeepSeek-V4.1-Flash on one DGX Spark.
#
#   ./start.sh                 # start, wait for /health, print the endpoint
#   ./start.sh --no-wait       # start and return immediately (tail logs yourself)
#   ./start.sh --print-env     # resolve the configuration, print it, start nothing
#   PORT=8001 ./start.sh       # environment beats .env
#   ARENA_GB=60 ./start.sh     # pin the resident expert arena instead of auto
#   PRUNE_KEEP=auto ./start.sh # size the keep-set for MAX_SEQ (the recommendation)
#
# This is not SGLang and not a container: it launches `server/app.py --engine v41`
# (engine/v41_engine.py) with nohup, writes logs/server.pid and logs to
# logs/server.log. Stop it with ./stop.sh.
#
# Why the health wait is 20 minutes: the warm start fills the resident FP4
# expert arena from the checkpoint, which is ~80 GB of NVMe reads before the
# HTTP socket is even bound. A server that is "not up yet" at minute 6 is normal.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

err()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "--- $*"; }

WAIT=true
PRINT_ENV=false
for arg in "$@"; do
    case "$arg" in
        --no-wait) WAIT=false ;;
        # A dry run: resolve everything .env and the environment imply -- including
        # PRUNE_KEEP=auto, which is the reason to want this -- print it, and stop.
        # It touches no model, no port and no memory, so it answers on any box.
        --print-env) PRINT_ENV=true ;;
        -h|--help) sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "unknown argument '$arg' (only --no-wait, --print-env)" ;;
    esac
done

# Environment wins over .env, so `PORT=8001 ./start.sh` works. Held in one
# variable per name rather than an associative array: those are bash 4, and the
# macOS system bash is 3.2, which made `./start.sh --print-env` -- the one thing
# in here worth running away from the box -- die on line 42 everywhere else.
_CLI_KEYS=""
for v in MODEL_DIR PYTHON SERVED_MODEL_NAME HOST PORT MAX_SEQ ARENA_GB \
         TRACE_STATS EXPERT_PROFILE EXPERT_TOPICS DEFAULT_THINKING DEFAULT_EFFORT SPEC EXTRA_FLAGS PRUNE_KEEP PRUNE_SELECT TRANSIENT_SLOTS KEEP_FREE_GB \
         EXPERT_FORMAT DSV41_PRUNE_RANK DSV41_PRUNE_SOURCE DSV41_PRUNE_MODE DSV41_DENSE_FP4 DSV41_HEAD_FMT \
         DSV41_ESCAPE_K DSV41_ESCAPE_MARGIN \
         DSV41_PREFILL_CHUNK DSV41_PREFILL_KV_FP8 DSV41_RING \
         DSV41_MTP_WEIGHTS DSV41_RECORD_DRAFT_DATA \
         DSV41_PREFILL_ATTN_GEMM DSV41_PREFILL_HC_GEMM DSV41_PREFILL_FP8_DEQUANT DSV41_PREFILL_FUSED_SINKHORN DSV41_PREFILL_UNPACK_CACHE_GB; do
    if [[ -n "${!v:-}" ]]; then
        _CLI_KEYS="$_CLI_KEYS $v"
        eval "_CLI_$v=\$$v"
    fi
done
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
for v in $_CLI_KEYS; do eval "$v=\$_CLI_$v"; done

MODEL_DIR="${MODEL_DIR:-./models/DeepSeek-V4.1-Flash}"
PYTHON="${PYTHON:-python3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4.1-flash}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_SEQ="${MAX_SEQ:-32768}"
ARENA_GB="${ARENA_GB:-}"                 # empty = size from free GPU memory
# Empty = auto: take the newest results/trace-*/stats/coverage.json (see below).
TRACE_STATS="${TRACE_STATS:-}"
DEFAULT_THINKING="${DEFAULT_THINKING:-off}"
DEFAULT_EFFORT="${DEFAULT_EFFORT:-75}"
SPEC="${SPEC:-1}"
# Whitespace-separated extra flags for server/app.py (A/B runs). Empty by default.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1200}"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"

# --- sanity ---------------------------------------------------------------
PYTHON_BIN="$(command -v -- "$PYTHON" 2>/dev/null || true)"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || err "interpreter not found or not executable: $PYTHON (set PYTHON in .env)"
PYTHON="$PYTHON_BIN"
# Everything below this point needs the checkout to be a real one. --print-env
# is a question about the configuration, not about the model, so it skips them.
if [[ "$PRINT_ENV" == false ]]; then
    [[ -d "$MODEL_DIR" ]] || err "model dir not found: $MODEL_DIR (set MODEL_DIR in .env)"
    [[ -f "$MODEL_DIR/tokenizer.json" ]] || err "$MODEL_DIR has no tokenizer.json"
    [[ -f "$MODEL_DIR/encoding/encoding.py" ]] || err "$MODEL_DIR has no encoding/encoding.py"
    [[ -f server/app.py ]] || err "server/app.py missing -- run this from a full checkout"
fi
case "$DEFAULT_THINKING" in on|off) ;; *) err "DEFAULT_THINKING must be on|off (got '$DEFAULT_THINKING')" ;; esac
case "$SPEC" in 0|1) ;; *) err "SPEC must be 0|1 (got '$SPEC')" ;; esac

# --- PRUNE_KEEP=auto: size the keep-set for the context ---------------------
# The keep fraction has to be chosen against MAX_SEQ, not on its own: the caches
# are allocated for the whole context up front and a prefill chunk costs more
# behind a longer one. 0.36 held a FILLED 256k context; 0.40 serves short
# prompts and was killed by the memory watchdog 582 s into a 195k-token prefill
# (RESULTS.md 2026-09-13 22:50 and the 2026-09-14 addenda). `auto` asks
# tools/keep_for_context.py -- the same arithmetic ./tune.sh budgets with, no
# torch and no model load -- for the largest keep step that fits, and the value
# is exported so the engine, and anything else this shell starts, sees it.
if [[ "${PRUNE_KEEP:-}" == "auto" ]]; then
    keep_args=(--max-seq "$MAX_SEQ" --format "${EXPERT_FORMAT:-cb3}")
    [[ -n "${ARENA_GB:-}" ]] && keep_args+=(--arena-gb "$ARENA_GB")
    [[ -n "${TRANSIENT_SLOTS:-}" ]] && keep_args+=(--transient-slots "$TRANSIENT_SLOTS")
    [[ -n "${KEEP_FREE_GB:-}" ]] && keep_args+=(--keep-free-gb "$KEEP_FREE_GB")
    # Warnings (a keep below anything that has been gated, a pinned arena that
    # does not fit) go to stderr from the tool itself and are left visible.
    if ! KEEP_LINE="$("$PYTHON" tools/keep_for_context.py "${keep_args[@]}")"; then
        [[ -n "$KEEP_LINE" ]] && echo "$KEEP_LINE" >&2
        err "PRUNE_KEEP=auto does not resolve to anything this box can serve at MAX_SEQ=$MAX_SEQ.
     Shorten MAX_SEQ, free memory, or set a number in .env and accept the risk."
    fi
    echo "$KEEP_LINE"
    PRUNE_KEEP="${KEEP_LINE#*-> }"
    PRUNE_KEEP="${PRUNE_KEEP%% *}"
    # A resolved keep with an unpinned arena is only half an answer. The engine's
    # own automatic sizing takes 82 % of what is free, which on a 121 GiB box is
    # about 89.6 GB -- more arena than a filled context can afford, and the way
    # that fails is the memory watchdog on the first long prefill, not a refusal
    # at load. Size it to the kept set instead, which is what ./tune.sh writes.
    # keep_args carries no --arena-gb here, by definition of this branch.
    if [[ -z "${ARENA_GB:-}" ]]; then
        ARENA_GB="$("$PYTHON" tools/keep_for_context.py "${keep_args[@]}" \
                    --keep "$PRUNE_KEEP" --arena 2>/dev/null)" || true
        [[ -n "$ARENA_GB" ]] && info "ARENA_GB unset; sized for keep $PRUNE_KEEP: $ARENA_GB GB"
    fi
fi
if [[ -n "${PRUNE_KEEP:-}" ]]; then
    case "$PRUNE_KEEP" in
        ''|*[!0-9.]*|*.*.*) err "PRUNE_KEEP must be a number or 'auto' (got '$PRUNE_KEEP')" ;;
    esac
    export PRUNE_KEEP
fi

# --- guard: is the port already taken? ------------------------------------
port_busy() {
    if command -v ss >/dev/null 2>&1; then
        [[ -n "$(ss -ltnH "sport = :$PORT" 2>/dev/null)" ]]
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null
    fi
}
if [[ "$PRINT_ENV" == false ]] && port_busy; then
    holder=""
    command -v ss >/dev/null 2>&1 && holder="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | tr -s ' ')"
    err "port $PORT is already in use${holder:+ by: $holder}.
     If it is our own server, ./stop.sh. Otherwise pick another PORT."
fi

# --- guard: does something else own the box's memory? ---------------------
# MemAvailable is the "available" column of `free -g`. This model needs the
# unified pool essentially to itself: the resident expert arena is sized from
# what is free at load time, so starting next to another server does not OOM,
# it silently gives us a tiny arena and a NVMe-bound 1 tok/s server -- or wedges
# the driver with no OOM and no logs. Refuse instead.
if [[ "$PRINT_ENV" == true ]]; then
    :                                    # a dry run holds no memory
elif [[ -r /proc/meminfo ]]; then
    avail_gib=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
    if (( avail_gib < MIN_FREE_GIB )); then
        echo "ERROR: only ${avail_gib} GiB available (need >= ${MIN_FREE_GIB} GiB)." >&2
        echo "     Biggest resident processes:" >&2
        ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -6 |
            awk 'NR==1{print "       PID      RSS_GB  COMMAND"; next} {printf "       %-8s %-7.1f %s\n", $1, $2/1048576, $3}' >&2
        if command -v docker >/dev/null 2>&1 && [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
            echo "     Containers are running and may be holding the unified pool:" >&2
            docker ps --format '       {{.Names}}\t{{.Image}}' 2>/dev/null >&2
            echo "     Stop the one that owns the GPU:  docker stop <container>" >&2
        else
            echo "     Stop whatever holds the pool (another inference server, a container)" >&2
            echo "     and wait for MemAvailable to recover." >&2
        fi
        exit 1
    fi
else
    echo "WARNING: no /proc/meminfo -- skipping the memory guard (not a Linux box?)" >&2
fi

# --- guard: are we already running? ---------------------------------------
if [[ "$PRINT_ENV" == false && -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    err "server already running (pid $(cat "$PID_FILE")). ./stop.sh first."
fi

# --- assemble the command -------------------------------------------------
FLAGS=(
    --model-dir "$MODEL_DIR"
    --host "$HOST" --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --default-thinking "$DEFAULT_THINKING"
    --default-effort "$DEFAULT_EFFORT"
    --engine v41
    --max-seq "$MAX_SEQ"
)
[[ -n "$ARENA_GB" ]] && FLAGS+=(--arena-gb "$ARENA_GB")
[[ "$SPEC" == "0" ]] && FLAGS+=(--no-spec)
# Pruned all-resident mode (RESULTS.md v0.2.0-wip): PRUNE_KEEP=0.31 keeps the top 31 % experts per
# layer routable and resident; pair it with ARENA_GB=90.5 TRANSIENT_SLOTS=16 KEEP_FREE_GB=10 on a
# 128 GB box. Unset = the full model with expert streaming.
EK="{"
[[ -n "${PRUNE_KEEP:-}" ]] && EK="$EK\"prune_keep\": $PRUNE_KEEP,"
# EXPERT_FORMAT=cb3 packs the resident arena into the 3-bit per-row codebook format (14.45 MB per
# expert instead of 18.80), so the same 90.5 GB holds ~40.8 % of all routed experts instead of
# 31.3 %; pair it with PRUNE_KEEP=0.40. Warm start pays the packing (see NOTES 2026-09-11).
[[ -n "${EXPERT_FORMAT:-}" ]] && EK="$EK\"expert_format\": \"$EXPERT_FORMAT\","
[[ -n "${EXPERT_TOPICS:-}" ]] && EK="$EK\"expert_topics\": \"$EXPERT_TOPICS\","
[[ -n "${PRUNE_SELECT:-}" ]] && EK="$EK\"prune_select\": \"$PRUNE_SELECT\","
[[ -n "${TRANSIENT_SLOTS:-}" ]] && EK="$EK\"transient_slots\": $TRANSIENT_SLOTS,"
[[ -n "${KEEP_FREE_GB:-}" ]] && EK="$EK\"keep_free_gb\": $KEEP_FREE_GB,"
EK="${EK%,}}"
[[ "$EK" != "{}" ]] && FLAGS+=(--engine-kwargs "$EK")

# Which coverage.json ranks the warm start. Without one the arena is filled in
# (layer, expert) index order, which is a measurably worse hot set. Trace
# directories carry a name and a date (results/trace-full-YYYYMMDD/), so when
# --- EXPERT_PROFILE: which keep-set the router is restricted to ---------------
# The resident expert set is a fixed budget (PRUNE_KEEP of 384 per layer), so every domain it
# covers competes for the same slots. A profile is one ranking of that budget, built from a trace
# corpus of the workloads it is meant to serve; keepsets/<name>/coverage.json is the whole artefact.
# A specialised profile beats the general one ON ITS OWN DOMAINS and degrades outside them --
# keepsets/<name>/GATE.md records exactly which domains were measured and which failed. Read it
# before choosing one. EXPERT_PROFILE is ignored when TRACE_STATS is set explicitly.
resolve_profile() {
    local p="$1"
    [[ -z "$p" ]] && return
    local f="results/keepsets/$p/coverage.json"
    if [[ -f "$f" ]]; then echo "$f"; return; fi
    info "EXPERT_PROFILE=$p has no results/keepsets/$p/coverage.json; available: $(ls -1 results/keepsets 2>/dev/null | tr '\n' ' ')"
}
if [[ -z "$TRACE_STATS" && -n "${EXPERT_PROFILE:-}" ]]; then
    TRACE_STATS="$(resolve_profile "$EXPERT_PROFILE")"
    [[ -n "$TRACE_STATS" ]] && info "expert profile '$EXPERT_PROFILE' -> $TRACE_STATS"
fi

# TRACE_STATS is unset -- or points at something that is not there -- take the
# newest results/trace-*/stats/coverage.json rather than nothing.
# `|| true` on every call: with no trace in the checkout the last command in here
# is a failed test, which under `set -e` took an assignment's exit status with it
# and killed the launcher before it printed why.
newest_trace_stats() {
    local c
    c=$(ls -1d results/trace-*/stats/coverage.json 2>/dev/null | sort | tail -1 || true)
    [[ -n "$c" ]] && echo "$c"
}
TRACE_USED=""
if [[ -n "$TRACE_STATS" ]]; then
    if [[ -f "$TRACE_STATS" ]]; then
        TRACE_USED="$TRACE_STATS"
    else
        TRACE_USED="$(newest_trace_stats)" || true
        [[ -n "$TRACE_USED" ]] && info "trace stats $TRACE_STATS missing; using $TRACE_USED instead"
    fi
else
    TRACE_USED="$(newest_trace_stats)" || true
fi
[[ -z "$TRACE_USED" ]] && info "no results/trace-*/stats/coverage.json -- warm start will use index order"
[[ -n "$TRACE_USED" ]] && FLAGS+=(--trace-stats "$TRACE_USED")
# shellcheck disable=SC2206
[[ -n "$EXTRA_FLAGS" ]] && FLAGS+=($EXTRA_FLAGS)

# --- the dry run ----------------------------------------------------------
# Everything resolved, nothing started. The settings first, in the same form
# .env writes them, then the command that would have run -- which is where an
# engine-kwargs typo or a trace that fell back to another file shows up.
if [[ "$PRINT_ENV" == true ]]; then
    for v in MODEL_DIR SERVED_MODEL_NAME HOST PORT MAX_SEQ ARENA_GB PRUNE_KEEP PRUNE_SELECT \
             EXPERT_FORMAT EXPERT_TOPICS TRANSIENT_SLOTS KEEP_FREE_GB EXPERT_PROFILE \
             DEFAULT_THINKING DEFAULT_EFFORT SPEC DSV41_PRUNE_RANK DSV41_PRUNE_SOURCE \
             DSV41_PRUNE_MODE DSV41_DENSE_FP4 DSV41_HEAD_FMT EXTRA_FLAGS; do
        printf '%s=%s\n' "$v" "${!v:-}"
    done
    printf 'TRACE_STATS=%s\n' "$TRACE_USED"
    # The one thing above that is assembled rather than read, and the one worth
    # reading back: a key spelled wrong here is accepted silently by the engine.
    printf 'ENGINE_KWARGS=%s\n' "$EK"
    printf 'command: %s server/app.py' "$PYTHON"
    printf ' %q' "${FLAGS[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$LOG_DIR"
info "model=$MODEL_DIR  max_seq=$MAX_SEQ  arena=${ARENA_GB:-auto}  spec=$SPEC  thinking=$DEFAULT_THINKING/$DEFAULT_EFFORT  trace=${TRACE_USED:-none}"
info "log: $LOG_FILE"

: > "$LOG_FILE"
nohup "$PYTHON" server/app.py "${FLAGS[@]}" >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
info "started pid $SERVER_PID"

if [[ "$WAIT" == false ]]; then
    echo "not waiting (--no-wait). Watch it come up with: tail -f $LOG_FILE"
    exit 0
fi

# --- wait for /health -----------------------------------------------------
# The socket is bound only after the engine is constructed, so this loop is
# mostly watching an 80 GB warm start, not an HTTP handshake.
echo -n "waiting for http://$HOST:$PORT/health (up to $((HEALTH_TIMEOUT_S / 60)) min, warm start reads ~80 GB from NVMe): "
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
while :; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "--- last 30 log lines ---" >&2
        tail -n 30 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        err "server exited during startup (see $LOG_FILE)"
    fi
    if curl -sf -o /dev/null "http://$HOST:$PORT/health"; then
        echo " up"
        break
    fi
    if (( $(date +%s) >= deadline )); then
        echo
        echo "--- last 30 log lines ---" >&2
        tail -n 30 "$LOG_FILE" >&2
        err "no /health after ${HEALTH_TIMEOUT_S}s. It may still be warming up: watch $LOG_FILE, or ./stop.sh."
    fi
    echo -n "."
    sleep 5
done

curl -sf "http://$HOST:$PORT/health" && echo
cat <<MSG

  endpoint  http://$HOST:$PORT/v1
  model     $SERVED_MODEL_NAME
  logs      $LOG_FILE
  bench     python3 bench/bench.py --workload prose --runs 3 --out results/prose.json
  stop      ./stop.sh
MSG
