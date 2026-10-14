#!/usr/bin/env bash
# Launch the asynchronous FVQA hard-case collection pipeline.
#
# This script starts the four long-running producer processes only.  When the
# desired quota is complete, use fix2's merge-teacher-results, judge-expert,
# blind-visual-review, filter, and ShareGPT conversion commands separately.
#
# Default target endpoint: http://localhost:6657/v1
# Default served model: Qwen3-VL-8B-SFT12-gpt54-only-epoch5
# Default local queue/tool state: /tmp/$USER/fix2_fvqa_<UTC timestamp>
#
# Typical pilot invocation (run from the repository root):
#   bash synthesis/post_process/fix2/launch_fvqa_pipeline.sh
#
# Before production collection, create STYLE_EXAMPLES_FILE from reviewed,
# sanitized anchors. It must not contain the unedited template placeholders.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash synthesis/post_process/fix2/launch_fvqa_pipeline.sh

Useful optional environment (pilot defaults shown):
  BASE_URL              default: http://localhost:6657/v1
  BASE_SERVED_MODEL     default: Qwen3-VL-8B-SFT12-gpt54-only-epoch5
  LOCAL_STATE_DIR       default: /tmp/$USER/fix2_fvqa_<UTC timestamp>
  BASE_EVAL_STATE_DIR   default: LOCAL_STATE_DIR/base_eval; local judge report/detail state
  RUN_DIR               HDFS/shared results root (default: runs/fix2_fvqa_<UTC timestamp>)
  SOURCE_PARQUET        default: data/benchmarks/fvqa/train_raw.parquet
  BASE_LIMIT             default: 100; 0 means all 4,856 FVQA train rows
  EXPAND_SOURCE_MANIFEST default: 0; set 1 to append through BASE_LIMIT when reusing RUN_DIR
  MATERIALIZE_WORKERS    default: 16; parallel source-image materialization workers
  HARD_CAP               default: 10; total independently-wrong cases to enqueue
  BASE_PARALLEL_WORKERS  default: 8
  JUDGE_WORKERS          default: 8
  TEACHER_WORKERS        default: 4
  MAX_READY_TASKS        default: 4 x TEACHER_WORKERS
  TEACHER_MODEL          default: gpt54_2_internal_azure_responses
  TEACHER_API_MODE       default: responses (manual_react|native_tools|responses)
  FIX2_ANSWER_JUDGE_MODEL optional override; default: multimodal_process_global_2
  VISION_JUDGE_MODEL     default: gemini38flash_internal_azure (used during finalize)
  STYLE_EXAMPLES_FILE    required by default; reviewed/sanitized JSON anchors
  REQUIRE_STYLE_EXAMPLES default: 1; set to 0 only for a disposable smoke run
  TEACHER_PRINT_ROUNDS   default: 1; write every expert tool round to teacher_workers.log
  FOLLOW_LOGS            default: 1; mirror all four process logs in this terminal

The base judge, queue router, and teacher workers are started with nohup.
PID and log files are written under RUN_DIR/{pids,logs}. Run all queue users
on one coordinator host because SQLite is not a cross-machine/HDFS queue.
The default /tmp queue state is not recoverable after a reboot or /tmp cleanup.
EOF
}

die() {
    echo "[fix2-launch] error: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || die "required file does not exist: $path"
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

pid_is_running() {
    local pid_file="$1"
    [[ -s "$pid_file" ]] || return 1
    local pid
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_process() {
    local name="$1"
    shift
    local pid_file="$RUN_DIR/pids/${name}.pid"
    local log_file="$RUN_DIR/logs/${name}.log"
    # These two long-running controllers print frequent progress records.
    # Redirecting them to HDFS/FUSE can turn a normal print into Errno 95 and
    # stop the controller, while its durable work queue remains unfinished.
    if [[ "$name" == "teacher_workers" || "$name" == "failure_router" ]]; then
        log_file="$LOCAL_RUNTIME_LOG_DIR/${name}.log"
        mkdir -p "$LOCAL_RUNTIME_LOG_DIR"
    fi
    if pid_is_running "$pid_file"; then
        die "${name} is already running (pid $(<"$pid_file")). Refusing duplicate launch."
    fi
    rm -f "$pid_file"
    nohup "$@" >"$log_file" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" >"$pid_file"
    echo "[fix2-launch] started ${name}: pid=${pid}, log=${log_file}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || die "unsupported argument: $1 (use environment variables; see --help)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SOURCE_PARQUET="${SOURCE_PARQUET:-$REPO_ROOT/data/benchmarks/fvqa/train_raw.parquet}"
LEGACY_SHAREGPT="${LEGACY_SHAREGPT:-$REPO_ROOT/data/sharegpt_dataset_refined_gpt54_only_17k_final/trajectories_sharegpt.json}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/fix2_fvqa_${RUN_TAG}}"
BASE_URL="${BASE_URL:-http://localhost:6657/v1}"
BASE_SERVED_MODEL="${BASE_SERVED_MODEL:-Qwen3-VL-8B-SFT12-gpt54-only-epoch5}"
# Some batch shells do not export USER.  Resolve it once so fresh launches do
# not unexpectedly fall under /tmp/unknown.
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_fvqa_${RUN_TAG}}"
BASE_LIMIT="${BASE_LIMIT:-100}"
EXPAND_SOURCE_MANIFEST="${EXPAND_SOURCE_MANIFEST:-0}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-16}"
HARD_CAP="${HARD_CAP:-10}"
# In addition to base-answer failures, route base-correct trajectories that
# consumed this many tool calls. Set to 0 to restore wrong-answer-only mining.
MIN_CORRECT_TOOL_CALLS="${MIN_CORRECT_TOOL_CALLS:-6}"
BASE_START="${BASE_START:-0}"
BASE_PARALLEL_WORKERS="${BASE_PARALLEL_WORKERS:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
TEACHER_WORKERS="${TEACHER_WORKERS:-4}"
MAX_READY_TASKS="${MAX_READY_TASKS:-$((TEACHER_WORKERS * 4))}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt54_2_internal_azure_responses}"
TEACHER_API_MODE="${TEACHER_API_MODE:-responses}"
# Do not read ANSWER_JUDGE_MODEL: an earlier launcher revision encouraged
# exporting it, which can silently pin a restarted run to GPT-5.4. Use the
# namespaced override only when intentionally changing the independent judge.
ANSWER_JUDGE_MODEL="${FIX2_ANSWER_JUDGE_MODEL:-multimodal_process_global_2}"
VISION_JUDGE_MODEL="${VISION_JUDGE_MODEL:-gemini38flash_internal_azure}"
TRAJECTORY_QUALITY_JUDGE_MODEL="${TRAJECTORY_QUALITY_JUDGE_MODEL:-gemini38flash_internal_azure}"
STYLE_EXAMPLES_FILE="${STYLE_EXAMPLES_FILE:-$RUN_DIR/style_examples.sanitized.json}"
REQUIRE_STYLE_EXAMPLES="${REQUIRE_STYLE_EXAMPLES:-1}"
BASE_MAX_TOKENS="${BASE_MAX_TOKENS:-8192}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-8192}"
TEACHER_MAX_TURNS="${TEACHER_MAX_TURNS:-30}"
TEACHER_TIMEOUT_S="${TEACHER_TIMEOUT_S:-600}"
TEACHER_PRINT_ROUNDS="${TEACHER_PRINT_ROUNDS:-1}"
FOLLOW_LOGS="${FOLLOW_LOGS:-1}"
# Keep the Responses rollout separate from the prior manual-ReAct queue: a
# task that exhausted its retry budget under the unsupported Chat `stop`
# parameter must be publishable again without mutating durable queue history.
QUEUE_NAME="${QUEUE_NAME:-teacher_collect_responses}"
POLL_S="${POLL_S:-10}"

for value in "$BASE_LIMIT" "$BASE_START" "$MAX_READY_TASKS" "$MIN_CORRECT_TOOL_CALLS"; do
    is_nonnegative_integer "$value" || die "expected a nonnegative integer, got: $value"
done
for value in "$HARD_CAP" "$MATERIALIZE_WORKERS" "$BASE_PARALLEL_WORKERS" "$JUDGE_WORKERS" "$TEACHER_WORKERS" "$BASE_MAX_TOKENS" "$TEACHER_MAX_TOKENS" "$TEACHER_MAX_TURNS" "$TEACHER_TIMEOUT_S"; do
    is_positive_integer "$value" || die "expected a positive integer, got: $value"
done
[[ "$REQUIRE_STYLE_EXAMPLES" == "0" || "$REQUIRE_STYLE_EXAMPLES" == "1" ]] || die "REQUIRE_STYLE_EXAMPLES must be 0 or 1"
[[ "$EXPAND_SOURCE_MANIFEST" == "0" || "$EXPAND_SOURCE_MANIFEST" == "1" ]] || die "EXPAND_SOURCE_MANIFEST must be 0 or 1"
[[ "$TEACHER_PRINT_ROUNDS" == "0" || "$TEACHER_PRINT_ROUNDS" == "1" ]] || die "TEACHER_PRINT_ROUNDS must be 0 or 1"
[[ "$FOLLOW_LOGS" == "0" || "$FOLLOW_LOGS" == "1" ]] || die "FOLLOW_LOGS must be 0 or 1"
case "$TEACHER_API_MODE" in
    manual_react|native_tools|responses) ;;
    *) die "TEACHER_API_MODE must be manual_react, native_tools, or responses" ;;
esac

# Background Python processes otherwise buffer log output when stdout is a file.
export PYTHONUNBUFFERED=1

require_file "$SOURCE_PARQUET"
require_file "$LEGACY_SHAREGPT"
mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/pids" "$LOCAL_STATE_DIR"
LOCAL_STATE_DIR="$(cd "$LOCAL_STATE_DIR" && pwd)"
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR resolves under HDFS: $LOCAL_STATE_DIR" ;;
esac
BASE_EVAL_STATE_DIR="${BASE_EVAL_STATE_DIR:-$LOCAL_STATE_DIR/base_eval}"
mkdir -p "$BASE_EVAL_STATE_DIR"
BASE_EVAL_STATE_DIR="$(cd "$BASE_EVAL_STATE_DIR" && pwd)"
case "$BASE_EVAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "BASE_EVAL_STATE_DIR must be local because evaluator resumes with append writes: $BASE_EVAL_STATE_DIR" ;;
esac
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"
LOCAL_RUNTIME_LOG_DIR="$LOCAL_STATE_DIR/logs"

if [[ "$REQUIRE_STYLE_EXAMPLES" == "1" ]]; then
    require_file "$STYLE_EXAMPLES_FILE"
    if grep -Fq 'Replace this with a sanitized' "$STYLE_EXAMPLES_FILE"; then
        die "STYLE_EXAMPLES_FILE still contains template placeholders; replace them with reviewed sanitized anchors"
    fi
fi

PRIVATE_CASES="$RUN_DIR/cases_private.jsonl"
PUBLIC_CASES="$RUN_DIR/cases_public.jsonl"
BASE_DIR="$RUN_DIR/base_trajectories"
BASE_REPORT="$BASE_EVAL_STATE_DIR/llm_eval_report.json"
BASE_DETAILS="$BASE_EVAL_STATE_DIR/llm_eval_report_details.jsonl"
RESULTS_DIR="$RUN_DIR/teacher_results"
TEACHER_WORKDIR="$LOCAL_STATE_DIR/teacher_runtime"
TEACHER_WRAPPER="$REPO_ROOT/synthesis/post_process/fix2/teacher_worker_group.sh"
require_file "$TEACHER_WRAPPER"

if [[ -f "$PUBLIC_CASES" && ! -f "$PRIVATE_CASES" ]]; then
    die "public manifest exists but private manifest is missing: $PUBLIC_CASES"
elif [[ ! -f "$PRIVATE_CASES" ]]; then
    echo "[fix2-launch] preparing private/public FVQA manifests"
    python -m synthesis.post_process.fix2.run prepare \
        --input-parquet "$SOURCE_PARQUET" \
        --output-cases "$PRIVATE_CASES" \
        --output-public-cases "$PUBLIC_CASES" \
        --image-dir "$RUN_DIR/source_images" \
        --split train \
        --start "$BASE_START" \
        --limit "$BASE_LIMIT" \
        --materialize-workers "$MATERIALIZE_WORKERS" \
        --progress
elif [[ ! -f "$PUBLIC_CASES" ]]; then
    echo "[fix2-launch] resuming partial private FVQA manifest"
    python -m synthesis.post_process.fix2.run prepare \
        --input-parquet "$SOURCE_PARQUET" \
        --output-cases "$PRIVATE_CASES" \
        --output-public-cases "$PUBLIC_CASES" \
        --image-dir "$RUN_DIR/source_images" \
        --split train \
        --start "$BASE_START" \
        --limit "$BASE_LIMIT" \
        --materialize-workers "$MATERIALIZE_WORKERS" \
        --resume \
        --progress
elif [[ "$EXPAND_SOURCE_MANIFEST" == "1" ]]; then
    echo "[fix2-launch] resuming and expanding FVQA manifests through source limit=$BASE_LIMIT"
    python -m synthesis.post_process.fix2.run prepare \
        --input-parquet "$SOURCE_PARQUET" \
        --output-cases "$PRIVATE_CASES" \
        --output-public-cases "$PUBLIC_CASES" \
        --image-dir "$RUN_DIR/source_images" \
        --split train \
        --start "$BASE_START" \
        --limit "$BASE_LIMIT" \
        --materialize-workers "$MATERIALIZE_WORKERS" \
        --resume \
        --progress
else
    echo "[fix2-launch] reusing manifests: $PRIVATE_CASES"
fi

python -m synthesis.post_process.fix2.run profile-style \
    --input-sharegpt "$LEGACY_SHAREGPT" \
    --output-profile "$RUN_DIR/legacy_style_profile.json" \
    --limit 500

# Start consumers first: they will wait safely until base trajectories arrive.
start_process base_judge \
    python -m synthesis.post_process.fix2.run watch-base-eval \
    --trajectory-dir "$BASE_DIR" \
    --answer-file "$SOURCE_PARQUET" \
    --output-path "$BASE_REPORT" \
    --judge-model-alias "$ANSWER_JUDGE_MODEL" \
    --judge-max-tokens 1024 \
    --max-workers "$JUDGE_WORKERS" \
    --poll-s "$POLL_S" \
    --debug

start_process failure_router \
    python -m synthesis.post_process.fix2.run enqueue-failures \
    --cases "$PRIVATE_CASES" \
    --base-eval-details "$BASE_DETAILS" \
    --base-trajectory-dir "$BASE_DIR" \
    --min-correct-tool-calls "$MIN_CORRECT_TOOL_CALLS" \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --limit "$HARD_CAP" \
    --max-ready-tasks "$MAX_READY_TASKS" \
    --max-attempts 3 \
    --follow \
    --poll-s "$POLL_S"

teacher_command=(
    python -m synthesis.post_process.fix2.run teacher-workers
    --queue-db "$QUEUE_DB"
    --queue-name "$QUEUE_NAME"
    --results-dir "$RESULTS_DIR"
    --workdir "$TEACHER_WORKDIR"
    --model "$TEACHER_MODEL"
    --api-mode "$TEACHER_API_MODE"
    --workers "$TEACHER_WORKERS"
    --idle-exit-s 0
    --lease-s 3600
    --max-turns "$TEACHER_MAX_TURNS"
    --max-tokens "$TEACHER_MAX_TOKENS"
    --timeout-s "$TEACHER_TIMEOUT_S"
    --debug
)
if [[ "$REQUIRE_STYLE_EXAMPLES" == "1" ]]; then
    teacher_command+=(--style-examples-file "$STYLE_EXAMPLES_FILE")
fi
if [[ "$TEACHER_PRINT_ROUNDS" == "1" ]]; then
    teacher_command+=(--print-rounds)
fi

echo "[fix2-launch] configuration"
echo "  source parquet:     $SOURCE_PARQUET"
echo "  run directory:      $RUN_DIR"
echo "  local state:        $LOCAL_STATE_DIR"
echo "  base-eval state:    $BASE_EVAL_STATE_DIR"
echo "  base endpoint:      $BASE_URL"
echo "  base served model:  $BASE_SERVED_MODEL"
echo "  base limit:         $BASE_LIMIT (start=$BASE_START)"
echo "  expand manifest:    $EXPAND_SOURCE_MANIFEST"
echo "  materialize workers:$MATERIALIZE_WORKERS"
echo "  hard-case cap:      $HARD_CAP"
echo "  correct-cost bucket:min tools=$MIN_CORRECT_TOOL_CALLS (0 disables)"
echo "  workers:            base=$BASE_PARALLEL_WORKERS judge=$JUDGE_WORKERS teacher=$TEACHER_WORKERS"
echo "  teacher:            model=$TEACHER_MODEL api_mode=$TEACHER_API_MODE"
echo "  final quality judge:$TRAJECTORY_QUALITY_JUDGE_MODEL (runs after deterministic merge)"
echo "  teacher rounds log: $TEACHER_PRINT_ROUNDS"

start_process teacher_workers bash "$TEACHER_WRAPPER" "${teacher_command[@]}"

base_infer_command=(
    python opensearch_vl/run_infer.py
    --model 8b
    --backend api
    --base-url "$BASE_URL"
    --served-model-name "$BASE_SERVED_MODEL"
    --data-path "$SOURCE_PARQUET"
    --output-dir "$BASE_DIR"
    --dataset train
    --start "$BASE_START"
    --parallel-workers "$BASE_PARALLEL_WORKERS"
    --max-tokens "$BASE_MAX_TOKENS"
    --temperature 0
    --api-timeout 600
    --api-max-retries 3
)
# ``prepare`` treats limit=0 as all rows, whereas run_infer treats an explicit
# ``--limit 0`` as an empty range. Omit the flag to request the full source.
if (( BASE_LIMIT > 0 )); then
    base_infer_command+=(--limit "$BASE_LIMIT")
fi

start_process base_infer "${base_infer_command[@]}"

echo
echo "[fix2-launch] pipeline launched"
echo "  run directory:  $RUN_DIR"
echo "  local queue DB: $QUEUE_DB"
echo "  logs:           $RUN_DIR/logs (teacher/router: $LOCAL_RUNTIME_LOG_DIR)"
echo "  PIDs:           $RUN_DIR/pids"
echo "  queue status:   python -m synthesis.post_process.fix2.run queue-stats --queue-db '$QUEUE_DB' --queue-name '$QUEUE_NAME'"
echo "  finalization:       judge-trajectory-quality defaults to $TRAJECTORY_QUALITY_JUDGE_MODEL; $VISION_JUDGE_MODEL is reserved for blind visual review."

if [[ "$FOLLOW_LOGS" == "1" ]]; then
    echo
    echo "[fix2-launch] following live logs below; Ctrl-C stops only log following, not the background pipeline."
    tail -n 0 -F \
        "$RUN_DIR/logs/base_infer.log" \
        "$RUN_DIR/logs/base_judge.log" \
        "$LOCAL_RUNTIME_LOG_DIR/failure_router.log" \
        "$LOCAL_RUNTIME_LOG_DIR/teacher_workers.log"
fi
