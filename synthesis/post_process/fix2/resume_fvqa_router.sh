#!/usr/bin/env bash
# Resume only the failed FVQA evaluation-to-teacher router on its original
# coordinator.  It never reruns base inference, base judging, or completed
# teacher tasks; the existing SQLite queue makes publication idempotent.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash synthesis/post_process/fix2/resume_fvqa_router.sh [TARGET_CASES]

Defaults to the durable train_scale_v1 run and a 1,024-task queue cap.  Run
this on the same coordinator that owns /tmp/$USER/fix2_fvqa_train_scale_v1.

Optional environment:
  RUN_TAG              default: train_scale_v1
  LOCAL_STATE_DIR      default: /tmp/$USER/fix2_fvqa_<RUN_TAG>
  TARGET_CASES         default: 1024
  MIN_CORRECT_TOOL_CALLS default: 6
  MAX_READY_TASKS      default: 64
  QUEUE_NAME           default: teacher_collect_responses
  POLL_S               default: 10
EOF
}

die() {
    echo "[fix2-router-resume] error: $*" >&2
    exit 1
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
[[ $# -le 1 ]] || die "usage: $0 [TARGET_CASES]"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_TAG="${RUN_TAG:-train_scale_v1}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_fvqa_${RUN_TAG}}"
TARGET_CASES="${1:-${TARGET_CASES:-1024}}"
MIN_CORRECT_TOOL_CALLS="${MIN_CORRECT_TOOL_CALLS:-6}"
MAX_READY_TASKS="${MAX_READY_TASKS:-64}"
QUEUE_NAME="${QUEUE_NAME:-teacher_collect_responses}"
POLL_S="${POLL_S:-10}"
RUN_DIR="$REPO_ROOT/runs/fix2_fvqa_${RUN_TAG}"
CASES="$RUN_DIR/cases_private.jsonl"
BASE_TRAJECTORY_DIR="$RUN_DIR/base_trajectories"
BASE_EVAL_DETAILS="$LOCAL_STATE_DIR/base_eval/llm_eval_report_details.jsonl"
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"
PID_FILE="$RUN_DIR/pids/failure_router.pid"
# The router logs frequently while following the base-evaluation JSONL.  Its
# stdout must be coordinator-local: HDFS/FUSE append writes can fail with
# Errno 95 and silently leave the teacher queue empty.
LOCAL_LOG_DIR="$LOCAL_STATE_DIR/logs"
LOG_FILE="$LOCAL_LOG_DIR/failure_router.log"

for value in "$TARGET_CASES" "$MIN_CORRECT_TOOL_CALLS" "$MAX_READY_TASKS"; do
    is_nonnegative_integer "$value" || die "expected a nonnegative integer, got: $value"
done
[[ -f "$CASES" ]] || die "missing private case manifest: $CASES"
[[ -d "$BASE_TRAJECTORY_DIR" ]] || die "missing base trajectory directory: $BASE_TRAJECTORY_DIR"
[[ -f "$BASE_EVAL_DETAILS" ]] || die "missing local base judge details: $BASE_EVAL_DETAILS"
[[ -f "$QUEUE_DB" ]] || die "missing local teacher queue: $QUEUE_DB"
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR must be coordinator-local, not HDFS: $LOCAL_STATE_DIR" ;;
esac

if [[ -s "$PID_FILE" ]]; then
    previous_pid="$(<"$PID_FILE")"
    if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        die "failure_router is already running (pid $previous_pid); do not start a duplicate"
    fi
fi

mkdir -p "$RUN_DIR/pids" "$RUN_DIR/logs" "$LOCAL_LOG_DIR"
echo "[fix2-router-resume] restarting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_FILE"
PYTHONUNBUFFERED=1 nohup python -m synthesis.post_process.fix2.run enqueue-failures \
    --cases "$CASES" \
    --base-eval-details "$BASE_EVAL_DETAILS" \
    --base-trajectory-dir "$BASE_TRAJECTORY_DIR" \
    --min-correct-tool-calls "$MIN_CORRECT_TOOL_CALLS" \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --limit "$TARGET_CASES" \
    --max-ready-tasks "$MAX_READY_TASKS" \
    --max-attempts 3 \
    --follow \
    --poll-s "$POLL_S" >>"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[fix2-router-resume] started pid=$pid"
echo "  run:       $RUN_DIR"
echo "  state:     $LOCAL_STATE_DIR"
echo "  queue cap: $TARGET_CASES"
echo "  log:       $LOG_FILE"
