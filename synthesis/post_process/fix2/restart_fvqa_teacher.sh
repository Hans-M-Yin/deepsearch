#!/usr/bin/env bash
# Stop and immediately resume only the FVQA teacher worker group. Completed
# atomic case results stay durable; only tasks held by stopped workers return
# to the local SQLite queue.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash synthesis/post_process/fix2/restart_fvqa_teacher.sh

Run this on the same coordinator that owns /tmp/$USER/fix2_fvqa_train_scale_v1.
It stops only teacher_workers, safely requeues its in-flight leases, then
starts a new teacher group from the same durable queue. It never stops or
reruns base_infer, base_judge, or failure_router.

Optional environment:
  RUN_TAG                default: train_scale_v1
  LOCAL_STATE_DIR        default: /tmp/$USER/fix2_fvqa_<RUN_TAG>
  TEACHER_WORKERS        default: 16
  TEACHER_MODEL          default: gpt54_2_internal_azure_responses
  TEACHER_API_MODE       default: responses
  TEACHER_MAX_TURNS      default: 30
  TEACHER_MAX_TOKENS     default: 8192
  TEACHER_TIMEOUT_S      default: 600
  TEACHER_PRINT_ROUNDS   default: 1; set 0 to reduce per-round log volume

Teacher stdout/stderr is always written under LOCAL_STATE_DIR/logs, not HDFS.
EOF
}

die() {
    echo "[fix2-teacher-restart] error: $*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || die "usage: $0"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_TAG="${RUN_TAG:-train_scale_v1}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_fvqa_${RUN_TAG}}"
TEACHER_WORKERS="${TEACHER_WORKERS:-16}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt54_2_internal_azure_responses}"
TEACHER_API_MODE="${TEACHER_API_MODE:-responses}"
TEACHER_MAX_TURNS="${TEACHER_MAX_TURNS:-30}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-8192}"
TEACHER_TIMEOUT_S="${TEACHER_TIMEOUT_S:-600}"
TEACHER_PRINT_ROUNDS="${TEACHER_PRINT_ROUNDS:-1}"
QUEUE_NAME="${QUEUE_NAME:-teacher_collect_responses}"
STYLE_EXAMPLES_FILE="${STYLE_EXAMPLES_FILE:-$REPO_ROOT/synthesis/post_process/fix2/style_examples.production.legacy.json}"
TEACHER_WRAPPER="$REPO_ROOT/synthesis/post_process/fix2/teacher_worker_group.sh"
RUN_DIR="$REPO_ROOT/runs/fix2_fvqa_${RUN_TAG}"
PID_FILE="$RUN_DIR/pids/teacher_workers.pid"
LOCAL_LOG_DIR="$LOCAL_STATE_DIR/logs"
LOG_FILE="$LOCAL_LOG_DIR/teacher_workers.log"
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"
RESULTS_DIR="$RUN_DIR/teacher_results"
TEACHER_WORKDIR="$LOCAL_STATE_DIR/teacher_runtime"

for value in "$TEACHER_WORKERS" "$TEACHER_MAX_TURNS" "$TEACHER_MAX_TOKENS" "$TEACHER_TIMEOUT_S"; do
    is_positive_integer "$value" || die "expected a positive integer, got: $value"
done
[[ "$TEACHER_PRINT_ROUNDS" == "0" || "$TEACHER_PRINT_ROUNDS" == "1" ]] || die "TEACHER_PRINT_ROUNDS must be 0 or 1"
case "$TEACHER_API_MODE" in
    manual_react|native_tools|responses) ;;
    *) die "TEACHER_API_MODE must be manual_react, native_tools, or responses" ;;
esac
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR must be coordinator-local, not HDFS: $LOCAL_STATE_DIR" ;;
esac
[[ -f "$QUEUE_DB" ]] || die "missing local teacher queue: $QUEUE_DB"
[[ -f "$STYLE_EXAMPLES_FILE" ]] || die "missing style examples: $STYLE_EXAMPLES_FILE"
[[ -f "$TEACHER_WRAPPER" ]] || die "missing teacher wrapper: $TEACHER_WRAPPER"

if [[ -s "$PID_FILE" ]]; then
    previous_pid="$(<"$PID_FILE")"
    if [[ "$previous_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        command_line="$(tr '\0' ' ' < "/proc/${previous_pid}/cmdline" 2>/dev/null || true)"
        [[ "$command_line" == *"synthesis.post_process.fix2.run teacher-workers"* ]] || die "refusing pid=$previous_pid; it is not this run's teacher_workers command"
        kill -TERM "$previous_pid"
        echo "[fix2-teacher-restart] sent TERM to teacher_workers pid=$previous_pid"
        for _ in {1..30}; do
            kill -0 "$previous_pid" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$previous_pid" 2>/dev/null && die "teacher_workers pid=$previous_pid did not stop; leases were not touched"
    else
        echo "[fix2-teacher-restart] previous teacher PID is already stopped; continuing"
    fi
fi

# This is safe only after the PID above has stopped. It avoids a one-hour
# delay from the normal lease timeout for interrupted samples. Succeeded tasks
# and their atomic case JSON remain untouched.
python -m synthesis.post_process.fix2.run requeue-leased \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --reason "teacher_workers_restarted"

mkdir -p "$RUN_DIR/pids" "$RUN_DIR/logs" "$RESULTS_DIR" "$TEACHER_WORKDIR" "$LOCAL_LOG_DIR"
command=(
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
    --style-examples-file "$STYLE_EXAMPLES_FILE"
    --debug
)
if [[ "$TEACHER_PRINT_ROUNDS" == "1" ]]; then
    command+=(--print-rounds)
fi

echo "[fix2-teacher-restart] restarting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_FILE"
PYTHONUNBUFFERED=1 nohup bash "$TEACHER_WRAPPER" "${command[@]}" >>"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[fix2-teacher-restart] started pid=$pid"
echo "  run:     $RUN_DIR"
echo "  state:   $LOCAL_STATE_DIR"
echo "  workers: $TEACHER_WORKERS"
echo "  log:     $LOG_FILE"
