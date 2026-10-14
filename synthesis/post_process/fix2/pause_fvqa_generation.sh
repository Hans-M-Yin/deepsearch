#!/usr/bin/env bash
# Safely pause the FVQA teacher/router pair without losing completed results.

set -Eeuo pipefail

die() {
    echo "[fix2-fvqa-pause] error: $*" >&2
    exit 1
}

stop_recorded_process() {
    local label="$1"
    local pid_file="$2"
    local expected="$3"
    [[ -s "$pid_file" ]] || {
        echo "[fix2-fvqa-pause] ${label}: no PID record"
        return 0
    }
    local pid command_line
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || die "${label}: malformed PID record: $pid_file"
    kill -0 "$pid" 2>/dev/null || {
        echo "[fix2-fvqa-pause] ${label}: already stopped (pid=$pid)"
        return 0
    }
    command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "$command_line" == *"$expected"* ]] || die "refusing ${label} pid=$pid; command does not match this run"
    kill -TERM "$pid"
    echo "[fix2-fvqa-pause] sent TERM to ${label} pid=$pid"
    for _ in {1..30}; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    die "${label} pid=$pid did not stop within 30 seconds; queue was not changed"
}

[[ $# -eq 0 ]] || die "usage: $0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_TAG="${RUN_TAG:-train_scale_v1}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_fvqa_${RUN_TAG}}"
RUN_DIR="$REPO_ROOT/runs/fix2_fvqa_${RUN_TAG}"
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"
QUEUE_NAME="${QUEUE_NAME:-teacher_collect_responses}"

[[ -f "$QUEUE_DB" ]] || die "missing local teacher queue: $QUEUE_DB (run on the original coordinator)"

# Stop publication first, then stop consumers. This prevents a producer from
# racing with the controlled lease return below.
stop_recorded_process failure_router "$RUN_DIR/pids/failure_router.pid" "synthesis.post_process.fix2.run enqueue-failures"
stop_recorded_process teacher_workers "$RUN_DIR/pids/teacher_workers.pid" "synthesis.post_process.fix2.run teacher-workers"

python -m synthesis.post_process.fix2.run requeue-leased \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --reason "controlled_fvqa_pause"

echo "[fix2-fvqa-pause] complete; raw results and completed queue rows were preserved"
