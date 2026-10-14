#!/usr/bin/env bash
# Rebuild the LiveVQA local queue after /tmp loss from public manifests and
# immutable shared teacher results. It never re-samples the 3,200 examples.

set -Eeuo pipefail

die() {
    echo "[fix2-livevqa-recover] error: $*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "usage: $0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_TAG="${RUN_TAG:-gpt55_loopfix_scale2400}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_livevqa_${RUN_TAG}}"
RUN_DIR="$REPO_ROOT/runs/fix2_livevqa_${RUN_TAG}"
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"
QUEUE_NAME="${QUEUE_NAME:-teacher_collect_livevqa_responses}"
PID_FILE="$RUN_DIR/pids/teacher_workers.pid"

[[ -f "$RUN_DIR/cases_public.jsonl" ]] || die "missing durable LiveVQA public manifest: $RUN_DIR/cases_public.jsonl"
[[ -d "$RUN_DIR/teacher_results" ]] || die "missing durable teacher results: $RUN_DIR/teacher_results"
[[ ! -e "$QUEUE_DB" ]] || die "local queue still exists: $QUEUE_DB (use normal resume, not recovery)"
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR must be coordinator-local, not HDFS: $LOCAL_STATE_DIR" ;;
esac
if [[ -s "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
        command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
        [[ "$command_line" == *"synthesis.post_process.fix2.run teacher-workers"* ]] && die "teacher_workers is still running (pid=$pid); stop it before local-state recovery"
    fi
fi

mkdir -p "$LOCAL_STATE_DIR"
echo "[fix2-livevqa-recover] rebuilding the full public queue from the existing 3,200-case manifest"
python -m synthesis.post_process.fix2.run enqueue-public-cases \
    --public-cases "$RUN_DIR/cases_public.jsonl" \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --max-attempts 3

echo "[fix2-livevqa-recover] reconciling durable atomic teacher results; these will not be regenerated"
python -m synthesis.post_process.fix2.run reconcile-teacher-results \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --results-dir "$RUN_DIR/teacher_results"

bash "$REPO_ROOT/synthesis/post_process/fix2/resume_livevqa_generation.sh"
echo "[fix2-livevqa-recover] complete; only no-result / previously failed cases remain for GPT-5.5 collection"
