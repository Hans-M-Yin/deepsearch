#!/usr/bin/env bash
# Rebuild FVQA coordinator-local state after /tmp loss, without redoing base
# inference or immutable teacher trajectories stored under RUN_DIR.

set -Eeuo pipefail

die() {
    echo "[fix2-fvqa-recover] error: $*" >&2
    exit 1
}

require_matching_process_stopped() {
    local label="$1"
    local pid_file="$2"
    local expected="$3"
    [[ -s "$pid_file" ]] || return 0
    local pid command_line
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "$command_line" == *"$expected"* ]] || return 0
    die "${label} is still running (pid=$pid); stop it before local-state recovery"
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
BASE_DETAILS="$LOCAL_STATE_DIR/base_eval/llm_eval_report_details.jsonl"
BASE_REPORT="$LOCAL_STATE_DIR/base_eval/llm_eval_report.json"
BASE_JUDGE_MODEL="${FIX2_ANSWER_JUDGE_MODEL:-multimodal_process_global_2}"
BASE_JUDGE_WORKERS="${BASE_JUDGE_WORKERS:-8}"

[[ -f "$RUN_DIR/cases_private.jsonl" ]] || die "missing durable FVQA private manifest: $RUN_DIR/cases_private.jsonl"
[[ -d "$RUN_DIR/base_trajectories" ]] || die "missing durable base trajectories: $RUN_DIR/base_trajectories"
[[ -d "$RUN_DIR/teacher_results" ]] || die "missing durable teacher results: $RUN_DIR/teacher_results"
[[ -f "$REPO_ROOT/data/benchmarks/fvqa/train_raw.parquet" ]] || die "missing FVQA source parquet"
[[ ! -e "$QUEUE_DB" ]] || die "local queue still exists: $QUEUE_DB (use normal resume, not recovery)"
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR must be coordinator-local, not HDFS: $LOCAL_STATE_DIR" ;;
esac

require_matching_process_stopped failure_router "$RUN_DIR/pids/failure_router.pid" "synthesis.post_process.fix2.run enqueue-failures"
require_matching_process_stopped teacher_workers "$RUN_DIR/pids/teacher_workers.pid" "synthesis.post_process.fix2.run teacher-workers"
mkdir -p "$LOCAL_STATE_DIR/base_eval"

if [[ -f "$BASE_DETAILS" ]]; then
    echo "[fix2-fvqa-recover] reusing surviving local base-judge details: $BASE_DETAILS"
else
    echo "[fix2-fvqa-recover] rebuilding base judge only (no base inference): all durable base trajectories will be judged"
    python -m synthesis.post_process.fix2.run watch-base-eval \
        --trajectory-dir "$RUN_DIR/base_trajectories" \
        --answer-file "$REPO_ROOT/data/benchmarks/fvqa/train_raw.parquet" \
        --output-path "$BASE_REPORT" \
        --judge-model-alias "$BASE_JUDGE_MODEL" \
        --judge-max-tokens 1024 \
        --max-workers "$BASE_JUDGE_WORKERS" \
        --once \
        --debug
fi

echo "[fix2-fvqa-recover] rebuilding teacher queue from the recovered base-judge details"
python -m synthesis.post_process.fix2.run enqueue-failures \
    --cases "$RUN_DIR/cases_private.jsonl" \
    --base-eval-details "$BASE_DETAILS" \
    --base-trajectory-dir "$RUN_DIR/base_trajectories" \
    --min-correct-tool-calls 6 \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --max-attempts 3 \
    --max-ready-tasks 0 \
    --limit 0

echo "[fix2-fvqa-recover] reconciling durable atomic teacher results; these will not be regenerated"
python -m synthesis.post_process.fix2.run reconcile-teacher-results \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --results-dir "$RUN_DIR/teacher_results"

bash "$REPO_ROOT/synthesis/post_process/fix2/resume_fvqa_generation.sh"
echo "[fix2-fvqa-recover] complete; only no-result / previously failed candidates remain for teacher generation"
