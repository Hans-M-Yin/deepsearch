#!/usr/bin/env bash
# Re-run only missing FVQA base trajectories after the original base process
# exits.  run_infer treats existing *_trajectory.json files as durable
# completions, so failed/no-output cases are retried and successful ones skip.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash synthesis/post_process/fix2/resume_fvqa_base_infer.sh

Run only after the original base_infer process has exited.  Defaults target the
durable train_scale_v1 experiment and preserve its full-source resume policy.

Optional environment:
  RUN_TAG                default: train_scale_v1
  LOCAL_STATE_DIR        default: /tmp/$USER/fix2_fvqa_<RUN_TAG>
  SOURCE_PARQUET         default: data/benchmarks/fvqa/train_raw.parquet
  BASE_URL               default: http://localhost:6657/v1
  BASE_SERVED_MODEL      default: Qwen3-VL-8B-SFT12-gpt54-only-epoch5
  BASE_PARALLEL_WORKERS  default: 16
  BASE_MAX_TOKENS        default: 8192
  BASE_LIMIT             default: 0 (all source rows)
EOF
}

die() {
    echo "[fix2-base-resume] error: $*" >&2
    exit 1
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
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
SOURCE_PARQUET="${SOURCE_PARQUET:-$REPO_ROOT/data/benchmarks/fvqa/train_raw.parquet}"
BASE_URL="${BASE_URL:-http://localhost:6657/v1}"
BASE_SERVED_MODEL="${BASE_SERVED_MODEL:-Qwen3-VL-8B-SFT12-gpt54-only-epoch5}"
BASE_PARALLEL_WORKERS="${BASE_PARALLEL_WORKERS:-16}"
BASE_MAX_TOKENS="${BASE_MAX_TOKENS:-8192}"
BASE_LIMIT="${BASE_LIMIT:-0}"
RUN_DIR="$REPO_ROOT/runs/fix2_fvqa_${RUN_TAG}"
BASE_DIR="$RUN_DIR/base_trajectories"
PID_FILE="$RUN_DIR/pids/base_infer.pid"
LOG_FILE="$RUN_DIR/logs/base_infer.log"

is_positive_integer "$BASE_PARALLEL_WORKERS" || die "BASE_PARALLEL_WORKERS must be positive"
is_positive_integer "$BASE_MAX_TOKENS" || die "BASE_MAX_TOKENS must be positive"
is_nonnegative_integer "$BASE_LIMIT" || die "BASE_LIMIT must be nonnegative"
[[ -f "$SOURCE_PARQUET" ]] || die "missing source parquet: $SOURCE_PARQUET"
[[ -d "$BASE_DIR" ]] || die "missing base trajectory directory: $BASE_DIR"
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR must be coordinator-local, not HDFS: $LOCAL_STATE_DIR" ;;
esac

if [[ -s "$PID_FILE" ]]; then
    previous_pid="$(<"$PID_FILE")"
    if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        die "base_infer is still running (pid $previous_pid); wait for it rather than starting a duplicate"
    fi
fi

command=(
    python opensearch_vl/run_infer.py
    --model 8b
    --backend api
    --base-url "$BASE_URL"
    --served-model-name "$BASE_SERVED_MODEL"
    --data-path "$SOURCE_PARQUET"
    --output-dir "$BASE_DIR"
    --dataset train
    --start 0
    --parallel-workers "$BASE_PARALLEL_WORKERS"
    --max-tokens "$BASE_MAX_TOKENS"
    --temperature 0
    --api-timeout 600
    --api-max-retries 3
)
if (( BASE_LIMIT > 0 )); then
    command+=(--limit "$BASE_LIMIT")
fi

mkdir -p "$RUN_DIR/pids" "$RUN_DIR/logs"
echo "[fix2-base-resume] restarting missing cases at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_FILE"
PYTHONUNBUFFERED=1 nohup "${command[@]}" >>"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$PID_FILE"

echo "[fix2-base-resume] started pid=$pid"
echo "  run:  $RUN_DIR"
echo "  log:  $LOG_FILE"
echo "  note: existing trajectories skip; missing/failed cases retry"
