#!/usr/bin/env bash
# Evaluation-only expert probe over a reproducible sample of completed FVQA
# test failures. This script MUST NOT be used to create SFT training data.
#
# Typical invocation (20 randomly sampled base-model failures):
#   bash synthesis/post_process/fix2/launch_fvqa_test_expert_probe.sh

set -Eeuo pipefail

die() {
    echo "[fix2-test-probe] error: $*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SOURCE_PARQUET="${SOURCE_PARQUET:-$REPO_ROOT/data/benchmarks/fvqa/train.parquet}"
BASE_EVAL_DETAILS="${BASE_EVAL_DETAILS:-$REPO_ROOT/opensearch_infer/outputs/Qwen3-VL-8B-SFT12-gpt54-only-epoch5/fvqa/llm_eval_report_details.jsonl}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/fix2_fvqa_test_expert_probe_${RUN_TAG}}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_fvqa_test_expert_probe_${RUN_TAG}}"
PROBE_SIZE="${PROBE_SIZE:-20}"
SAMPLE_SEED="${SAMPLE_SEED:-20260918}"
TEACHER_WORKERS="${TEACHER_WORKERS:-4}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt54_2_internal_azure_responses}"
TEACHER_API_MODE="${TEACHER_API_MODE:-responses}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-8192}"
TEACHER_MAX_TURNS="${TEACHER_MAX_TURNS:-30}"
TEACHER_TIMEOUT_S="${TEACHER_TIMEOUT_S:-600}"
TEACHER_PRINT_ROUNDS="${TEACHER_PRINT_ROUNDS:-1}"
QUEUE_NAME="${QUEUE_NAME:-fvqa_test_expert_probe_responses}"
ANSWER_JUDGE_MODEL="${FIX2_ANSWER_JUDGE_MODEL:-multimodal_process_global_2}"
TRAJECTORY_QUALITY_JUDGE_MODEL="${TRAJECTORY_QUALITY_JUDGE_MODEL:-gemini38flash_internal_azure}"
VISION_JUDGE_MODEL="${VISION_JUDGE_MODEL:-gemini38flash_internal_azure}"

for path in "$SOURCE_PARQUET" "$BASE_EVAL_DETAILS"; do
    [[ -f "$path" ]] || die "required file does not exist: $path"
done
for value in "$PROBE_SIZE" "$TEACHER_WORKERS" "$TEACHER_MAX_TOKENS" "$TEACHER_MAX_TURNS" "$TEACHER_TIMEOUT_S"; do
    is_positive_integer "$value" || die "expected a positive integer, got: $value"
done
[[ "$SAMPLE_SEED" =~ ^[0-9]+$ ]] || die "SAMPLE_SEED must be a nonnegative integer"
[[ "$TEACHER_PRINT_ROUNDS" == "0" || "$TEACHER_PRINT_ROUNDS" == "1" ]] || die "TEACHER_PRINT_ROUNDS must be 0 or 1"
[[ "$TEACHER_API_MODE" == "responses" || "$TEACHER_API_MODE" == "native_tools" || "$TEACHER_API_MODE" == "manual_react" ]] || die "invalid TEACHER_API_MODE"

mkdir -p "$RUN_DIR" "$LOCAL_STATE_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
LOCAL_STATE_DIR="$(cd "$LOCAL_STATE_DIR" && pwd)"
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR must be local; SQLite and review state cannot live on HDFS" ;;
esac
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"

PRIVATE_CASES="$RUN_DIR/test_probe_cases_private.jsonl"
PUBLIC_CASES="$RUN_DIR/test_probe_cases_public.jsonl"
FAILURE_MANIFEST="$RUN_DIR/test_probe_base_failure_manifest.jsonl"
SELECTION_REPORT="$RUN_DIR/test_probe_selection.json"
IMAGE_DIR="$RUN_DIR/test_probe_images"
RESULTS_DIR="$RUN_DIR/teacher_results"

echo "[fix2-test-probe] EVALUATION ONLY — no output of this run may be used for SFT training."
echo "  source/test parquet: $SOURCE_PARQUET"
echo "  base eval details:   $BASE_EVAL_DETAILS"
echo "  run directory:       $RUN_DIR"
echo "  local state:         $LOCAL_STATE_DIR"
echo "  sample:              $PROBE_SIZE independently judged base failures (seed=$SAMPLE_SEED)"
echo "  teacher:             $TEACHER_MODEL / $TEACHER_API_MODE / workers=$TEACHER_WORKERS"

python -m synthesis.post_process.fix2.run prepare-evaluation-probe \
    --input-parquet "$SOURCE_PARQUET" \
    --base-eval-details "$BASE_EVAL_DETAILS" \
    --output-cases "$PRIVATE_CASES" \
    --output-public-cases "$PUBLIC_CASES" \
    --output-failure-manifest "$FAILURE_MANIFEST" \
    --output-selection-report "$SELECTION_REPORT" \
    --image-dir "$IMAGE_DIR" \
    --limit "$PROBE_SIZE" \
    --seed "$SAMPLE_SEED"

python -m synthesis.post_process.fix2.run enqueue-public-cases \
    --public-cases "$PUBLIC_CASES" \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME"

TEACHER_ARGS=(
    python -m synthesis.post_process.fix2.run teacher-workers
    --queue-db "$QUEUE_DB"
    --queue-name "$QUEUE_NAME"
    --results-dir "$RESULTS_DIR"
    --model "$TEACHER_MODEL"
    --workdir "$LOCAL_STATE_DIR/teacher_runtime"
    --workers "$TEACHER_WORKERS"
    --idle-exit-s 10
    --lease-s 3600
    --api-mode "$TEACHER_API_MODE"
    --max-tokens "$TEACHER_MAX_TOKENS"
    --max-turns "$TEACHER_MAX_TURNS"
    --timeout-s "$TEACHER_TIMEOUT_S"
    --debug
)
if [[ "$TEACHER_PRINT_ROUNDS" == "1" ]]; then
    TEACHER_ARGS+=(--print-rounds)
fi
"${TEACHER_ARGS[@]}"

FIX2_ANSWER_JUDGE_MODEL="$ANSWER_JUDGE_MODEL" \
TRAJECTORY_QUALITY_JUDGE_MODEL="$TRAJECTORY_QUALITY_JUDGE_MODEL" \
VISION_JUDGE_MODEL="$VISION_JUDGE_MODEL" \
QUEUE_NAME="$QUEUE_NAME" \
LOCAL_STATE_DIR="$LOCAL_STATE_DIR" \
PRIVATE_CASES_MANIFEST="$PRIVATE_CASES" \
bash synthesis/post_process/fix2/finalize_fvqa_pipeline.sh "$RUN_DIR" "$LOCAL_STATE_DIR"

echo "[fix2-test-probe] complete; inspect with:"
echo "  bash synthesis/post_process/fix2/inspect_fvqa_finalization.sh '$RUN_DIR' '$LOCAL_STATE_DIR'"
