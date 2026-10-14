#!/usr/bin/env bash
# Finalize an FVQA teacher run after its target quota has completed.
#
# Example (mixed visual + grounded nonvisual export):
#   bash synthesis/post_process/fix2/finalize_fvqa_pipeline.sh \
#     runs/fix2_fvqa_20260918_014403
#
# For a run whose private case manifest has a non-default name (for example,
# the evaluation-only test probe), set PRIVATE_CASES_MANIFEST explicitly.
#
# Set REQUIRE_VISUAL_COMPARE=1 to export only the strong visual-comparison
# branch. By default, grounded weak/nonvisual trajectories can pass the
# label-blind Gemini trajectory-quality review.
# A successful train-run finalization also writes a ShareGPT export unless
# EXPORT_SHAREGPT=0 (used by the evaluation-only test-probe wrapper).

set -euo pipefail

die() {
    echo "[fix2-finalize] error: $*" >&2
    exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || die "usage: $0 RUN_DIR [LOCAL_STATE_DIR]"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="$1"
[[ -d "$RUN_DIR" ]] || die "RUN_DIR does not exist: $RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
RUN_NAME="$(basename "$RUN_DIR")"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${2:-${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/${RUN_NAME}}}"
[[ -d "$LOCAL_STATE_DIR" ]] || die "LOCAL_STATE_DIR does not exist: $LOCAL_STATE_DIR"
LOCAL_STATE_DIR="$(cd "$LOCAL_STATE_DIR" && pwd)"
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR must be coordinator-local, not HDFS: $LOCAL_STATE_DIR" ;;
esac
FINAL_STATE_DIR="${FINAL_STATE_DIR:-$LOCAL_STATE_DIR/finalization}"
mkdir -p "$FINAL_STATE_DIR"
FINAL_STATE_DIR="$(cd "$FINAL_STATE_DIR" && pwd)"
case "$FINAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "FINAL_STATE_DIR must be local because review JSONL files are append/resume state: $FINAL_STATE_DIR" ;;
esac

QUEUE_NAME="${QUEUE_NAME:-teacher_collect_responses}"
# Queue state is coordinator-local and is useful only for diagnostics here.
# Final judge inputs are rebuilt from immutable shared per-case result files.
QUEUE_DB="${QUEUE_DB:-$LOCAL_STATE_DIR/teacher_queue.sqlite}"
PRIVATE_CASES_MANIFEST="${PRIVATE_CASES_MANIFEST:-$RUN_DIR/cases_private.jsonl}"
ANSWER_JUDGE_MODEL="${FIX2_ANSWER_JUDGE_MODEL:-multimodal_process_global_2}"
TRAJECTORY_QUALITY_JUDGE_MODEL="${TRAJECTORY_QUALITY_JUDGE_MODEL:-gemini38flash_internal_azure}"
VISION_JUDGE_MODEL="${VISION_JUDGE_MODEL:-gemini38flash_internal_azure}"
ANSWER_JUDGE_WORKERS="${ANSWER_JUDGE_WORKERS:-64}"
QUALITY_JUDGE_WORKERS="${QUALITY_JUDGE_WORKERS:-64}"
VISION_JUDGE_WORKERS="${VISION_JUDGE_WORKERS:-64}"
REQUIRE_VISUAL_COMPARE="${REQUIRE_VISUAL_COMPARE:-0}"
# Keep completed expert chains up to the same 30-turn ceiling used by the
# collector.  Correctness and the label-blind trajectory-quality review remain
# mandatory acceptance gates.
FILTER_MAX_ASSISTANT_TURNS="${FILTER_MAX_ASSISTANT_TURNS:-30}"
EXPORT_SHAREGPT="${EXPORT_SHAREGPT:-1}"

[[ -f "$PRIVATE_CASES_MANIFEST" ]] || die "missing private case manifest: $PRIVATE_CASES_MANIFEST"
[[ -d "$RUN_DIR/teacher_results/cases" ]] || die "missing teacher result directory"
[[ "$REQUIRE_VISUAL_COMPARE" == "0" || "$REQUIRE_VISUAL_COMPARE" == "1" ]] || die "REQUIRE_VISUAL_COMPARE must be 0 or 1"
[[ "$FILTER_MAX_ASSISTANT_TURNS" =~ ^[1-9][0-9]*$ ]] || die "FILTER_MAX_ASSISTANT_TURNS must be a positive integer"
[[ "$EXPORT_SHAREGPT" == "0" || "$EXPORT_SHAREGPT" == "1" ]] || die "EXPORT_SHAREGPT must be 0 or 1"
[[ "$ANSWER_JUDGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "ANSWER_JUDGE_WORKERS must be a positive integer"
[[ "$QUALITY_JUDGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "QUALITY_JUDGE_WORKERS must be a positive integer"
[[ "$VISION_JUDGE_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "VISION_JUDGE_WORKERS must be a positive integer"

RAW_JSONL="$FINAL_STATE_DIR/teacher_raw.jsonl"
ANSWER_REVIEWS="$FINAL_STATE_DIR/teacher_answer_reviews.jsonl"
QUALITY_REVIEWS="$FINAL_STATE_DIR/teacher_trajectory_quality_reviews.jsonl"
BLIND_REVIEWS="$FINAL_STATE_DIR/teacher_blind_visual_reviews.jsonl"
ACCEPTED="$FINAL_STATE_DIR/teacher_accepted_raw.jsonl"
REJECTED="$FINAL_STATE_DIR/teacher_rejected_raw.jsonl"
AUDIT="$FINAL_STATE_DIR/teacher_filter_audit.jsonl"

completed_final_outputs=0
for target in "$ACCEPTED" "$REJECTED" "$AUDIT"; do
    [[ -e "$target" ]] && ((completed_final_outputs += 1))
done
[[ "$completed_final_outputs" == "0" || "$completed_final_outputs" == "3" ]] \
    || die "incomplete final filter state (expected all or none of accepted/rejected/audit outputs)"

if [[ "$completed_final_outputs" == "3" ]]; then
    echo "[fix2-finalize] reusing completed final filter outputs; continuing with ShareGPT export"
else
    if [[ -f "$QUEUE_DB" ]]; then
        echo "[fix2-finalize] queue diagnostic: $(python -m synthesis.post_process.fix2.run queue-stats --queue-db "$QUEUE_DB" --queue-name "$QUEUE_NAME")"
    else
        echo "[fix2-finalize] queue diagnostic: unavailable (finalization uses shared atomic results)"
    fi
    if [[ ! -s "$RAW_JSONL" ]]; then
        python -m synthesis.post_process.fix2.run merge-atomic-teacher-results \
            --results-dir "$RUN_DIR/teacher_results" \
            --output-jsonl "$RAW_JSONL"
    else
        echo "[fix2-finalize] reusing merged teacher records: $RAW_JSONL"
    fi

    python -m synthesis.post_process.fix2.run judge-expert \
        --input-jsonl "$RAW_JSONL" \
        --private-cases "$PRIVATE_CASES_MANIFEST" \
        --output-jsonl "$ANSWER_REVIEWS" \
        --model-alias "$ANSWER_JUDGE_MODEL" \
        --max-workers "$ANSWER_JUDGE_WORKERS" \
        --resume

    python -m synthesis.post_process.fix2.run judge-trajectory-quality \
        --input-jsonl "$RAW_JSONL" \
        --output-jsonl "$QUALITY_REVIEWS" \
        --model-alias "$TRAJECTORY_QUALITY_JUDGE_MODEL" \
        --max-workers "$QUALITY_JUDGE_WORKERS" \
        --resume

    python -m synthesis.post_process.fix2.run blind-visual-review \
        --input-jsonl "$RAW_JSONL" \
        --output-jsonl "$BLIND_REVIEWS" \
        --model-alias "$VISION_JUDGE_MODEL" \
        --max-workers "$VISION_JUDGE_WORKERS" \
        --resume

    FILTER_ARGS=()
    if [[ "$REQUIRE_VISUAL_COMPARE" == "1" ]]; then
        FILTER_ARGS+=(--require-visual-compare)
    fi
    python -m synthesis.post_process.fix2.run filter \
        --input-jsonl "$RAW_JSONL" \
        --private-cases "$PRIVATE_CASES_MANIFEST" \
        --answer-reviews "$ANSWER_REVIEWS" \
        --trajectory-quality-reviews "$QUALITY_REVIEWS" \
        --blind-reviews "$BLIND_REVIEWS" \
        --accepted-output "$ACCEPTED" \
        --rejected-output "$REJECTED" \
        --audit-output "$AUDIT" \
        --max-assistant-turns "$FILTER_MAX_ASSISTANT_TURNS" \
        "${FILTER_ARGS[@]}"
fi

echo "[fix2-finalize] filter summary: $(python -m synthesis.post_process.fix2.run summarize-filter-audit --audit-jsonl "$AUDIT")"

if [[ "$EXPORT_SHAREGPT" == "1" ]]; then
    bash synthesis/post_process/fix2/export_fvqa_sharegpt.sh "$RUN_DIR" "$LOCAL_STATE_DIR"
else
    echo "[fix2-finalize] ShareGPT export skipped (EXPORT_SHAREGPT=0)"
fi

echo "[fix2-finalize] complete"
echo "  local final state: $FINAL_STATE_DIR"
echo "  accepted: $ACCEPTED"
echo "  rejected: $REJECTED"
echo "  audit:    $AUDIT"
