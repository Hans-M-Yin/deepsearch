#!/usr/bin/env bash
# Resume the exact 3,200-sample GPT-5.5 LiveVQA collection without resampling.

set -Eeuo pipefail

[[ $# -eq 0 ]] || {
    echo "[fix2-livevqa-resume] usage: $0" >&2
    exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_TAG="${RUN_TAG:-gpt55_loopfix_scale2400}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"

RUN_TAG="$RUN_TAG" \
RUN_DIR="$REPO_ROOT/runs/fix2_livevqa_${RUN_TAG}" \
LOCAL_STATE_DIR="/tmp/${LOCAL_USER}/fix2_livevqa_${RUN_TAG}" \
SOURCE_PARQUET="$REPO_ROOT/../data/benchmarks/livevqa/train_random10k.parquet" \
SAMPLE_COUNT=3200 \
SAMPLE_SEED=20260920 \
EXPAND_SOURCE_MANIFEST=1 \
DOWNLOAD_WORKERS=16 \
ARCHIVE_CACHE_DIR="/tmp/${LOCAL_USER}/livevqa_archives" \
TEACHER_WORKERS=16 \
TEACHER_MODEL=gpt55_2_internal_azure \
TEACHER_API_MODE=responses \
TEACHER_MAX_TURNS=30 \
FOLLOW_LOGS=0 \
bash "$REPO_ROOT/synthesis/post_process/fix2/launch_livevqa_pipeline.sh"

echo "[fix2-livevqa-resume] complete; the existing 3,200-case manifest, queue, and raw results were reused"
