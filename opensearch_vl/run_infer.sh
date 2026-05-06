#!/bin/bash
# Convenience wrapper for run_infer.py. Edit the variables below or pass
# them in via the environment to run a benchmark end-to-end.
#
# Required environment variables vary by --model:
#   * 8b / 32b / 30b-a3b
#       QWEN3VL_8B_PATH | QWEN3VL_32B_PATH | QWEN3VL_30B_A3B_PATH
#       (or pass --checkpoint explicitly)
#   * claude
#       CLAUDE_API_HOST, CLAUDE_API_USER, CLAUDE_API_KEY
#
# Tools that require remote services use the API gateway when
# (API_HOST, API_USER, API_KEY) are set, and otherwise fall back to
# (SERPER_API_KEY, JINA_API_KEY).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:-8b}"
GPUS="${GPUS:-0}"
DTYPE="${DTYPE:-bfloat16}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the FVQA-style parquet file}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a writable directory}"
DATASET="${DATASET:-train}"
START="${START:-0}"
LIMIT="${LIMIT:-}"
CATEGORY="${CATEGORY:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

CMD=(python3 "${SCRIPT_DIR}/run_infer.py"
    --model "${MODEL}"
    --gpus "${GPUS}"
    --dtype "${DTYPE}"
    --data-path "${DATA_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --dataset "${DATASET}"
    --start "${START}"
    --log-level "${LOG_LEVEL}"
)

if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
fi
if [[ -n "${CATEGORY}" ]]; then
    CMD+=(--category "${CATEGORY}")
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
fi

echo "Launching: ${CMD[*]}"
exec "${CMD[@]}"
