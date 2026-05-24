#!/bin/bash
# Run an evaluation set with an OpenAI-compatible model endpoint.
# PARQUET can be a single .parquet file or a directory of .parquet files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PARQUET="${PARQUET:?Set PARQUET to a .parquet file or directory}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_set}"
MODEL="${MODEL:-30b-a3b}"
BACKEND="${BACKEND:-api}"
BASE_URL="${BASE_URL:-${AGENT_BASE_URL:-${OPENAI_BASE_URL:-}}}"
API_KEY="${API_KEY:-${AGENT_API_KEY:-EMPTY}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${AGENT_MODEL_NAME:-}}"
PARALLEL_TASKS="${PARALLEL_TASKS:-10}"
DATASET="${DATASET:-test}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

CMD=(python3 "${SCRIPT_DIR}/run_eval_set.py"
    --data-path "${PARQUET}"
    --output-dir "${OUTPUT_DIR}"
    --model "${MODEL}"
    --backend "${BACKEND}"
    --parallel-tasks "${PARALLEL_TASKS}"
    --dataset "${DATASET}"
    --temperature "${TEMPERATURE}"
    --max-tokens "${MAX_TOKENS}"
)

if [[ -n "${BASE_URL}" ]]; then
    CMD+=(--base-url "${BASE_URL}")
fi
if [[ -n "${API_KEY}" ]]; then
    CMD+=(--api-key "${API_KEY}")
fi
if [[ -n "${SERVED_MODEL_NAME}" ]]; then
    CMD+=(--served-model-name "${SERVED_MODEL_NAME}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
fi

echo "Launching: ${CMD[*]}"
exec "${CMD[@]}"
