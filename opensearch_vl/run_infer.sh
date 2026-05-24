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
#   * api
#       AGENT_BASE_URL / AGENT_MODEL_NAME (or OPENAI_BASE_URL)
#
# Tools that require remote services use the API gateway when
# (API_HOST, API_USER, API_KEY) are set, and otherwise fall back to
# (SERPER_API_KEY, JINA_API_KEY).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:-8b}"
BACKEND="${BACKEND:-local}"
GPUS="${GPUS:-0}"
DTYPE="${DTYPE:-bfloat16}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the FVQA-style parquet file}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a writable directory}"
DATASET="${DATASET:-train}"
START="${START:-0}"
LIMIT="${LIMIT:-}"
CATEGORY="${CATEGORY:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-${PARALLEL_TASKS:-1}}"
BASE_URL="${BASE_URL:-${AGENT_BASE_URL:-${OPENAI_BASE_URL:-}}}"
API_KEY="${API_KEY:-${AGENT_API_KEY:-EMPTY}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${AGENT_MODEL_NAME:-}}"

CMD=(python3 "${SCRIPT_DIR}/run_infer.py"
    --model "${MODEL}"
    --backend "${BACKEND}"
    --gpus "${GPUS}"
    --dtype "${DTYPE}"
    --data-path "${DATA_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --dataset "${DATASET}"
    --start "${START}"
    --parallel-workers "${PARALLEL_WORKERS}"
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
if [[ "${BACKEND}" == "api" ]]; then
    if [[ -n "${BASE_URL}" ]]; then
        CMD+=(--base-url "${BASE_URL}")
    fi
    if [[ -n "${API_KEY}" ]]; then
        CMD+=(--api-key "${API_KEY}")
    fi
    if [[ -n "${SERVED_MODEL_NAME}" ]]; then
        CMD+=(--served-model-name "${SERVED_MODEL_NAME}")
    fi
fi

echo "Launching: ${CMD[*]}"
exec "${CMD[@]}"
