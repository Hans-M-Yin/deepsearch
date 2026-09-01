#!/usr/bin/env bash
set -euo pipefail

# One-step multimodal RL smoke test with update-side token-id diagnostics.
# This intentionally does not repair the multimodal payload path.  It only
# records what reaches the Megatron update and keeps the missing-payload check
# from aborting before the update-side instrumentation can run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export RLLM_MM_DEBUG=1
export RLLM_MM_DEBUG_STDOUT=1
export RLLM_MM_DEBUG_PRINT_TOKEN_IDS=1
export RLLM_MM_DEBUG_TOKEN_IDS_MAX="${RLLM_MM_DEBUG_TOKEN_IDS_MAX:-32768}"
export RLLM_MM_DEBUG_ABORT_ON_MISSING=0
export RLLM_MM_DEBUG_MAX_EVENTS="${RLLM_MM_DEBUG_MAX_EVENTS:-200}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RLLM_MM_DEBUG_LOG_FILE="${RLLM_MM_DEBUG_LOG_FILE:-${PROJECT_ROOT}/synthesis/.ignore/rl_test_update_${RUN_STAMP}.log}"

echo "[RLLM_MM_DEBUG] log_file=${RLLM_MM_DEBUG_LOG_FILE}"
echo "[RLLM_MM_DEBUG] abort_on_missing=${RLLM_MM_DEBUG_ABORT_ON_MISSING}"
echo "[RLLM_MM_DEBUG] print_token_ids=${RLLM_MM_DEBUG_PRINT_TOKEN_IDS}"

exec bash "${SCRIPT_DIR}/qwen3-vl-8b-2node-minimal.sh"
