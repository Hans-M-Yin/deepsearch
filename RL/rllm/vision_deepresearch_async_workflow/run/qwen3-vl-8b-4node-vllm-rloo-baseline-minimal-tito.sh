#!/usr/bin/env bash
set -euo pipefail

# Isolated opt-in experiment; the existing cumulative path stays unchanged.
export RLLM_MM_TITO="${RLLM_MM_TITO:-1}"
export RLLM_MM_CONSISTENCY_DEBUG=0
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_4node_vllm_rloo_minimal_tito}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/qwen3-vl-8b-4node-vllm-rloo-baseline-minimal.sh" "$@"
