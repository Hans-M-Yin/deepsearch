#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Dry-run needs neither API credentials nor training dependencies.
if [[ " ${*} " != *" --dry-run "* ]]; then
  set -a
  source RL/.env_remote
  set +a
fi
export PARTIAL_CREDIT_JUDGE_MODEL="${PARTIAL_CREDIT_JUDGE_MODEL:-gemini38flash_internal_azure}"
export PARTIAL_CREDIT_TRACE_CHUNK_STEPS="${PARTIAL_CREDIT_TRACE_CHUNK_STEPS:-15}"
export PARTIAL_CREDIT_JUDGE_TIMEOUT_S="${PARTIAL_CREDIT_JUDGE_TIMEOUT_S:-120}"
export PARTIAL_CREDIT_JUDGE_MAX_TOKENS="${PARTIAL_CREDIT_JUDGE_MAX_TOKENS:-4096}"
exec python -u debug/rejudge_partial_credit_audit12.py "$@"
