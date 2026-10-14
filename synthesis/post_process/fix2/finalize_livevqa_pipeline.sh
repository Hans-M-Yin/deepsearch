#!/usr/bin/env bash
# LiveVQA uses the same post-teacher, label-safe finalization as FVQA. The
# wrapper supplies only the distinct queue default; no base-stage artifact is
# read by the shared finalizer.

set -euo pipefail

QUEUE_NAME="${QUEUE_NAME:-teacher_collect_livevqa_responses}"
export QUEUE_NAME
exec bash "$(dirname "${BASH_SOURCE[0]}")/finalize_fvqa_pipeline.sh" "$@"
