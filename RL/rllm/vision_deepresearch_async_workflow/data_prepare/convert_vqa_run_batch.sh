#!/usr/bin/env bash
set -euo pipefail

# Example:
#   RUN_DIR=runs/.../vqa/0803_113820 \
#   OUTPUT_JSONL=data/vqa_0803.jsonl \
#   ./convert_vqa_run_batch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${RUN_DIR:?Set RUN_DIR to a vqa/run_batch output directory}"
: "${OUTPUT_JSONL:?Set OUTPUT_JSONL to the normalized JSONL path}"

ARGS=(
  --run_dir "${RUN_DIR}"
  --output_jsonl "${OUTPUT_JSONL}"
  --image_mode "${IMAGE_MODE:-download}"
  --timeout "${IMAGE_TIMEOUT:-30}"
  --retries "${IMAGE_RETRIES:-3}"
)

if [[ -n "${IMAGE_DIR:-}" ]]; then
  ARGS+=(--image_dir "${IMAGE_DIR}")
fi
if [[ -n "${SYSTEM_PROMPT_FILE:-}" ]]; then
  ARGS+=(--system_prompt_file "${SYSTEM_PROMPT_FILE}")
fi
if [[ "${INCLUDE_SAMPLES_METADATA:-0}" == "1" ]]; then
  ARGS+=(--include_samples_metadata)
fi
if [[ -n "${OFFSET:-}" ]]; then
  ARGS+=(--offset "${OFFSET}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${INDICES_FILE:-}" ]]; then
  ARGS+=(--indices-file "${INDICES_FILE}")
fi

python3 "${SCRIPT_DIR}/convert_vqa_run_batch.py" "${ARGS[@]}"
