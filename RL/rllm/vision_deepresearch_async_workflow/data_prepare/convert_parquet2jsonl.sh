#!/usr/bin/env bash
set -e

# Convert a HuggingFace-style parquet shard (with embedded image bytes) into
# a JSONL file with extracted image files. Override DATA_ROOT to point at
# your downloaded copy of Vision-DeepResearch-RL-Data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT=${DATA_ROOT:-data/Vision-DeepResearch-RL-Data}

python3 convert_parquet2jsonl.py \
    --parquet_path "${DATA_ROOT}/vision-deepresearch_RL_Demo_1k.parquet" \
    --output_jsonl "${DATA_ROOT}/vision-deepresearch_RL_Demo_1k.jsonl" \
    --image_dir    "${DATA_ROOT}/image"
