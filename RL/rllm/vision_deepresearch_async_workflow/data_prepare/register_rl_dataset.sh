#!/usr/bin/env bash
set -e

# Register a JSONL dataset into rLLM's DatasetRegistry under the name
# "Vision-DeepResearch-QA". The JSONL file should be produced by
# ``convert_parquet2jsonl.py`` (one sample per line, with at least the fields
# ``question``, ``answer`` and ``images``).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JSONL_PATH=${JSONL_PATH:-data/vision-deepresearch_RL_Demo_1k.jsonl}

python3 register_rl_dataset.py \
    --jsonl_path "${JSONL_PATH}" \
    --register_name Vision-DeepResearch-QA \
    --train_ratio 0.9 \
    --random_seed 42