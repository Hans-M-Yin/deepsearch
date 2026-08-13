#!/usr/bin/env bash
set -e

# Register a normalized JSONL dataset into rLLM's DatasetRegistry. The JSONL
# can be produced by convert_vqa_run_batch.py and should contain at least
# ``question``, ``answer`` and ``images``. Additional fields are preserved and
# become available through the runtime task's extra_info.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JSONL_PATH=${JSONL_PATH:-data/vision-deepresearch_RL_Demo_1k.jsonl}
REGISTER_NAME=${REGISTER_NAME:-Vision-DeepResearch-QA}
TRAIN_RATIO=${TRAIN_RATIO:-0.9}
RANDOM_SEED=${RANDOM_SEED:-42}
VAL_SPLIT=${VAL_SPLIT:-test}

python3 register_rl_dataset.py \
    --jsonl_path "${JSONL_PATH}" \
    --register_name "${REGISTER_NAME}" \
    --train_ratio "${TRAIN_RATIO}" \
    --random_seed "${RANDOM_SEED}" \
    --val_split "${VAL_SPLIT}"
