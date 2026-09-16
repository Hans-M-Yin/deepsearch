#!/usr/bin/env bash
# Download the immutable local checkpoint used by the similarity-reward service.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE="${EMBEDDING_MODEL_SIZE:-0.6B}"
case "${MODEL_SIZE,,}" in
  0.6b|0p6b)
    MODEL_LABEL="0.6B"
    DEFAULT_MODEL_ID="Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_MODEL_SUBDIR="qwen3-embedding-0p6b"
    ;;
  4b)
    MODEL_LABEL="4B"
    DEFAULT_MODEL_ID="Qwen/Qwen3-Embedding-4B"
    DEFAULT_MODEL_SUBDIR="qwen3-embedding-4b"
    ;;
  *)
    echo "EMBEDDING_MODEL_SIZE must be 0.6B or 4B, got: ${MODEL_SIZE}" >&2
    exit 1
    ;;
esac
MODEL_ROOT="${EMBEDDING_MODEL_ROOT:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models}"
MODEL_ID="${EMBEDDING_MODEL_ID:-${DEFAULT_MODEL_ID}}"
MODEL_DIR="${EMBEDDING_MODEL_DIR:-${MODEL_ROOT}/${DEFAULT_MODEL_SUBDIR}}"
MAX_WORKERS="${EMBEDDING_DOWNLOAD_MAX_WORKERS:-1}"

if ! command -v huggingface-cli >/dev/null 2>&1 && ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install with: python -m pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"
echo "[embedding-download] size=${MODEL_LABEL} model=${MODEL_ID}"
echo "[embedding-download] destination=${MODEL_DIR}"
echo "[embedding-download] max_workers=${MAX_WORKERS}"

if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}" --max-workers "${MAX_WORKERS}"
else
  huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}" --max-workers "${MAX_WORKERS}"
fi

test -f "${MODEL_DIR}/config.json"
if [[ -f "${MODEL_DIR}/model.safetensors" ]]; then
  :  # Single-file checkpoint (Qwen3-Embedding-0.6B).
elif [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] \
  && compgen -G "${MODEL_DIR}/model-*.safetensors" >/dev/null; then
  :  # Sharded checkpoint (Qwen3-Embedding-4B).
else
  echo "Downloaded checkpoint has no complete safetensors weights: ${MODEL_DIR}" >&2
  exit 1
fi
echo "[embedding-download] complete"
