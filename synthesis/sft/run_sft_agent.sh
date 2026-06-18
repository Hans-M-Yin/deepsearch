#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.sft_env"

usage() {
  cat <<'EOF'
Usage:
  bash synthesis/sft/run_sft_agent.sh --prompt "你的问题"
  bash synthesis/sft/run_sft_agent.sh --messages-file /abs/path/messages.json

Optional:
  --image /abs/path/to/image.png      Preload a local image as img_n
  --image-url https://...             Preload a remote image as img_n
  --workdir /abs/path/to/output_dir   Override runtime working directory
  --gpt54                             Use the GPT-5.4 Responses-API branch
  --verbose                           Enable verbose logging
  --office-net                        Switch endpoint to tiktok-row office domain
  --help                              Show this help

Examples:
  bash synthesis/sft/run_sft_agent.sh \
    --prompt "请搜索上海最近的AI新闻，并给出处"

  bash synthesis/sft/run_sft_agent.sh \
    --messages-file /tmp/messages.json \
    --image /abs/path/to/example.png

messages.json example:
[
  {
    "role": "system",
    "content": "You are a deep research assistant."
  },
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "请看这张图并在必要时调用工具。"},
      {"type": "image_path", "path": "/absolute/path/to/image.png"}
    ]
  }
]
EOF
}

require_env() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required environment variable: ${var_name}" >&2
    exit 1
  fi
}

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

PROMPT=""
MESSAGES_FILE=""
WORKDIR=""
VERBOSE=0
OFFICE_NET=0
GPT54=0

PY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --messages-file)
      MESSAGES_FILE="${2:-}"
      shift 2
      ;;
    --image|--image-url)
      PY_ARGS+=("$1" "${2:-}")
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:-}"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    --gpt54)
      GPT54=1
      shift
      ;;
    --office-net)
      OFFICE_NET=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${PROMPT}" && -n "${MESSAGES_FILE}" ]]; then
  echo "Use only one of --prompt or --messages-file." >&2
  exit 1
fi

if [[ -z "${PROMPT}" && -z "${MESSAGES_FILE}" ]]; then
  echo "One of --prompt or --messages-file is required." >&2
  usage
  exit 1
fi

if [[ "${OFFICE_NET}" -eq 1 ]]; then
  if [[ "${GPT54}" -eq 1 ]]; then
    export SFT_GPT54_AZURE_ENDPOINT="https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/responses"
  else
    export SFT_OPENAI_AZURE_ENDPOINT="https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl"
    export SFT_OPENAI_BASE_URL="${SFT_OPENAI_AZURE_ENDPOINT}"
  fi
fi

if [[ "${GPT54}" -eq 1 ]]; then
  require_env "SFT_GPT54_MODEL"
  require_env "SFT_GPT54_AZURE_ENDPOINT"
  require_env "SFT_GPT54_API_VERSION"
  require_env "SFT_GPT54_API_KEY"
  PY_ARGS+=(--gpt54)
else
  require_env "SFT_OPENAI_MODEL"
  require_env "SFT_OPENAI_AZURE_ENDPOINT"
  require_env "SFT_OPENAI_API_VERSION"
  require_env "OPENAI_API_KEY"
fi

if [[ -n "${WORKDIR}" ]]; then
  PY_ARGS+=(--workdir "${WORKDIR}")
fi

if [[ "${VERBOSE}" -eq 1 ]]; then
  PY_ARGS+=(--verbose)
fi

if [[ -n "${PROMPT}" ]]; then
  PY_ARGS+=(--prompt "${PROMPT}")
else
  if [[ ! -f "${MESSAGES_FILE}" ]]; then
    echo "messages file not found: ${MESSAGES_FILE}" >&2
    exit 1
  fi
  PY_ARGS+=(--messages-file "${MESSAGES_FILE}")
fi

cd "${PROJECT_ROOT}"

echo "=== SFT Agent Config ==="
if [[ "${GPT54}" -eq 1 ]]; then
  echo "api_mode: responses"
  echo "model: ${SFT_GPT54_MODEL}"
  echo "azure_endpoint: ${SFT_GPT54_AZURE_ENDPOINT}"
  echo "api_version: ${SFT_GPT54_API_VERSION}"
else
  echo "api_mode: chat_completions"
  echo "model: ${SFT_OPENAI_MODEL}"
  echo "azure_endpoint: ${SFT_OPENAI_AZURE_ENDPOINT}"
  echo "api_version: ${SFT_OPENAI_API_VERSION}"
fi
echo "reader: ${ENHANCED_READER_URL:-${JINA_READER_URL:-<unset>}}"
echo "summarizer_base: ${SFT_SUMMARIZER_API_BASE:-${QWEN_API_BASE:-<unset>}}"
echo "summarizer_model: ${SFT_SUMMARIZER_MODEL:-${QWEN_MODEL_NAME:-<unset>}}"
echo

python -m synthesis.sft.api_tools "${PY_ARGS[@]}"
