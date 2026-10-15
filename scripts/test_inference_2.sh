#!/usr/bin/env bash
# Resume the four existing BrowseComp-V3 tools runs. Successful trajectories
# are skipped automatically; missing/failed cases are retried in the same dirs.
# GPT-5 and Gemini run concurrently; each model's two repeats run sequentially.
# Launch: nohup bash scripts/test_inference_2.sh > /tmp/browsecomp_resume_all.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${BROWSECOMP_V3_DATA:-$PROJECT_ROOT/../data/benchmarks/browsecomp_v3/train.parquet}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
API_TIMEOUT="${API_TIMEOUT:-600}"
API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
GPT5_WORKERS="${GPT5_WORKERS:-2}"
GEMINI25PRO_WORKERS="${GEMINI25PRO_WORKERS:-2}"

GPT5_ROOT="$PROJECT_ROOT/opensearch_infer/outputs/GPT5"
GEMINI_ROOT="$PROJECT_ROOT/opensearch_infer/outputs/Gemini-2.5-Pro"
GPT5_TAG="browsecomp_v3_tools_responses_20261014T151136Z_638441_limit0"
GEMINI_TAG="browsecomp_v3_tools_manual_react_20261014T151136Z_638441_limit0"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "BrowseComp-V3 dataset not found: $DATA_PATH" >&2
    exit 2
fi

run_one() {
    local model="$1" alias="$2" entrypoint="$3" output_dir="$4" workers="$5" logfile="$6"
    local -a command=("$PYTHON_BIN" -u "$entrypoint"
        --model-alias "$alias" --data-path "$DATA_PATH" --output-dir "$output_dir"
        --dataset test --start 0 --parallel-workers "$workers"
        --max-tokens "$MAX_TOKENS" --api-timeout "$API_TIMEOUT"
        --api-max-retries "$API_MAX_RETRIES" --log-level INFO)

    mkdir -p "$output_dir"
    printf '[start] model=%s output=%s workers=%s log=%s\n' "$model" "$output_dir" "$workers" "$logfile"
    printf '%q ' "${command[@]}" >> "$output_dir/launch_commands.log"
    printf '\n' >> "$output_dir/launch_commands.log"
    "${command[@]}" >> "$logfile" 2>&1
    printf '[finished] model=%s output=%s\n' "$model" "$output_dir"
}

run_gpt5() {
    run_one GPT5 gpt5 opensearch_vl/run_infer_responses.py \
        "$GPT5_ROOT/${GPT5_TAG}_run1" "$GPT5_WORKERS" /tmp/gpt5_browsecomp_run1_resume.log
    run_one GPT5 gpt5 opensearch_vl/run_infer_responses.py \
        "$GPT5_ROOT/${GPT5_TAG}_run2" "$GPT5_WORKERS" /tmp/gpt5_browsecomp_run2_resume.log
}

run_gemini() {
    run_one Gemini-2.5-Pro gemini25pro_internal_azure opensearch_vl/run_infer_react_online.py \
        "$GEMINI_ROOT/${GEMINI_TAG}_run1" "$GEMINI25PRO_WORKERS" /tmp/gemini_browsecomp_run1_resume.log
    run_one Gemini-2.5-Pro gemini25pro_internal_azure opensearch_vl/run_infer_react_online.py \
        "$GEMINI_ROOT/${GEMINI_TAG}_run2" "$GEMINI25PRO_WORKERS" /tmp/gemini_browsecomp_run2_resume.log
}

printf 'Resuming BrowseComp-V3 runs; GPT5_WORKERS=%s GEMINI25PRO_WORKERS=%s\n' \
    "$GPT5_WORKERS" "$GEMINI25PRO_WORKERS"
run_gpt5 &
gpt5_pid=$!
run_gemini &
gemini_pid=$!

status=0
wait "$gpt5_pid" || status=1
wait "$gemini_pid" || status=1
exit "$status"
