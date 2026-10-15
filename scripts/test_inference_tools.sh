#!/usr/bin/env bash
# GPT5 uses native Responses tools; Gemini-2.5-Pro uses manual ReAct tools.
# Two models run concurrently: SearchWeave -> VisBrowse -> BrowseComp-V3.
# Formal mode (default): full datasets, two independent trials per model.
# Launch: nohup bash scripts/test_inference_tools.sh > /tmp/tools_full.log 2>&1 &
# Preview: DRY_RUN=1 bash scripts/test_inference_tools.sh
# Resume: RUN_TAG=<printed tag> bash scripts/test_inference_tools.sh
# Continue the existing first full SearchWeave trial and add the missing trials:
# RUN_TAG=20261014T151136Z_638441 bash scripts/test_inference_tools.sh
# Smoke test only: MODE=smoke bash scripts/test_inference_tools.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AGENT_TRACE_LOGS="${AGENT_TRACE_LOGS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/opensearch_infer/outputs}"
DRY_RUN="${DRY_RUN:-0}"
MODE="${MODE:-full}"
case "$MODE" in
    full) default_limit=0; default_repeats=2 ;;
    smoke) default_limit=5; default_repeats=1 ;;
    *) echo "MODE must be full or smoke." >&2; exit 2 ;;
esac
LIMIT="${LIMIT:-$default_limit}"  # 0 processes the full dataset.
REPEATS="${REPEATS:-$default_repeats}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
API_TIMEOUT="${API_TIMEOUT:-600}"
API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
SEARCHWEAVE_DATA="${SEARCHWEAVE_DATA:-$PROJECT_ROOT/../data/benchmarks/searchweave/annotation_benchmark_300_en.parquet}"
VISBROWSE_DATA="${VISBROWSE_DATA:-$PROJECT_ROOT/../data/benchmarks/visbrowse_bench/train.parquet}"
BROWSECOMP_V3_DATA="${BROWSECOMP_V3_DATA:-$PROJECT_ROOT/../data/benchmarks/browsecomp_v3/train.parquet}"

BENCHMARKS=(searchweave)
DATA_PATHS=("$SEARCHWEAVE_DATA")
if [[ "$MODE" == full ]]; then
    BENCHMARKS+=(visbrowse browsecomp_v3)
    DATA_PATHS+=("$VISBROWSE_DATA" "$BROWSECOMP_V3_DATA")
fi

if [[ ! "$RUN_TAG" =~ ^[[:alnum:]_.-]+$ ]]; then
    echo "RUN_TAG must contain only letters, numbers, dots, underscores or hyphens." >&2
    exit 2
fi
if [[ ! "$LIMIT" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "LIMIT must be a nonnegative integer (0 means the full dataset)." >&2
    exit 2
fi
if [[ ! "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS must be a positive integer." >&2
    exit 2
fi
for data_path in "${DATA_PATHS[@]}"; do
    if [[ ! -f "$data_path" ]]; then
        echo "Dataset not found: $data_path" >&2
        exit 2
    fi
done

MODEL_NAMES=(GPT5 Gemini-2.5-Pro)
MODEL_ALIASES=("${GPT5_MODEL_ALIAS:-gpt5}" "${GEMINI25PRO_MODEL_ALIAS:-gemini25pro_internal_azure}")
MODEL_WORKERS=("${GPT5_WORKERS:-10}" "${GEMINI25PRO_WORKERS:-10}")
MODEL_ENTRYPOINTS=(opensearch_vl/run_infer_responses.py opensearch_vl/run_infer_react_online.py)
MODEL_API_MODES=(responses manual_react)
stage_pids=()

stop_stage() {
    trap - INT TERM
    for pid in "${stage_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${stage_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_stage INT TERM

run_model() (
    local model_index="$1" benchmark="$2" data_path="$3"
    local model="${MODEL_NAMES[$model_index]}" alias="${MODEL_ALIASES[$model_index]}"
    local entrypoint="${MODEL_ENTRYPOINTS[$model_index]}" api_mode="${MODEL_API_MODES[$model_index]}"
    local workers="${MODEL_WORKERS[$model_index]}" repeat output_dir child_pid=""
    trap 'if [[ -n "$child_pid" ]]; then kill "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi; exit 130' INT TERM
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        output_dir="$OUTPUT_ROOT/$model/${benchmark}_tools_${api_mode}_${RUN_TAG}_limit${LIMIT}_run${repeat}"
        command=("$PYTHON_BIN" -u "$entrypoint"
            --model-alias "$alias" --data-path "$data_path" --output-dir "$output_dir"
            --dataset test --start 0 --parallel-workers "$workers"
            --max-tokens "$MAX_TOKENS" --api-timeout "$API_TIMEOUT"
            --api-max-retries "$API_MAX_RETRIES" --log-level INFO)
        if (( LIMIT > 0 )); then
            command+=(--limit "$LIMIT")
        fi
        printf '[start] model=%s api=%s benchmark=%s repeat=%s limit=%s output=%s\n' "$model" "$api_mode" "$benchmark" "$repeat" "$LIMIT" "$output_dir"
        if [[ "$DRY_RUN" == 1 ]]; then
            printf 'COMMAND '
            printf '%q ' "${command[@]}"
            printf '\n'
            continue
        fi
        mkdir -p "$output_dir"
        printf '%q ' "${command[@]}" >> "$output_dir/launch_commands.log"
        printf '\n' >> "$output_dir/launch_commands.log"
        "${command[@]}" >> "$output_dir/inference.log" 2>&1 &
        child_pid=$!
        if wait "$child_pid"; then
            child_pid=""
        else
            local status=$?
            child_pid=""
            printf '[failed] model=%s benchmark=%s repeat=%s exit=%s log=%s\n' "$model" "$benchmark" "$repeat" "$status" "$output_dir/inference.log" >&2
            return "$status"
        fi
        printf '[finished] model=%s benchmark=%s repeat=%s\n' "$model" "$benchmark" "$repeat"
    done
)

printf 'MODE=%s\nRUN_TAG=%s\nOUTPUT_ROOT=%s\nREPEATS=%s\nLIMIT=%s (per model, per benchmark, per repeat; 0=full)\n' "$MODE" "$RUN_TAG" "$OUTPUT_ROOT" "$REPEATS" "$LIMIT"
for benchmark_index in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$benchmark_index]}"
    data_path="${DATA_PATHS[$benchmark_index]}"
    printf '[stage] %s\n' "$benchmark"
    stage_pids=()
    if [[ "$DRY_RUN" == 1 ]]; then
        for model_index in "${!MODEL_NAMES[@]}"; do
            run_model "$model_index" "$benchmark" "$data_path"
        done
        continue
    fi
    for model_index in "${!MODEL_NAMES[@]}"; do
        run_model "$model_index" "$benchmark" "$data_path" &
        stage_pids+=("$!")
    done
    stage_failed=0
    for pid in "${stage_pids[@]}"; do
        if ! wait "$pid"; then
            stage_failed=1
        fi
    done
    stage_pids=()
    if (( stage_failed )); then
        echo "Stage $benchmark failed. Resume with RUN_TAG=$RUN_TAG after fixing the error." >&2
        exit 1
    fi
    printf '[stage finished] %s\n' "$benchmark"
done
printf '[batch finished] RUN_TAG=%s\n' "$RUN_TAG"
