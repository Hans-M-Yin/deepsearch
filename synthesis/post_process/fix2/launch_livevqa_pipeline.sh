#!/usr/bin/env bash
# Launch direct LiveVQA teacher collection.  This intentionally has no base
# inference, base answer judge, or failure router.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash synthesis/post_process/fix2/launch_livevqa_pipeline.sh

The source must be the train split of ONE-Lab/LiveVQA-new. Build it once if
needed:
  python scripts/build_benchmark_dataset.py --benchmark livevqa --split train

Useful optional environment:
  SOURCE_PARQUET        default: ../data/benchmarks/livevqa/train_random10k.parquet
  SAMPLE_COUNT          default: 512; must be divisible by 4 (L1:L2 = 1:3)
  SAMPLE_SEED           default: 20260919
  EXPAND_SOURCE_MANIFEST default: 1; append a larger same-seed sample on resume
  DOWNLOAD_WORKERS      default: 16
  RUN_DIR               default: runs/fix2_livevqa_<UTC timestamp>
  LOCAL_STATE_DIR       default: /tmp/$USER/fix2_livevqa_<UTC timestamp>
  TEACHER_WORKERS       default: 16
  TEACHER_MODEL         default: gpt54_2_internal_azure_responses
  TEACHER_MAX_TURNS     default: 30
  STYLE_EXAMPLES_FILE   default: reviewed production legacy anchors in fix2/
  REQUIRE_STYLE_EXAMPLES default: 1; set to 0 only for a disposable smoke run
  FOLLOW_LOGS           default: 1

The SQLite queue and tool scratch must remain on this coordinator's local
disk. Immutable manifests/images/results are stored under RUN_DIR.
EOF
}

die() {
    echo "[fix2-livevqa-launch] error: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "required file does not exist: $1"
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

pid_is_running() {
    local pid_file="$1"
    [[ -s "$pid_file" ]] || return 1
    local pid
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_process() {
    local name="$1"
    shift
    local pid_file="$RUN_DIR/pids/${name}.pid"
    local log_file="$LOCAL_STATE_DIR/logs/${name}.log"
    if pid_is_running "$pid_file"; then
        die "${name} is already running (pid $(<"$pid_file"))"
    fi
    mkdir -p "$(dirname "$log_file")"
    rm -f "$pid_file"
    nohup "$@" >"$log_file" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" >"$pid_file"
    echo "[fix2-livevqa-launch] started ${name}: pid=${pid}, log=${log_file}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || die "unsupported argument: $1 (use environment variables; see --help)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SOURCE_PARQUET="${SOURCE_PARQUET:-$REPO_ROOT/../data/benchmarks/livevqa/train_random10k.parquet}"
LEGACY_SHAREGPT="${LEGACY_SHAREGPT:-$REPO_ROOT/data/sharegpt_dataset_refined_gpt54_only_17k_final/trajectories_sharegpt.json}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/fix2_livevqa_${RUN_TAG}}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_livevqa_${RUN_TAG}}"
SAMPLE_COUNT="${SAMPLE_COUNT:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-20260919}"
EXPAND_SOURCE_MANIFEST="${EXPAND_SOURCE_MANIFEST:-1}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
DOWNLOAD_TIMEOUT_S="${DOWNLOAD_TIMEOUT_S:-45}"
MAX_IMAGE_BYTES="${MAX_IMAGE_BYTES:-33554432}"
ARCHIVE_CACHE_DIR="${ARCHIVE_CACHE_DIR:-$LOCAL_STATE_DIR/livevqa_archives}"
TEACHER_WORKERS="${TEACHER_WORKERS:-16}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt54_2_internal_azure_responses}"
TEACHER_API_MODE="${TEACHER_API_MODE:-responses}"
TEACHER_MAX_TURNS="${TEACHER_MAX_TURNS:-30}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-8192}"
TEACHER_TIMEOUT_S="${TEACHER_TIMEOUT_S:-600}"
TEACHER_PRINT_ROUNDS="${TEACHER_PRINT_ROUNDS:-1}"
STYLE_EXAMPLES_FILE="${STYLE_EXAMPLES_FILE:-$REPO_ROOT/synthesis/post_process/fix2/style_examples.production.legacy.json}"
REQUIRE_STYLE_EXAMPLES="${REQUIRE_STYLE_EXAMPLES:-1}"
FOLLOW_LOGS="${FOLLOW_LOGS:-1}"
QUEUE_NAME="${QUEUE_NAME:-teacher_collect_livevqa_responses}"

for value in "$SAMPLE_COUNT" "$DOWNLOAD_WORKERS" "$TEACHER_WORKERS" "$TEACHER_MAX_TURNS" "$TEACHER_MAX_TOKENS" "$TEACHER_TIMEOUT_S" "$MAX_IMAGE_BYTES"; do
    is_positive_integer "$value" || die "expected a positive integer, got: $value"
done
[[ "$SAMPLE_SEED" =~ ^[0-9]+$ ]] || die "SAMPLE_SEED must be a nonnegative integer"
[[ "$DOWNLOAD_TIMEOUT_S" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || die "DOWNLOAD_TIMEOUT_S must be positive"
(( SAMPLE_COUNT % 4 == 0 )) || die "SAMPLE_COUNT must be divisible by 4 for exact L1:L2=1:3 sampling"
[[ "$EXPAND_SOURCE_MANIFEST" == "0" || "$EXPAND_SOURCE_MANIFEST" == "1" ]] || die "EXPAND_SOURCE_MANIFEST must be 0 or 1"
[[ "$REQUIRE_STYLE_EXAMPLES" == "0" || "$REQUIRE_STYLE_EXAMPLES" == "1" ]] || die "REQUIRE_STYLE_EXAMPLES must be 0 or 1"
[[ "$TEACHER_PRINT_ROUNDS" == "0" || "$TEACHER_PRINT_ROUNDS" == "1" ]] || die "TEACHER_PRINT_ROUNDS must be 0 or 1"
[[ "$FOLLOW_LOGS" == "0" || "$FOLLOW_LOGS" == "1" ]] || die "FOLLOW_LOGS must be 0 or 1"
case "$TEACHER_API_MODE" in
    manual_react|native_tools|responses) ;;
    *) die "TEACHER_API_MODE must be manual_react, native_tools, or responses" ;;
esac

require_file "$SOURCE_PARQUET"
require_file "$LEGACY_SHAREGPT"
if [[ "$REQUIRE_STYLE_EXAMPLES" == "1" ]]; then
    require_file "$STYLE_EXAMPLES_FILE"
    grep -Fq 'Replace this with a sanitized' "$STYLE_EXAMPLES_FILE" && die "STYLE_EXAMPLES_FILE still contains template placeholders"
fi

mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/pids" "$LOCAL_STATE_DIR"
LOCAL_STATE_DIR="$(cd "$LOCAL_STATE_DIR" && pwd)"
case "$LOCAL_STATE_DIR" in
    /mnt/hdfs/*|*/hdfs/*) die "LOCAL_STATE_DIR must be local, not HDFS: $LOCAL_STATE_DIR" ;;
esac
export PYTHONUNBUFFERED=1

PRIVATE_CASES="$RUN_DIR/cases_private.jsonl"
PUBLIC_CASES="$RUN_DIR/cases_public.jsonl"
SELECTION_REPORT="$RUN_DIR/livevqa_selection_report.json"
IMAGE_DIR="$RUN_DIR/source_images"
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"
RESULTS_DIR="$RUN_DIR/teacher_results"
TEACHER_WORKDIR="$LOCAL_STATE_DIR/teacher_runtime"
LEVEL1_COUNT=$(( SAMPLE_COUNT / 4 ))

if [[ -f "$PUBLIC_CASES" && ! -f "$PRIVATE_CASES" ]]; then
    die "public manifest exists but private manifest is missing: $PUBLIC_CASES"
fi
if [[ ! -f "$PRIVATE_CASES" || ! -f "$PUBLIC_CASES" || "$EXPAND_SOURCE_MANIFEST" == "1" ]]; then
    prepare_command=(
        python -m synthesis.post_process.fix2.run prepare-livevqa
        --input-parquet "$SOURCE_PARQUET"
        --output-cases "$PRIVATE_CASES"
        --output-public-cases "$PUBLIC_CASES"
        --selection-report "$SELECTION_REPORT"
        --image-dir "$IMAGE_DIR"
        --level1-count "$LEVEL1_COUNT"
        --seed "$SAMPLE_SEED"
        --download-workers "$DOWNLOAD_WORKERS"
        --download-timeout-s "$DOWNLOAD_TIMEOUT_S"
        --max-image-bytes "$MAX_IMAGE_BYTES"
        --archive-cache-dir "$ARCHIVE_CACHE_DIR"
        --progress
    )
    if [[ -f "$PRIVATE_CASES" || -f "$PUBLIC_CASES" ]]; then
        prepare_command+=(--resume)
        if [[ "$EXPAND_SOURCE_MANIFEST" == "1" ]]; then
            echo "[fix2-livevqa-launch] resuming and expanding LiveVQA manifest through total=$SAMPLE_COUNT"
        else
            echo "[fix2-livevqa-launch] resuming LiveVQA source materialization"
        fi
    else
        echo "[fix2-livevqa-launch] sampling LiveVQA train directly (no base-model filter)"
    fi
    "${prepare_command[@]}"
else
    echo "[fix2-livevqa-launch] reusing completed manifests: $PRIVATE_CASES"
fi

python -m synthesis.post_process.fix2.run profile-style \
    --input-sharegpt "$LEGACY_SHAREGPT" \
    --output-profile "$RUN_DIR/legacy_style_profile.json" \
    --limit 500

echo "[fix2-livevqa-launch] enqueueing sampled public cases directly to teacher queue"
python -m synthesis.post_process.fix2.run enqueue-public-cases \
    --public-cases "$PUBLIC_CASES" \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --max-attempts 3

teacher_command=(
    python -m synthesis.post_process.fix2.run teacher-workers
    --queue-db "$QUEUE_DB"
    --queue-name "$QUEUE_NAME"
    --results-dir "$RESULTS_DIR"
    --workdir "$TEACHER_WORKDIR"
    --model "$TEACHER_MODEL"
    --api-mode "$TEACHER_API_MODE"
    --workers "$TEACHER_WORKERS"
    --idle-exit-s 0
    --lease-s 3600
    --max-turns "$TEACHER_MAX_TURNS"
    --max-tokens "$TEACHER_MAX_TOKENS"
    --timeout-s "$TEACHER_TIMEOUT_S"
    --debug
)
if [[ "$REQUIRE_STYLE_EXAMPLES" == "1" ]]; then
    teacher_command+=(--style-examples-file "$STYLE_EXAMPLES_FILE")
fi
if [[ "$TEACHER_PRINT_ROUNDS" == "1" ]]; then
    teacher_command+=(--print-rounds)
fi

echo "[fix2-livevqa-launch] configuration"
echo "  source parquet:       $SOURCE_PARQUET"
echo "  run directory:        $RUN_DIR"
echo "  local state:          $LOCAL_STATE_DIR"
echo "  direct sample:        total=$SAMPLE_COUNT (L1=$LEVEL1_COUNT, L2=$((LEVEL1_COUNT * 3)), seed=$SAMPLE_SEED)"
echo "  expand manifest:      $EXPAND_SOURCE_MANIFEST"
echo "  source dedup:         exact image SHA256 only"
echo "  download workers:     $DOWNLOAD_WORKERS"
echo "  local archive cache:  $ARCHIVE_CACHE_DIR (source ZIPs are staged lazily)"
echo "  teacher:              workers=$TEACHER_WORKERS model=$TEACHER_MODEL api_mode=$TEACHER_API_MODE max_turns=$TEACHER_MAX_TURNS"
echo "  queue:                $QUEUE_NAME"

teacher_pid_file="$RUN_DIR/pids/teacher_workers.pid"
if pid_is_running "$teacher_pid_file"; then
    # Workers deliberately use idle-exit=0 so an already-running group can
    # consume cases appended by a same-run expansion without being restarted.
    # This also keeps active leases intact if SAMPLE_COUNT is raised mid-run.
    echo "[fix2-livevqa-launch] reusing active teacher_workers (pid $(<"$teacher_pid_file"))"
else
    start_process teacher_workers "${teacher_command[@]}"
fi

echo
echo "[fix2-livevqa-launch] direct-teacher pipeline launched"
echo "  queue status: python -m synthesis.post_process.fix2.run queue-stats --queue-db '$QUEUE_DB' --queue-name '$QUEUE_NAME'"
echo "  after teacher completion: QUEUE_NAME='$QUEUE_NAME' bash synthesis/post_process/fix2/finalize_livevqa_pipeline.sh '$RUN_DIR' '$LOCAL_STATE_DIR'"

if [[ "$FOLLOW_LOGS" == "1" ]]; then
    echo "[fix2-livevqa-launch] following teacher log; Ctrl-C stops only log following."
    tail -n 0 -F "$LOCAL_STATE_DIR/logs/teacher_workers.log"
fi
