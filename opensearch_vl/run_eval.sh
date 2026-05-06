#!/bin/bash
# Evaluation harness for BrowseComp-VL, HLE and VDR-Bench using a GPT-4o judge.
#
# Configure the trajectory directories (and optional VDR answer parquet)
# through environment variables before running:
#
#   TRAJ_BC_VL_LEVEL1   - directory holding BrowseComp-VL Level 1 trajectories
#   TRAJ_BC_VL_LEVEL2   - directory holding BrowseComp-VL Level 2 trajectories
#   TRAJ_HLE            - directory holding HLE trajectories
#   TRAJ_VDR_PRIMARY    - directory holding VDR-Bench trajectories (primary run)
#   TRAJ_VDR_SECONDARY  - directory holding VDR-Bench trajectories (second run)
#   VDR_ANSWER_PARQUET  - parquet file with id/answer columns for VDR-Bench
#
# Optional flags:
#   --limit N    - evaluate only the first N trajectories (default: all)
#   --workers N  - judge concurrency (default: 20)
#
# Required env vars for the GPT-4o judge:
#   JUDGE_API_BASE_URL / JUDGE_APP_ID / JUDGE_APP_KEY (see eval_with_gpt4o.py)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_with_gpt4o.py"

LIMIT=${LIMIT:-0}
WORKERS=${WORKERS:-20}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)   LIMIT="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_eval() {
    local label="$1"
    local traj_dir="$2"
    local benchmark="$3"
    local extra_args="${4:-}"

    if [[ -z "${traj_dir}" ]]; then
        echo ">>> [skip] ${label}: trajectory directory env var is not set"
        return 0
    fi
    if [[ ! -d "${traj_dir}" ]]; then
        echo ">>> [skip] ${label}: ${traj_dir} does not exist"
        return 0
    fi

    echo ""
    echo ">>> ${label} -> ${traj_dir}"
    # shellcheck disable=SC2086
    python3 "${EVAL_SCRIPT}" \
        --traj_dir "${traj_dir}" \
        --benchmark "${benchmark}" \
        --max_workers "${WORKERS}" \
        --limit "${LIMIT}" \
        ${extra_args}
}

echo "============================================"
echo "  Eval Config"
echo "  Limit:   ${LIMIT} (0 = all)"
echo "  Workers: ${WORKERS}"
echo "============================================"

run_eval "BrowseComp-VL Level 1"  "${TRAJ_BC_VL_LEVEL1:-}" bc_vl
run_eval "BrowseComp-VL Level 2"  "${TRAJ_BC_VL_LEVEL2:-}" bc_vl
run_eval "HLE"                    "${TRAJ_HLE:-}"          hle

VDR_EXTRA=""
if [[ -n "${VDR_ANSWER_PARQUET:-}" ]]; then
    VDR_EXTRA="--answer_file ${VDR_ANSWER_PARQUET}"
fi
run_eval "VDR-Bench primary"   "${TRAJ_VDR_PRIMARY:-}"   vdr "${VDR_EXTRA}"
run_eval "VDR-Bench secondary" "${TRAJ_VDR_SECONDARY:-}" vdr "${VDR_EXTRA}"

echo ""
echo "============================================"
echo "  All evaluations complete."
echo "============================================"
