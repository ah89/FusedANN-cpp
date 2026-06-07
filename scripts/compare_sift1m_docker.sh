#!/usr/bin/env bash
# Run fusedann locally and compare it against the official NeurIPS'23 filter-track
# baselines executed through the big-ann-benchmarks Docker harness.
set -euo pipefail

function usage() {
  cat <<'EOF'
Usage: compare_sift1m_docker.sh [options]

Options:
  --algos LIST         Comma or space separated list of NeurIPS'23 filter algorithms
                       to run via Docker (default: faiss,faissplus,parlayivf).
  --runs N             Number of benchmark repetitions to let big-ann pick the best
                       measurement from (default: 1).
  --count K            Override the recall@K target passed to run.py (default: dataset default).
  --fused-backend B    fusedann backend to run before comparisons: efanna | parlay | skip
                       (default: parlay).
  --force-opponents    Re-run Docker opponents even if a matching results file already exists.
  --skip-plot          Skip the final plot.py invocation.
  --help               Show this message.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BIGANN_DIR="${ROOT_DIR}/third_party/big-ann-benchmarks"
BIGANN_PATCH="${SCRIPT_DIR}/bigann_sift1m_dataset.patch"
DATASET="sift1m-filter"
TRACK="filter"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

ALGO_ARG="faiss,faissplus,parlayivf"
RUNS=1
COUNT=""
FUSED_BACKEND="parlay"
FORCE_FLAG=""
RUN_PLOT=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algos)
      ALGO_ARG=${2:-}
      if [[ -z "${ALGO_ARG}" ]]; then
        echo "--algos requires a value" >&2
        usage
        exit 1
      fi
      shift 2
      ;;
    --runs)
      RUNS=${2:-}
      if ! [[ ${RUNS} =~ ^[0-9]+$ ]] || [[ ${RUNS} -le 0 ]]; then
        echo "--runs expects a positive integer" >&2
        exit 1
      fi
      shift 2
      ;;
    --count|-k)
      COUNT=${2:-}
      if ! [[ ${COUNT} =~ ^[0-9]+$ ]] || [[ ${COUNT} -le 0 ]]; then
        echo "--count expects a positive integer" >&2
        exit 1
      fi
      shift 2
      ;;
    --fused-backend)
      FUSED_BACKEND=${2:-}
      if [[ -z "${FUSED_BACKEND}" ]]; then
        echo "--fused-backend requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --force-opponents)
      FORCE_FLAG="--force"
      shift
      ;;
    --skip-plot)
      RUN_PLOT=false
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${BIGANN_DIR}" ]]; then
  echo "Expected big-ann-benchmarks checkout at ${BIGANN_DIR}." >&2
  exit 1
fi

function require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command '$1'." >&2
    exit 1
  fi
}

require_command python3
require_command docker
require_command bash

IFS=' ,'
read -r -a ALGO_LIST <<< "${ALGO_ARG}"
IFS=$' \n\t'
if [[ ${#ALGO_LIST[@]} -eq 0 ]]; then
  echo "No algorithms provided via --algos." >&2
  exit 1
fi

case "${FUSED_BACKEND}" in
  efanna|parlay)
    RUN_FUSED=true
    ;;
  skip|none)
    RUN_FUSED=false
    ;;
  *)
    echo "Unknown --fused-backend value '${FUSED_BACKEND}'. Use efanna|parlay|skip." >&2
    exit 1
    ;;
 esac

if ${RUN_FUSED}; then
  echo "[1/5] Running fusedann pipeline (${FUSED_BACKEND})"
  FUSED_LOG="${LOG_DIR}/fusedann_${FUSED_BACKEND}_$(date +%Y%m%d_%H%M%S).log"
  bash "${SCRIPT_DIR}/run_sift1m_pipeline.sh" --backend "${FUSED_BACKEND}" | tee "${FUSED_LOG}"
  echo "fusedann log saved to ${FUSED_LOG}"
else
  echo "[1/5] Skipping fusedann pipeline (per --fused-backend)."
fi

echo "[2/5] Emitting big-ann formatted SIFT1M files"
python3 "${SCRIPT_DIR}/prepare_sift1m_bigann.py" --sift-root "${ROOT_DIR}/data/sift1m" --bigann-root "${ROOT_DIR}/third_party/big-ann-benchmarks"

echo "Applying SIFT1M dataset patch inside big-ann"
if (cd "${BIGANN_DIR}" && patch -p1 --forward --silent < "${BIGANN_PATCH}"); then
  echo "✅ big-ann patch applied"
else
  if grep -q "sift1m-filter" "${BIGANN_DIR}/benchmark/datasets.py"; then
    echo "ℹ️  big-ann patch already present"
  else
    echo "❌ Failed to apply ${BIGANN_PATCH}." >&2
    exit 1
  fi
fi

pushd "${BIGANN_DIR}" >/dev/null

echo "[3/5] Building Docker images for: ${ALGO_LIST[*]}"
for algo in "${ALGO_LIST[@]}"; do
  echo "  - docker build for ${algo}"
  python3 install.py --neurips23track "${TRACK}" --algorithm "${algo}"
done

COUNT_ARGS=()
if [[ -n "${COUNT}" ]]; then
  COUNT_ARGS=("--count" "${COUNT}")
fi

echo "[4/5] Running Docker benchmarks"
for algo in "${ALGO_LIST[@]}"; do
  echo "  - running ${algo}"
  python3 run.py \
    --neurips23track "${TRACK}" \
    --dataset "${DATASET}" \
    --algorithm "${algo}" \
    --runs "${RUNS}" \
    ${FORCE_FLAG} \
    "${COUNT_ARGS[@]}"
done

RESULT_CSV="results/${DATASET}_docker_summary.csv"

echo "[5/5] Exporting aggregate metrics -> ${RESULT_CSV}"
python3 data_export.py \
  --out "${RESULT_CSV}" \
  --datasets "${DATASET}" \
  --tracks "${TRACK}"

if ${RUN_PLOT}; then
  echo "Generating QPS/recall plot"
  python3 plot.py --dataset "${DATASET}" --neurips23track "${TRACK}"
else
  echo "Skipping plot generation per --skip-plot."
fi

popd >/dev/null

echo "All done."
if ${RUN_FUSED}; then
  echo "- fusedann log: ${FUSED_LOG:-n/a}"
else
  echo "- fusedann log: not requested"
fi
echo "- big-ann CSV: ${BIGANN_DIR}/${RESULT_CSV}"
echo "- big-ann CSV: ${BIGANN_DIR}/${RESULT_CSV}"