#!/usr/bin/env bash
# Compare fusedann-parlay against the ParlayIVF Docker baseline on SIFT1M.
set -euo pipefail

function usage() {
  cat <<'EOF'
Usage: compare_sift1m_parlay_only.sh [options]

Options:
  --runs N        Number of ParlayIVF docker repetitions to keep the best run (default: 1)
  --count K       Override recall@K for big-ann run.py (default: dataset default)
  --skip-fused    Skip the fusedann pipeline step if you already ran it (default: run)
  --force-docker  Re-run ParlayIVF docker even if cached results exist (default: reuse)
  --help          Show this message.
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

RUNS=1
COUNT=""
RUN_FUSED=true
FORCE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --skip-fused)
      RUN_FUSED=false
      shift
      ;;
    --force-docker)
      FORCE_ARGS=("--force")
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

function require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command '$1'." >&2
    exit 1
  fi
}

require_command python3
require_command docker
require_command bash

if [[ ! -d "${BIGANN_DIR}" ]]; then
  echo "Missing big-ann-benchmarks checkout at ${BIGANN_DIR}." >&2
  exit 1
fi

if ${RUN_FUSED}; then
  echo "[1/6] Running fusedann_parlay pipeline"
  FUSED_LOG="${LOG_DIR}/fusedann_parlay_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
  bash "${SCRIPT_DIR}/run_sift1m_pipeline.sh" --backend parlay | tee "${FUSED_LOG}"
  echo "   fusedann log saved to ${FUSED_LOG}"
else
  echo "[1/6] Skipping fusedann pipeline per --skip-fused"
fi

SIFT_ROOT="${ROOT_DIR}/data/sift1m"
BIGANN_DATA="${BIGANN_DIR}/data/${DATASET}"

echo "[2/6] Regenerating big-ann formatted SIFT1M files"
python3 "${SCRIPT_DIR}/prepare_sift1m_bigann.py" --sift-root "${SIFT_ROOT}" --bigann-root "${BIGANN_DIR}"

echo "[3/6] Verifying vector/attribute/GT parity"
python3 "${SCRIPT_DIR}/verify_sift1m_shapes.py" --sift-root "${SIFT_ROOT}" --bigann-root "${BIGANN_DATA}" --autofix

echo "Applying SIFT1M dataset patch inside big-ann"
if (cd "${BIGANN_DIR}" && patch -p1 --forward --silent < "${BIGANN_PATCH}"); then
  echo "✅ big-ann patch applied"
else
  if grep -q "${DATASET}" "${BIGANN_DIR}/benchmark/datasets.py"; then
    echo "ℹ️  big-ann patch already present"
  else
    echo "❌ Failed to apply ${BIGANN_PATCH}." >&2
    exit 1
  fi
fi

pushd "${BIGANN_DIR}" >/dev/null

echo "[4/6] Building ParlayIVF docker image"
python3 install.py --neurips23track "${TRACK}" --algorithm parlayivf

COUNT_ARGS=()
if [[ -n "${COUNT}" ]]; then
  COUNT_ARGS=("--count" "${COUNT}")
fi

echo "[5/6] Running ParlayIVF benchmark"
python3 run.py \
  --neurips23track "${TRACK}" \
  --dataset "${DATASET}" \
  --algorithm parlayivf \
  --runs "${RUNS}" \
  "${FORCE_ARGS[@]}" \
  "${COUNT_ARGS[@]}"

RESULT_CSV="results/${DATASET}_docker_summary.csv"

echo "[6/6] Exporting results"
python3 data_export.py --out "${RESULT_CSV}" --datasets "${DATASET}" --tracks "${TRACK}"

popd >/dev/null

echo "All done."
if ${RUN_FUSED}; then
  echo "- fusedann log: ${FUSED_LOG}"
else
  echo "- fusedann log: skipped"
fi
echo "- ParlayIVF CSV: ${BIGANN_DIR}/${RESULT_CSV}"
