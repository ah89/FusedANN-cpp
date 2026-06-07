#!/usr/bin/env bash
set -euo pipefail

function usage() {
  cat <<'EOF'
Usage: run_pipeline.sh --dataset <name> [--sample-size N] [--gt PATH] [--fresh] [--backend <efanna|parlay>] [--parlay-root DIR] [--parlaylib-root DIR] [-- <backend-flags...>]

--dataset          Name of the dataset (e.g., sift-1M, yfcc-10M, random-filter-s).
--sample-size      Optional subset size for datasets that support it (e.g., 1000000 for YFCC).
--gt               Override groundtruth path passed to the C++ binary (useful for cached fused GT).
--fresh            Remove generated artefacts before running.
--backend          Select the ANN backend (efanna|parlay). Default: efanna.
--parlay-root      Path to ParlayANN checkout.
--parlaylib-root   Path to parlaylib headers.
--                Remaining args are passed to the backend executable (e.g., --alpha-beta-cache-only).
EOF
}

DATASET=""
SAMPLE_SIZE=""
GT_OVERRIDE=""
FRESH_RUN=false
BACKEND="efanna"
PARLAYANN_ROOT=""
PARLAYLIB_ROOT=""
EXTRA_BACKEND_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --gt)
      GT_OVERRIDE="$2"
      shift 2
      ;;
    --fresh)
      FRESH_RUN=true
      shift
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --parlay-root)
      PARLAYANN_ROOT="$2"
      shift 2
      ;;
    --parlaylib-root)
      PARLAYLIB_ROOT="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_BACKEND_ARGS=("$@")
      break
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Missing --dataset" >&2
  usage
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${ROOT_DIR}/build"
EFANNA_DIR="${ROOT_DIR}/third_party/efanna"
EFANNA_REPO="https://github.com/ZJULearning/efanna_graph"

mkdir -p "${ROOT_DIR}/logs"

DATASET_SAFE_NAME=${DATASET// /_}
DATASET_SAFE_NAME=${DATASET_SAFE_NAME//[^A-Za-z0-9._-]/_}
DATASET_LOG_FILE="${ROOT_DIR}/${DATASET_SAFE_NAME}_results.log"

function log_experiment_result() {
  local log_file="$1"
  shift
  local -a cmd=("$@")
  local capture_file
  capture_file=$(mktemp)
  if ! "${cmd[@]}" 2>&1 | tee "${capture_file}"; then
    local status=$?
    rm -f "${capture_file}"
    return ${status}
  fi

  local recall_line
  recall_line=$(grep -E 'FusedANN recall:' "${capture_file}" | tail -n 1 || true)
  local qps_line
  qps_line=$(grep -E 'QPS:' "${capture_file}" | tail -n 1 || true)

  local cmd_string=""
  if [[ ${#cmd[@]} -gt 0 ]]; then
    for arg in "${cmd[@]}"; do
      cmd_string+=$(printf "%q " "$arg")
    done
    cmd_string=${cmd_string%% }
  fi

  touch "${log_file}"
  {
    echo
    echo "------------------------"
    echo "cd ${ROOT_DIR} && ${cmd_string}"
    echo
    if [[ -n "${recall_line}" ]]; then
      echo "${recall_line}"
    else
      echo "FusedANN recall: (not detected)"
    fi
    if [[ -n "${qps_line}" ]]; then
      echo "${qps_line}"
    else
      echo "⚡ QPS: (not detected)"
    fi
  } >> "${log_file}"

  echo "📝 Appended results to ${log_file}"
  rm -f "${capture_file}"
}

if [[ -z "$PARLAYANN_ROOT" ]]; then
  PARLAYANN_ROOT="${ROOT_DIR}/third_party/parlayann"
fi
if [[ -z "$PARLAYLIB_ROOT" ]]; then
  PARLAYLIB_ROOT="${PARLAYANN_ROOT}/parlaylib/include"
fi

PARLAYANN_REPO="git@github.com:cmuparlay/ParlayANN.git"
PARLAYANN_REPO_FALLBACK="https://github.com/cmuparlay/ParlayANN.git"

if ${FRESH_RUN}; then
  echo "Fresh run requested: removing generated artefacts."
  rm -rf "${BUILD_DIR}" "${EFANNA_DIR}" "${ROOT_DIR}/third_party/parlayann"
  # Note: We don't remove data here as it might be shared/large.
fi

# Sync EFANNA
if [ ! -d "${EFANNA_DIR}" ]; then
  echo "Cloning EFANNA into ${EFANNA_DIR}..."
  git clone --depth 1 "${EFANNA_REPO}" "${EFANNA_DIR}"
else
  echo "Updating EFANNA in ${EFANNA_DIR}..."
  git -C "${EFANNA_DIR}" pull --ff-only || true
fi

# Sync ParlayANN if needed
if [[ "$BACKEND" == "parlay" ]]; then
  if [ ! -d "${PARLAYANN_ROOT}" ]; then
    echo "Cloning ParlayANN into ${PARLAYANN_ROOT}..."
    if ! git clone --depth 1 "${PARLAYANN_REPO}" "${PARLAYANN_ROOT}"; then
      rm -rf "${PARLAYANN_ROOT}"
      git clone --depth 1 "${PARLAYANN_REPO_FALLBACK}" "${PARLAYANN_ROOT}"
    fi
  fi
  if [ -d "${PARLAYANN_ROOT}/.git" ]; then
    git -C "${PARLAYANN_ROOT}" submodule update --init --recursive || true
  fi
fi

# Prepare dataset
echo "Preparing dataset $DATASET..."
CMD="python3 ${SCRIPT_DIR}/prepare_dataset.py --dataset $DATASET"
if [[ -n "$SAMPLE_SIZE" ]]; then
  CMD="$CMD --sample-size $SAMPLE_SIZE"
fi
$CMD

# Get paths
INFO_CMD="python3 ${SCRIPT_DIR}/get_dataset_info.py --dataset $DATASET"
if [[ -n "$SAMPLE_SIZE" ]]; then
  INFO_CMD="$INFO_CMD --sample-size $SAMPLE_SIZE"
fi

BASE_FVECS=$($INFO_CMD --info base)
QUERY_FVECS=$($INFO_CMD --info query)
GT_FILE=$($INFO_CMD --info groundtruth)
BASE_DIR=$($INFO_CMD --info basedir)

if [[ -n "$GT_OVERRIDE" ]]; then
  GT_FILE="$GT_OVERRIDE"
fi

# Attribute generation
if [[ "$DATASET" == "sift-1M" ]]; then
    echo "Generating attributes for SIFT1M..."
    python3 "${SCRIPT_DIR}/process_sift1m.py" --raw-dir "$BASE_DIR" --out-dir "$BASE_DIR" --force
    BASE_ATTRS="${BASE_DIR}/sift_base_attrs.bvecs"
    QUERY_ATTRS="${BASE_DIR}/sift_query_attrs.bvecs"
else
    BASE_ATTRS=$($INFO_CMD --info base_attrs)
    QUERY_ATTRS=$($INFO_CMD --info query_attrs)
fi

if [[ -z "$BASE_ATTRS" || -z "$QUERY_ATTRS" ]]; then
    echo "Error: Could not determine attribute files for $DATASET"
    exit 1
fi

# Build
echo "Configuring CMake..."
if [[ "$BACKEND" == "parlay" ]]; then
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEFANNA_ROOT="${EFANNA_DIR}" \
    -DENABLE_PARLAYANN=ON \
    -DPARLAYANN_ROOT="${PARLAYANN_ROOT}" \
    -DPARLAYLIB_INCLUDE_DIR="${PARLAYLIB_ROOT}"
else
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DEFANNA_ROOT="${EFANNA_DIR}"
fi

echo "Building..."
if [[ "$BACKEND" == "parlay" ]]; then
  cmake --build "${BUILD_DIR}" --target fusedann_parlayann --config Release
  EXECUTABLE="${BUILD_DIR}/fusedann_parlayann"
  if [ ! -x "${EXECUTABLE}" ]; then EXECUTABLE="${BUILD_DIR}/Release/fusedann_parlayann"; fi
  CACHE_DIR="${BASE_DIR}/fused_cache"
  echo "Running ANN..."
  log_experiment_result "${DATASET_LOG_FILE}" \
    "${EXECUTABLE}" \
    "$BASE_FVECS" \
    "$BASE_ATTRS" \
    "$QUERY_FVECS" \
    "$QUERY_ATTRS" \
    "$GT_FILE" \
    "$CACHE_DIR" \
    "${EXTRA_BACKEND_ARGS[@]}"
else
  cmake --build "${BUILD_DIR}" --target fusedann_efanna --config Release
  EXECUTABLE="${BUILD_DIR}/fusedann_efanna"
  if [ ! -x "${EXECUTABLE}" ]; then EXECUTABLE="${BUILD_DIR}/Release/fusedann_efanna"; fi
  echo "Running EFANNA..."
  log_experiment_result "${DATASET_LOG_FILE}" \
    "${EXECUTABLE}" \
    "$BASE_FVECS" \
    "$BASE_ATTRS" \
    "$QUERY_FVECS" \
    "$QUERY_ATTRS" \
    "$GT_FILE"
fi
