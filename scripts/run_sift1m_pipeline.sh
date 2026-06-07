#!/usr/bin/env bash
# Orchestrate the full SIFT1M experiment: dataset download, preprocessing,
# EFANNA build, project build, and experiment execution.
set -euo pipefail

function usage() {
  cat <<'EOF'
Usage: run_sift1m_pipeline.sh [--fresh] [--backend <efanna|parlay>] [--parlay-root DIR] [--parlaylib-root DIR]
                              [--emit-diagnostics <none|jsonl|tsv>] [--diagnostics-out PATH] [-- <extra args>...]

--fresh            Remove all generated artefacts before running (data/, build/, third_party/efanna).
                   This forces a clean download and rebuild.
--backend          Select the ANN backend to build and execute. Defaults to "efanna".
--parlay-root      Path to a ParlayANN checkout when using the parlay backend (default: third_party/parlayann).
--parlaylib-root   Directory containing the parlay headers (default: <parlay-root>/parlaylib).
--emit-diagnostics Enable per-query diagnostics output for the backend binary.
--diagnostics-out  Destination path for diagnostics when enabled.
--                Pass all remaining arguments through to the backend binary.
EOF
}

FRESH_RUN=false
BACKEND="efanna"
PARLAYANN_ROOT=""
PARLAYLIB_ROOT=""
PARLAYANN_ROOT_SET=false
PARLAYLIB_ROOT_SET=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FRESH_RUN=true
      shift
      ;;
    --backend)
      BACKEND=${2:-}
      if [[ -z "$BACKEND" ]]; then
        echo "Missing value for --backend" >&2
        usage
        exit 1
      fi
      shift 2
      ;;
    --parlay-root)
      PARLAYANN_ROOT=${2:-}
      if [[ -z "$PARLAYANN_ROOT" ]]; then
        echo "Missing value for --parlay-root" >&2
        usage
        exit 1
      fi
      PARLAYANN_ROOT_SET=true
      shift 2
      ;;
    --parlaylib-root)
      PARLAYLIB_ROOT=${2:-}
      if [[ -z "$PARLAYLIB_ROOT" ]]; then
        echo "Missing value for --parlaylib-root" >&2
        usage
        exit 1
      fi
      PARLAYLIB_ROOT_SET=true
      shift 2
      ;;
    --emit-diagnostics)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --emit-diagnostics" >&2
        usage
        exit 1
      fi
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --diagnostics-out)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --diagnostics-out" >&2
        usage
        exit 1
      fi
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      break
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ "$BACKEND" != "efanna" && "$BACKEND" != "parlay" ]]; then
  echo "Unknown backend '$BACKEND'. Choose 'efanna' or 'parlay'." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

DATASET_NAME="sift1m"
DATASET_LOG_FILE="${ROOT_DIR}/${DATASET_NAME}_results.log"

function log_experiment_result() {
  local log_file="$1"
  shift
  local -a cmd=("$@")
  local capture_file
  capture_file=$(mktemp)
  # Run the command, mirror output to the console, and capture it for parsing.
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
    cmd_string=${cmd_string%% } # trim trailing space
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

DATA_ROOT="${ROOT_DIR}/data/sift1m"
RAW_DIR="${DATA_ROOT}/raw"
PROC_DIR="${DATA_ROOT}"
SIFT_ARCHIVE="${RAW_DIR}/sift.tar.gz"
SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

EFANNA_DIR="${ROOT_DIR}/third_party/efanna"
EFANNA_REPO="https://github.com/ZJULearning/efanna_graph"

PROJECT_BUILD_DIR="${ROOT_DIR}/build"
PYTHON_SCRIPT="${SCRIPT_DIR}/process_sift1m.py"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements.txt"

if [[ -z "$PARLAYANN_ROOT" ]]; then
  PARLAYANN_ROOT="${ROOT_DIR}/third_party/parlayann"
fi
if [[ -z "$PARLAYLIB_ROOT" ]]; then
  PARLAYLIB_ROOT="${PARLAYANN_ROOT}/parlaylib/include"
fi

  PARLAYANN_REPO="git@github.com:cmuparlay/ParlayANN.git"
  PARLAYANN_REPO_FALLBACK="https://github.com/cmuparlay/ParlayANN.git"
  PARLAYLIB_REPO="https://github.com/cmuparlay/parlaylib.git"

TOTAL_STEPS=10
if [[ "$BACKEND" == "parlay" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
CURRENT_STEP=0

function progress() {
  local label="$1"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  local width=20
  local filled=$((CURRENT_STEP * width / TOTAL_STEPS))
  local empty=$((width - filled))
  local bar=""
  local pad=""
  if (( filled > 0 )); then
    bar=$(printf '%0.s#' $(seq 1 $filled))
  fi
  if (( empty > 0 )); then
    pad=$(printf '%0.s-' $(seq 1 $empty))
  fi
  printf '\n[%02d/%02d] [%s%s] %s\n' "$CURRENT_STEP" "$TOTAL_STEPS" "$bar" "$pad" "$label"
}

if ${FRESH_RUN}; then
  echo "Fresh run requested: removing generated artefacts."
  rm -rf "${DATA_ROOT}" "${PROJECT_BUILD_DIR}" "${EFANNA_DIR}" "${ROOT_DIR}/third_party/parlayann"
fi

function require_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "Missing required command '$1'. Please install it and re-run." >&2
    exit 1
  fi
}

require_command wget
require_command tar
require_command git
require_command cmake
require_command python3

progress "Checking prerequisites"
echo "✔️  Required commands are available."

mkdir -p "${RAW_DIR}"

progress "Downloading SIFT1M dataset"
if [ ! -f "${SIFT_ARCHIVE}" ]; then
  echo "Downloading SIFT1M dataset..."
  wget -O "${SIFT_ARCHIVE}" "${SIFT_URL}"
else
  echo "Dataset archive already present at ${SIFT_ARCHIVE}. Skipping download."
fi

progress "Extracting SIFT1M vectors"
if [ ! -f "${RAW_DIR}/sift_base.fvecs" ]; then
  echo "Extracting SIFT1M vectors..."
  tar -xzf "${SIFT_ARCHIVE}" -C "${RAW_DIR}" --strip-components=1
else
  echo "SIFT1M vectors already extracted in ${RAW_DIR}."
fi

progress "Validating dataset files"
for required in "sift_base.fvecs" "sift_query.fvecs" "sift_groundtruth.ivecs"; do
  if [ ! -f "${RAW_DIR}/${required}" ]; then
    echo "Expected ${required} in ${RAW_DIR}." >&2
    exit 1
  fi
done
echo "✔️  Found required raw dataset artefacts."

progress "Checking Python environment"
if ! python3 -c "import numpy" &>/dev/null; then
  echo "Python package 'numpy' is required."
  if [ -f "${REQUIREMENTS_FILE}" ]; then
    echo "Install dependencies via: python3 -m pip install -r ${REQUIREMENTS_FILE}" >&2
  fi
  exit 1
fi
echo "✔️  Python environment ready."

progress "Generating attribute bitmaps"
echo "Generating attribute bitmaps via ${PYTHON_SCRIPT}..."
python3 "${PYTHON_SCRIPT}" --raw-dir "${RAW_DIR}" --out-dir "${PROC_DIR}" --force

if [ ! -f "${PROC_DIR}/sift_base_attrs.bvecs" ] || [ ! -f "${PROC_DIR}/sift_query_attrs.bvecs" ]; then
  echo "Attribute generation failed; required *.bvecs files missing." >&2
  exit 1
fi
echo "✔️  Attribute bitmaps ready."

progress "Syncing EFANNA sources"
if [ ! -d "${EFANNA_DIR}" ]; then
  echo "Cloning EFANNA into ${EFANNA_DIR}..."
  git clone --depth 1 "${EFANNA_REPO}" "${EFANNA_DIR}"
else
  echo "Updating EFANNA in ${EFANNA_DIR}..."
  git -C "${EFANNA_DIR}" pull --ff-only || true
fi

if [[ "$BACKEND" == "parlay" ]]; then
  progress "Syncing ParlayANN sources"
  if [ ! -d "${PARLAYANN_ROOT}" ]; then
    if [[ "${PARLAYANN_ROOT_SET}" == true ]]; then
      echo "ParlayANN root ${PARLAYANN_ROOT} not found. Set --parlay-root to a valid checkout." >&2
      exit 1
    fi
    echo "Cloning ParlayANN into ${PARLAYANN_ROOT}..."
    if ! git clone --depth 1 "${PARLAYANN_REPO}" "${PARLAYANN_ROOT}"; then
      echo "SSH clone failed; retrying with HTTPS." >&2
      rm -rf "${PARLAYANN_ROOT}"
      git clone --depth 1 "${PARLAYANN_REPO_FALLBACK}" "${PARLAYANN_ROOT}"
    fi
  else
    if [ ! -d "${PARLAYANN_ROOT}/.git" ]; then
      if [[ "${PARLAYANN_ROOT_SET}" == true ]]; then
        echo "ParlayANN root ${PARLAYANN_ROOT} exists without git metadata; please clean it manually." >&2
        exit 1
      fi
      echo "⚠️  Removing stale ParlayANN directory lacking git metadata (${PARLAYANN_ROOT})."
      rm -rf "${PARLAYANN_ROOT}"
      echo "Cloning ParlayANN into ${PARLAYANN_ROOT}..."
      if ! git clone --depth 1 "${PARLAYANN_REPO}" "${PARLAYANN_ROOT}"; then
        echo "SSH clone failed; retrying with HTTPS." >&2
        rm -rf "${PARLAYANN_ROOT}"
        git clone --depth 1 "${PARLAYANN_REPO_FALLBACK}" "${PARLAYANN_ROOT}"
      fi
    else
      echo "Updating ParlayANN in ${PARLAYANN_ROOT}..."

      # If the existing checkout uses an SSH remote and the environment lacks SSH credentials,
      # git operations may block waiting for auth. Prefer HTTPS for non-interactive pipelines.
      origin_url=$(git -C "${PARLAYANN_ROOT}" remote get-url origin 2>/dev/null || true)
      if [[ "${origin_url}" == git@github.com:* ]]; then
        echo "⚠️  ParlayANN origin uses SSH (${origin_url}); switching to HTTPS (${PARLAYANN_REPO_FALLBACK})."
        git -C "${PARLAYANN_ROOT}" remote set-url origin "${PARLAYANN_REPO_FALLBACK}" || true
      fi

      if ! git -C "${PARLAYANN_ROOT}" fetch --tags >/dev/null 2>&1; then
        if [[ "${PARLAYANN_ROOT_SET}" == true ]]; then
          echo "Failed to fetch ParlayANN updates in ${PARLAYANN_ROOT}." >&2
          exit 1
        fi
        echo "⚠️  Fetch failed; recloning ParlayANN."
        rm -rf "${PARLAYANN_ROOT}"
        if ! git clone --depth 1 "${PARLAYANN_REPO}" "${PARLAYANN_ROOT}"; then
          echo "SSH clone failed; retrying with HTTPS." >&2
          rm -rf "${PARLAYANN_ROOT}"
          git clone --depth 1 "${PARLAYANN_REPO_FALLBACK}" "${PARLAYANN_ROOT}"
        fi
      else
        if ! git -C "${PARLAYANN_ROOT}" pull --ff-only; then
          if [[ "${PARLAYANN_ROOT_SET}" == true ]]; then
            echo "ParlayANN checkout in ${PARLAYANN_ROOT} has divergent history; clean or choose another directory." >&2
            exit 1
          fi
          echo "⚠️  ParlayANN pull failed; recloning ${PARLAYANN_ROOT}."
          rm -rf "${PARLAYANN_ROOT}"
          if ! git clone --depth 1 "${PARLAYANN_REPO}" "${PARLAYANN_ROOT}"; then
            echo "SSH clone failed; retrying with HTTPS." >&2
            rm -rf "${PARLAYANN_ROOT}"
            git clone --depth 1 "${PARLAYANN_REPO_FALLBACK}" "${PARLAYANN_ROOT}"
          fi
        fi
      fi
    fi
  fi

  if [ -d "${PARLAYANN_ROOT}/.git" ]; then
    git -C "${PARLAYANN_ROOT}" submodule update --init --recursive || true
  fi

  if [ ! -d "${PARLAYANN_ROOT}" ]; then
    echo "ParlayANN root ${PARLAYANN_ROOT} not found after sync." >&2
    exit 1
  fi
  if [ ! -d "${PARLAYLIB_ROOT}" ]; then
    if [[ "${PARLAYLIB_ROOT_SET}" == true ]]; then
      echo "Parlaylib headers ${PARLAYLIB_ROOT} not found." >&2
      exit 1
    fi
    echo "Parlaylib headers ${PARLAYLIB_ROOT} not found after syncing ParlayANN." >&2
    echo "Try deleting ${PARLAYANN_ROOT} and rerunning with --backend parlay." >&2
    exit 1
  fi
  echo "✔️  ParlayANN sources ready at ${PARLAYANN_ROOT}."
fi

BASE_FVECS="${PROC_DIR}/sift_base.fvecs"
QUERY_FVECS="${PROC_DIR}/sift_query.fvecs"

if [ ! -f "${BASE_FVECS}" ]; then
  BASE_FVECS="${RAW_DIR}/sift_base.fvecs"
fi
if [ ! -f "${QUERY_FVECS}" ]; then
  QUERY_FVECS="${RAW_DIR}/sift_query.fvecs"
fi

BASE_ATTRS="${PROC_DIR}/sift_base_attrs.bvecs"
QUERY_ATTRS="${PROC_DIR}/sift_query_attrs.bvecs"
GT_FILE="${RAW_DIR}/sift_groundtruth.ivecs"

for file in "${BASE_FVECS}" "${QUERY_FVECS}" "${BASE_ATTRS}" "${QUERY_ATTRS}" "${GT_FILE}"; do
  if [ ! -f "${file}" ]; then
    echo "Required file ${file} is missing." >&2
    exit 1
  fi
done

progress "Configuring CMake project"
echo "Configuring fusedann-cpp (backend: ${BACKEND})..."
if [[ "$BACKEND" == "parlay" ]]; then
  cmake -S "${ROOT_DIR}" -B "${PROJECT_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEFANNA_ROOT="${EFANNA_DIR}" \
    -DENABLE_PARLAYANN=ON \
    -DPARLAYANN_ROOT="${PARLAYANN_ROOT}" \
    -DPARLAYLIB_INCLUDE_DIR="${PARLAYLIB_ROOT}"
else
  cmake -S "${ROOT_DIR}" -B "${PROJECT_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DEFANNA_ROOT="${EFANNA_DIR}"
fi

if [[ "$BACKEND" == "parlay" ]]; then
  progress "Building fusedann_parlayann"
  echo "Building fusedann_parlayann executable..."
  cmake --build "${PROJECT_BUILD_DIR}" --target fusedann_parlayann --config Release
  EXECUTABLE="${PROJECT_BUILD_DIR}/fusedann_parlayann"
  if [ ! -x "${EXECUTABLE}" ]; then
    EXECUTABLE="${PROJECT_BUILD_DIR}/Release/fusedann_parlayann"
  fi
  if [ ! -x "${EXECUTABLE}" ]; then
    echo "Could not find the fusedann_parlayann executable in ${PROJECT_BUILD_DIR}." >&2
    exit 1
  fi
  CACHE_DIR="${PROC_DIR}/fused_cache"
  progress "Running ANN experiment"
  echo "Running ANN experiment..."
  mkdir -p "${ROOT_DIR}/logs"
  log_experiment_result "${DATASET_LOG_FILE}" \
    "${EXECUTABLE}" \
    "${BASE_FVECS}" \
    "${BASE_ATTRS}" \
    "${QUERY_FVECS}" \
    "${QUERY_ATTRS}" \
    "${GT_FILE}" \
    "${CACHE_DIR}" \
    "${EXTRA_ARGS[@]}"
else
  progress "Building fusedann_efanna"
  echo "Building fusedann_efanna executable..."
  cmake --build "${PROJECT_BUILD_DIR}" --target fusedann_efanna --config Release
  EXECUTABLE="${PROJECT_BUILD_DIR}/fusedann_efanna"
  if [ ! -x "${EXECUTABLE}" ]; then
    EXECUTABLE="${PROJECT_BUILD_DIR}/Release/fusedann_efanna"
  fi
  if [ ! -x "${EXECUTABLE}" ]; then
    echo "Could not find the fusedann_efanna executable in ${PROJECT_BUILD_DIR}." >&2
    exit 1
  fi
  progress "Running EFANNA experiment"
  echo "Running EFANNA experiment..."
  log_experiment_result "${DATASET_LOG_FILE}" \
    "${EXECUTABLE}" \
    "${BASE_FVECS}" \
    "${BASE_ATTRS}" \
    "${QUERY_FVECS}" \
    "${QUERY_ATTRS}" \
    "${GT_FILE}"
fi

echo "Pipeline completed successfully."
