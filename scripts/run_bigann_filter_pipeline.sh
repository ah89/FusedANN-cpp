#!/usr/bin/env bash
# Run the YFCC (filter track) pipeline on top of fusedann-parlayann.
set -euo pipefail

function usage() {
  cat <<'EOF'
Usage: run_bigann_filter_pipeline.sh [--fresh] [--backend parlay] [--parlay-root DIR] [--parlaylib-root DIR]

--fresh            Remove cached artefacts before running (data/yfcc10m, build/, third_party/parlayann).
--backend          Supported backend. Only "parlay" is available for the filter track.
--parlay-root      Path to a ParlayANN checkout (default: third_party/parlayann).
--parlaylib-root   Directory containing parlaylib headers (default: <parlay-root>/parlaylib/include).
EOF
}

FRESH_RUN=false
BACKEND="parlay"
PARLAYANN_ROOT=""
PARLAYLIB_ROOT=""
PARLAYANN_ROOT_SET=false
PARLAYLIB_ROOT_SET=false

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
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ "$BACKEND" != "parlay" ]]; then
  echo "Filter track pipeline currently supports only the ParlayANN backend." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

DATA_ROOT="${ROOT_DIR}/data/yfcc10m"
CACHE_DIR="${DATA_ROOT}/fused_cache"
BUILD_DIR="${ROOT_DIR}/build"

if [[ -z "$PARLAYANN_ROOT" ]]; then
  PARLAYANN_ROOT="${ROOT_DIR}/third_party/parlayann"
fi
if [[ -z "$PARLAYLIB_ROOT" ]]; then
  PARLAYLIB_ROOT="${PARLAYANN_ROOT}/parlaylib/include"
fi

PARLAYANN_REPO="git@github.com:cmuparlay/ParlayANN.git"
PARLAYANN_REPO_FALLBACK="https://github.com/cmuparlay/ParlayANN.git"
PARLAYLIB_REPO="https://github.com/cmuparlay/parlaylib.git"

BASE_URL="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M"
declare -A FILES
FILES[base.10M.u8bin]="${BASE_URL}/base.10M.u8bin"
FILES[query.public.100K.u8bin]="${BASE_URL}/query.public.100K.u8bin"
FILES[base.metadata.10M.spmat]="${BASE_URL}/base.metadata.10M.spmat"
FILES[query.metadata.public.100K.spmat]="${BASE_URL}/query.metadata.public.100K.spmat"
FILES[GT.public.ibin]="${BASE_URL}/GT.public.ibin"

TOTAL_STEPS=6
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

function require_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "Missing required command '$1'. Please install it and re-run." >&2
    exit 1
  fi
}

if ${FRESH_RUN}; then
  echo "Fresh run requested: purging generated artefacts."
  rm -rf "${DATA_ROOT}" "${CACHE_DIR}" "${BUILD_DIR}" "${ROOT_DIR}/third_party/parlayann"
fi

progress "Checking prerequisites"
require_command wget
require_command git
require_command cmake
require_command python3
require_command tar
echo "✔️  Required commands are available."

mkdir -p "${DATA_ROOT}" "${CACHE_DIR}"

progress "Downloading YFCC filter dataset"
for filename in "${!FILES[@]}"; do
  target="${DATA_ROOT}/${filename}"
  url="${FILES[$filename]}"
  if [[ -f "$target" ]]; then
    echo "• ${filename} already present."
    continue
  fi
  echo "• Fetching ${filename} (this may take a while)..."
  wget -O "$target" "$url"
done

for required in \
  "base.10M.u8bin" \
  "query.public.100K.u8bin" \
  "base.metadata.10M.spmat" \
  "query.metadata.public.100K.spmat" \
  "GT.public.ibin"; do
  if [[ ! -f "${DATA_ROOT}/${required}" ]]; then
    echo "Required dataset artefact ${required} is missing." >&2
    exit 1
  fi
done

progress "Syncing ParlayANN sources"
if [[ ! -d "${PARLAYANN_ROOT}" ]]; then
  if [[ "${PARLAYANN_ROOT_SET}" == true ]]; then
    echo "ParlayANN root ${PARLAYANN_ROOT} not found." >&2
    exit 1
  fi
  echo "Cloning ParlayANN into ${PARLAYANN_ROOT}..."
  if ! git clone --depth 1 "${PARLAYANN_REPO}" "${PARLAYANN_ROOT}"; then
    echo "SSH clone failed; retrying with HTTPS." >&2
    rm -rf "${PARLAYANN_ROOT}"
    git clone --depth 1 "${PARLAYANN_REPO_FALLBACK}" "${PARLAYANN_ROOT}"
  fi
else
  echo "Updating ParlayANN in ${PARLAYANN_ROOT}..."
  git -C "${PARLAYANN_ROOT}" pull --ff-only || true
fi

git -C "${PARLAYANN_ROOT}" submodule update --init --recursive || true

PARLAYLIB_PARENT=$(dirname "${PARLAYLIB_ROOT}")
if [[ ! -d "${PARLAYLIB_ROOT}" ]]; then
  if [[ "${PARLAYLIB_ROOT_SET}" == true ]]; then
    echo "Parlaylib headers ${PARLAYLIB_ROOT} not found." >&2
    exit 1
  fi
  if [[ -d "${PARLAYLIB_PARENT}" && ! -d "${PARLAYLIB_PARENT}/.git" ]]; then
    echo "Parlaylib directory ${PARLAYLIB_PARENT} exists without git metadata; cannot auto-clone." >&2
    exit 1
  fi
  if [[ -d "${PARLAYLIB_PARENT}/.git" ]]; then
    git -C "${PARLAYLIB_PARENT}" pull --ff-only || true
  else
    echo "Cloning parlaylib into ${PARLAYLIB_PARENT}..."
    mkdir -p "${PARLAYLIB_PARENT}"
    git clone --depth 1 "${PARLAYLIB_REPO}" "${PARLAYLIB_PARENT}"
  fi
else
  if [[ -d "${PARLAYLIB_PARENT}/.git" ]]; then
    git -C "${PARLAYLIB_PARENT}" pull --ff-only || true
  fi
fi

progress "Configuring CMake"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PARLAYANN=ON \
  -DPARLAYANN_ROOT="${PARLAYANN_ROOT}" \
  -DPARLAYLIB_INCLUDE_DIR="${PARLAYLIB_ROOT}"

progress "Building fusedann_parlayann"
cmake --build "${BUILD_DIR}" --target fusedann_parlayann --config Release

EXECUTABLE="${BUILD_DIR}/fusedann_parlayann"
if [[ ! -x "${EXECUTABLE}" ]]; then
  EXECUTABLE="${BUILD_DIR}/Release/fusedann_parlayann"
fi
if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Could not find fusedann_parlayann executable after build." >&2
  exit 1
fi

BASE_VECS="${DATA_ROOT}/base.10M.u8bin"
BASE_ATTRS="${DATA_ROOT}/base.metadata.10M.spmat"
QUERY_VECS="${DATA_ROOT}/query.public.100K.u8bin"
QUERY_ATTRS="${DATA_ROOT}/query.metadata.public.100K.spmat"
GT_FILE="${DATA_ROOT}/GT.public.ibin"

if [[ -z "${FUSEDANN_PARLAY_RERANK_K:-}" ]]; then
  export FUSEDANN_PARLAY_RERANK_K=256
fi

if [[ -z "${FUSEDANN_PARLAY_BEAM_WIDTH:-}" ]]; then
  export FUSEDANN_PARLAY_BEAM_WIDTH=512
fi

if [[ -z "${FUSEDANN_PARLAY_VISIT_LIMIT:-}" ]]; then
  export FUSEDANN_PARLAY_VISIT_LIMIT=1024
fi

progress "Running ANN filter experiment"
"${EXECUTABLE}" \
  "${BASE_VECS}" \
  "${BASE_ATTRS}" \
  "${QUERY_VECS}" \
  "${QUERY_ATTRS}" \
  "${GT_FILE}" \
  "${CACHE_DIR}"

echo "Pipeline completed successfully."
