#!/usr/bin/env bash
# Run the fusedann SIFT1M pipeline and benchmark NeurIPS'23 filter opponents on the same data.
set -euo pipefail

function usage() {
  cat <<'EOF'
Usage: run_sift1m_filter_benchmarks.sh [options]

General options:
  --fresh                 Remove cached artefacts before running (fused cache, big-ann checkout, results).
  --fused-backend MODE    Which fusedann backend(s) to execute: efanna|parlay|both|none (default: parlay).
  --opponents LIST        Comma-separated list of NeurIPS'23 filter algorithms to run (default: faiss,faissplus,parlayivf,pyanns,puck).
  --runs N                Number of repeated runs per opponent (default: 1).
  -k, --count K           Recall@K to request from big-ann (default: 10).
  --force-opponents       Re-run opponents even if a previous result exists.
  --skip-plot             Skip plotting the recall/QPS curve.
  --bigann-root DIR       Override the checkout directory for big-ann-benchmarks.
  --help                  Show this message.
EOF
}

FRESH=false
FUSED_MODE="parlay"
OPPONENT_ARG=""
RUNS=1
COUNT=10
FORCE_OPP=false
SKIP_PLOT=false
BIGANN_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FRESH=true
      shift
      ;;
    --fused-backend)
      FUSED_MODE=${2:-}
      if [[ -z "$FUSED_MODE" ]]; then
        usage; exit 1
      fi
      shift 2
      ;;
    --opponents)
      OPPONENT_ARG=${2:-}
      if [[ -z "$OPPONENT_ARG" ]]; then
        usage; exit 1
      fi
      shift 2
      ;;
    --runs)
      RUNS=${2:-}
      shift 2
      ;;
    -k|--count)
      COUNT=${2:-}
      shift 2
      ;;
    --force-opponents)
      FORCE_OPP=true
      shift
      ;;
    --skip-plot)
      SKIP_PLOT=true
      shift
      ;;
    --bigann-root)
      BIGANN_ROOT=${2:-}
      if [[ -z "$BIGANN_ROOT" ]]; then
        usage; exit 1
      fi
      shift 2
      ;;
    --help)
      usage; exit 0
      ;;
    *)
      usage; exit 1
      ;;
  esac
done

case "$FUSED_MODE" in
  efanna|parlay|both|none) ;;
  *) echo "Unknown fused backend '$FUSED_MODE'." >&2; exit 1 ;;
esac

if [[ "$FUSED_MODE" != "none" && "$COUNT" != "10" ]]; then
  echo "fusedann currently reports Recall@10 only; rerun with -k 10 or disable fused backends." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT="${ROOT_DIR}/data/sift1m"
DEFAULT_OPPONENTS=(faiss faissplus parlayivf pyanns puck)
IFS=',' read -r -a PROVIDED <<< "$OPPONENT_ARG"
if [[ -n "$OPPONENT_ARG" ]]; then
  OPPONENTS=("${PROVIDED[@]}")
else
  OPPONENTS=("${DEFAULT_OPPONENTS[@]}")
fi
if [[ -z "$BIGANN_ROOT" ]]; then
  BIGANN_ROOT="${ROOT_DIR}/third_party/big-ann-benchmarks"
fi
BIGANN_PATCH="${SCRIPT_DIR}/bigann_sift1m_dataset.patch"
PARLAYANN_REPO="https://github.com/cmuparlay/ParlayANN.git"
PARLAYANN_COMMIT="f7208ba5795a4bd433361871579057c26106c4ba"
PARLAYANN_ARCHIVE="https://codeload.github.com/cmuparlay/ParlayANN/zip/${PARLAYANN_COMMIT}"
PARLAYLIB_COMMIT="51017699dcc421f80479cdb238d3092233ad0d26"
PARLAYLIB_ARCHIVE="https://codeload.github.com/cmuparlay/parlaylib/zip/${PARLAYLIB_COMMIT}"
PARLAYANN_DIR="${ROOT_DIR}/third_party/parlayann-filter"
PARLAYANN_PYTHON="${PARLAYANN_DIR}/python"
SUMMARY_FILE="${BIGANN_ROOT}/results/sift1m-filter-summary.csv"
PARLAYIVF_SOURCE="${BIGANN_ROOT}/neurips23/filter/parlayivf/parlayivf.py"

declare -A FUSED_RECALL=()
declare -A FUSED_QPS=()

NEEDS_PARLAY=false
for opp in "${OPPONENTS[@]}"; do
  if [[ "$opp" == "parlayivf" ]]; then
    NEEDS_PARLAY=true
    break
  fi
done

function require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command '$1'." >&2
    exit 1
  fi
}

require_command git
require_command python3
require_command patch
require_command cmake
require_command unzip
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "Missing required downloader (curl or wget)." >&2
  exit 1
fi
TOTAL_STEPS=6
if ${FRESH}; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "$FUSED_MODE" == "efanna" || "$FUSED_MODE" == "parlay" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
elif [[ "$FUSED_MODE" == "both" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 2))
fi
if ${SKIP_PLOT}; then
  TOTAL_STEPS=$((TOTAL_STEPS - 1))
fi
if ${NEEDS_PARLAY}; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
CURRENT_STEP=0
function progress() {
  CURRENT_STEP=$((CURRENT_STEP + 1))
  printf '\n[%02d/%02d] %s\n' "$CURRENT_STEP" "$TOTAL_STEPS" "$1"
}

function parse_fused_metrics() {
  local backend="$1"
  local log_file="$2"
  local parsed
  if ! parsed=$(python3 - "$log_file" <<'PY'
import pathlib, re, sys

log_path = pathlib.Path(sys.argv[1])
text = log_path.read_text(encoding='utf-8', errors='ignore')

recall = None
qps = None

for line in text.splitlines():
  lower = line.lower()
  if 'recall' in lower:
    match = re.search(r"recall:\s*([0-9]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
    if match:
      recall = float(match.group(1))
  if 'qps' in lower:
    match = re.search(r"QPS:\s*([0-9]+(?:\.[0-9]+)?|inf)", line, re.IGNORECASE)
    if match:
      value = match.group(1)
      qps = float('inf') if value.lower() == 'inf' else float(value)

if recall is None or qps is None:
  sys.exit(1)

if qps == float('inf'):
  qps_str = 'inf'
else:
  qps_str = f"{qps:.6f}"

print(f"{recall:.6f} {qps_str}")
PY
); then
  return 1
  fi

  local recall qps
  read -r recall qps <<< "$parsed"
  FUSED_RECALL["$backend"]="$recall"
  FUSED_QPS["$backend"]="$qps"
  printf '   Captured fusedann-%s metrics: Recall@10=%s, QPS=%s\n' "$backend" "$recall" "$qps"
}

function run_fused_backend() {
  local backend="$1"
  local label="$2"
  local log_file
  log_file=$(mktemp -t "fusedann-${backend}-XXXX.log")
  if "${SCRIPT_DIR}/run_sift1m_pipeline.sh" --backend "$backend" 2>&1 | tee "$log_file"; then
    if parse_fused_metrics "$backend" "$log_file"; then
      rm -f "$log_file"
    else
      echo "⚠️  Unable to parse fusedann (${label}) metrics. Inspect $log_file" >&2
    fi
  else
    local status=$?
    echo "❌ fusedann (${label}) pipeline failed. Logs: $log_file" >&2
    exit $status
  fi
}

function persist_fused_metrics() {
  if [[ ${#FUSED_RECALL[@]} -eq 0 ]]; then
    return
  fi
  for backend in "${!FUSED_RECALL[@]}"; do
    local recall="${FUSED_RECALL[$backend]}"
    local qps="${FUSED_QPS[$backend]}"
    python3 - "$backend" "$SUMMARY_FILE" "$COUNT" "$recall" "$qps" <<'PY'
import csv, math, pathlib, sys

backend = sys.argv[1]
summary_path = pathlib.Path(sys.argv[2])
k = sys.argv[3]
recall = float(sys.argv[4])
raw_qps = sys.argv[5]
qps = math.inf if raw_qps.lower() == 'inf' else float(raw_qps)

dataset = 'sift1m-filter'
algo = f'fusedann-{backend}'
run_name = algo

summary_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
if summary_path.exists():
    with summary_path.open(newline='') as existing:
        reader = csv.DictReader(existing)
        rows = [row for row in reader if not (row.get('algorithm') == algo and row.get('run_name') == run_name and row.get('k') == k)]

rows.append({
    'dataset': dataset,
    'algorithm': algo,
    'run_name': run_name,
    'k': k,
    'qps': 'inf' if math.isinf(qps) else f"{qps:.6f}",
    'recall': f"{recall:.6f}",
})

fieldnames = ['dataset', 'algorithm', 'run_name', 'k', 'qps', 'recall']
rows.sort(key=lambda row: (row['algorithm'], row['run_name']))

with summary_path.open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"📊 Stored fusedann-{backend} metrics in {summary_path}")
PY
  done
}

function ensure_parlayivf_beamsearch_patch() {
  local target="$1"
  if [[ ! -f "$target" ]]; then
    echo "❌ Missing ParlayIVF source at $target" >&2
    return 1
  fi
  python3 - "$target" <<'PY'
import pathlib, sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
patch_marker = "lanes = max(len(self._beam_widths), len(self._search_limits), len(self._max_degree))"

if patch_marker in text:
    print("ℹ️  ParlayIVF beam-search patch already applied.")
    raise SystemExit(0)

old = """    def set_beamsearch_params(self, k=10):\n        for i in range(3):\n            self.index.set_query_params(wp.QueryParams(k, self._beam_widths[i], 1.35, self._search_limits[i], self._max_degree[i]), i)\n        \n    \n"""

new = """    def set_beamsearch_params(self, k=10):\n        lanes = max(len(self._beam_widths), len(self._search_limits), len(self._max_degree))\n        if lanes == 0:\n            raise ValueError(\"Beam search parameters incomplete\")\n\n        def pick(seq, idx):\n            if not seq:\n                raise ValueError(\"Beam search parameter sequence missing entries\")\n            return seq[idx] if idx < len(seq) else seq[-1]\n\n        for i in range(lanes):\n            self.index.set_query_params(\n                wp.QueryParams(\n                    k,\n                    pick(self._beam_widths, i),\n                    1.35,\n                    pick(self._search_limits, i),\n                    pick(self._max_degree, i),\n                ),\n                i,\n            )\n        \n    \n"""

if old not in text:
    print("❌ Unable to locate ParlayIVF beam-search block for patching.", file=sys.stderr)
    raise SystemExit(1)

path.write_text(text.replace(old, new, 1))
print("✅ Applied ParlayIVF beam-search multi-lane patch.")
PY
}

if ${FRESH}; then
  progress "Cleaning previous artefacts"
  rm -rf "${DATA_ROOT}/fused_cache" "${BIGANN_ROOT}" "${ROOT_DIR}/results"
fi

if [[ "$FUSED_MODE" != "none" ]]; then
  if [[ "$FUSED_MODE" == "efanna" || "$FUSED_MODE" == "both" ]]; then
    progress "Running fusedann (EFANNA backend)"
    run_fused_backend efanna "EFANNA"
  fi
  if [[ "$FUSED_MODE" == "parlay" || "$FUSED_MODE" == "both" ]]; then
    progress "Running fusedann (Parlay backend)"
    run_fused_backend parlay "Parlay"
  fi
fi

progress "Ensuring big-ann checkout"
if [[ ! -d "${BIGANN_ROOT}/.git" ]]; then
  git clone https://github.com/harsha-simhadri/big-ann-benchmarks.git "${BIGANN_ROOT}"
fi

progress "Applying SIFT1M dataset patch"
if (cd "${BIGANN_ROOT}" && patch -p1 --forward --silent < "${BIGANN_PATCH}"); then
  echo "✅ big-ann patch applied"
else
  if grep -q "sift1m-filter" "${BIGANN_ROOT}/benchmark/datasets.py"; then
    echo "ℹ️  big-ann patch already in place"
  else
    echo "❌ Failed to apply ${BIGANN_PATCH}." >&2
    exit 1
  fi
fi

if ! ensure_parlayivf_beamsearch_patch "$PARLAYIVF_SOURCE"; then
  echo "❌ Unable to ensure ParlayIVF beam-search patch." >&2
  exit 1
fi

persist_fused_metrics

function download_and_extract_zip() {
  local url="$1"
  local dest="$2"
  local tmpdir
  tmpdir=$(mktemp -d)
  local archive="${tmpdir}/archive.zip"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$archive"
  else
    wget -O "$archive" "$url"
  fi
  unzip -q "$archive" -d "$tmpdir"
  local extracted
  extracted=$(find "$tmpdir" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  mv "$extracted" "$dest"
  rm -rf "$tmpdir"
}

function ensure_parlayann_archive_tree() {
  if [[ -d "${PARLAYANN_DIR}/python" ]]; then
    return
  fi
  echo "ℹ️  Downloading ParlayANN archive fallback (${PARLAYANN_COMMIT})."
  download_and_extract_zip "${PARLAYANN_ARCHIVE}" "${PARLAYANN_DIR}"
}

function ensure_parlaylib_tree() {
  if [[ -d "${PARLAYANN_DIR}/parlaylib/include" ]]; then
    return
  fi
  echo "ℹ️  Fetching parlaylib archive (${PARLAYLIB_COMMIT})."
  download_and_extract_zip "${PARLAYLIB_ARCHIVE}" "${PARLAYANN_DIR}/parlaylib"
}

function ensure_parlayann_wrapper() {
  local have_git=false
  local have_sources=false
  if [[ -d "${PARLAYANN_DIR}/.git" ]]; then
    have_git=true
  fi
  if [[ -d "${PARLAYANN_DIR}/python" ]]; then
    have_sources=true
  fi

  if [[ "$have_git" == true ]]; then
    if ! (cd "${PARLAYANN_DIR}" && git fetch --tags >/dev/null 2>&1 && git checkout --force "${PARLAYANN_COMMIT}" && git reset --hard "${PARLAYANN_COMMIT}" >/dev/null && git clean -fdx >/dev/null); then
      echo "⚠️  Unable to sync ParlayANN via git; removing checkout."
      rm -rf "${PARLAYANN_DIR}"
      have_git=false
      have_sources=false
    else
      if ! (cd "${PARLAYANN_DIR}" && git submodule update --init --recursive); then
        echo "⚠️  Failed to update ParlayANN submodules; removing checkout."
        rm -rf "${PARLAYANN_DIR}"
        have_git=false
        have_sources=false
      fi
    fi
  fi

  if [[ "$have_git" == false && "$have_sources" == false ]]; then
    if git clone "${PARLAYANN_REPO}" "${PARLAYANN_DIR}" >/dev/null 2>&1; then
      if (cd "${PARLAYANN_DIR}" && git checkout --force "${PARLAYANN_COMMIT}" && git reset --hard "${PARLAYANN_COMMIT}" >/dev/null && git clean -fdx >/dev/null && git submodule update --init --recursive); then
        have_git=true
        have_sources=true
      else
        echo "⚠️  git clone succeeded but sync failed; removing checkout."
        rm -rf "${PARLAYANN_DIR}"
      fi
    fi
    if [[ "$have_git" == false && "$have_sources" == false ]]; then
      echo "⚠️  git clone failed; falling back to archive."
      ensure_parlayann_archive_tree
      ensure_parlaylib_tree
      have_sources=true
    fi
  fi

  if [[ "$have_git" == true ]]; then
    have_sources=true
  fi

  if [[ "$have_sources" == true && "$have_git" == false ]]; then
    ensure_parlaylib_tree
  fi

  python3 -m pip install pybind11 >/dev/null
  local ext_suffix
  ext_suffix=$(python3 - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or "")
PY
)
  local module_path="${PARLAYANN_PYTHON}/_ParlayANNpy${ext_suffix}"
  if [[ ! -f "${module_path}" ]]; then
    (cd "${PARLAYANN_PYTHON}" && bash compile.sh)
  fi
  if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${PARLAYANN_PYTHON}"
  else
    case ":${PYTHONPATH}:" in
      *:"${PARLAYANN_PYTHON}":*) ;;
      *) export PYTHONPATH="${PARLAYANN_PYTHON}:${PYTHONPATH}" ;;
    esac
  fi
  local sys_lib="/usr/lib/x86_64-linux-gnu"
  if [[ -d "${sys_lib}" ]]; then
    if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
      export LD_LIBRARY_PATH="${sys_lib}"
    else
      case ":${LD_LIBRARY_PATH}:" in
        *:"${sys_lib}":*) ;;
        *) export LD_LIBRARY_PATH="${sys_lib}:${LD_LIBRARY_PATH}" ;;
      esac
    fi
    local stdcpp="${sys_lib}/libstdc++.so.6"
    if [[ -f "${stdcpp}" ]]; then
      if [[ -z "${LD_PRELOAD:-}" ]]; then
        export LD_PRELOAD="${stdcpp}"
      else
        case ":${LD_PRELOAD}:" in
          *:"${stdcpp}":*) ;;
          *) export LD_PRELOAD="${stdcpp}:${LD_PRELOAD}" ;;
        esac
      fi
    fi
  fi
}

if ${NEEDS_PARLAY}; then
  progress "Preparing ParlayANN IVF wrapper"
  ensure_parlayann_wrapper
  parlay_index_dir="${BIGANN_ROOT}/data/indices/filter/parlayivf/sift1m-filter"
  if [[ -d "${parlay_index_dir}" ]]; then
    echo "ℹ️  Removing stale ParlayIVF index at ${parlay_index_dir}"
    rm -rf "${parlay_index_dir}"
  fi
fi

function summarize_algo() {
  local algo_name=$1
  python3 - "${BIGANN_ROOT}" "${algo_name}" "${COUNT}" "${SUMMARY_FILE}" <<'PY'
import os
import sys
import numpy as np
from pathlib import Path
import csv

root = Path(sys.argv[1]).resolve()
algo = sys.argv[2]
k = int(sys.argv[3])
summary_path = Path(sys.argv[4]).resolve()
dataset = "sift1m-filter"

sys.path.insert(0, str(root))
os.chdir(root)

from benchmark.results import load_all_results  # pylint: disable=wrong-import-position
from benchmark.datasets import DATASETS  # pylint: disable=wrong-import-position

records = []
ds = DATASETS[dataset]()
gt, _ = ds.get_groundtruth(k)
gt = gt[:, :k]

for props, handle in load_all_results(dataset, k, neurips23track='filter'):
    if props.get('algo') != algo:
        continue
    neighbors = handle['neighbors'][:]
    if neighbors.size == 0:
        continue
    total = 0
    for i, row in enumerate(neighbors):
        total += np.intersect1d(row, gt[i], assume_unique=False).size
    recall = total / (neighbors.shape[0] * neighbors.shape[1])
    best_time = float(props.get('best_search_time', 0.0))
    qps = neighbors.shape[0] / best_time if best_time > 0 else float('inf')
    records.append((props.get('name', 'unknown'), qps, recall))

if not records:
    print(f"[warn] No sift1m-filter results found for {algo}")
else:
    print(f"Summary for {algo}:")
    for name, qps, recall in sorted(records):
        print(f"  {name}: QPS={qps:.2f}, Recall@{k}={recall:.4f}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if summary_path.exists():
        with summary_path.open(newline='') as existing:
            reader = csv.DictReader(existing)
            rows = [row for row in reader if not (row.get('algorithm') == algo and row.get('k') == str(k))]
    for name, qps, recall in sorted(records):
        rows.append({
            'dataset': dataset,
            'algorithm': algo,
            'run_name': name,
            'k': str(k),
            'qps': f"{qps:.6f}",
            'recall': f"{recall:.6f}",
        })
    fieldnames = ['dataset', 'algorithm', 'run_name', 'k', 'qps', 'recall']
    with summary_path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row['algorithm'], row['run_name'])))
PY
}

REQ_PATCH_FILE="${BIGANN_ROOT}/requirements_py3.10.txt"
python3 - "$REQ_PATCH_FILE" <<'PY'
import pathlib, sys

path = pathlib.Path(sys.argv[1])
targets = {
  "matplotlib": "3.9.2",
  "numpy": "2.1.3",
  "pyyaml": "6.0.1",
  "scipy": "1.14.1",
  "scikit-learn": "1.5.2",
  "pandas": "2.2.3",
  "h5py": "3.12.1",
  "psutil": "6.0.0",
}

def rewrite(line: str) -> str:
  stripped = line.strip()
  if not stripped or stripped.startswith('#'):
    return line
  if '==' not in stripped:
    return line
  name = stripped.split('==', 1)[0]
  if name in targets:
    return f"{name}=={targets[name]}"
  return line

lines = [rewrite(line) for line in path.read_text().splitlines()]
path.write_text("\n".join(lines) + "\n")
PY

progress "Installing big-ann Python deps"
if ! python3 - <<'PY'
import sys, configparser
sys.exit(0 if hasattr(configparser, 'SafeConfigParser') else 1)
PY
then
  python3 -m pip install "configparser==5.3.0"
fi
python3 -m pip install -r "${BIGANN_ROOT}/requirements_py3.10.txt"

progress "Converting SIFT1M to big-ann format"
python3 "${SCRIPT_DIR}/prepare_sift1m_bigann.py" \
  --sift-root "${DATA_ROOT}" \
  --bigann-root "${BIGANN_ROOT}"

progress "Executing opponent algorithms"
for algo in "${OPPONENTS[@]}"; do
  echo "→ Running ${algo}"
  CMD=(python3 run.py
    --neurips23track filter
    --dataset sift1m-filter
    --algorithm "${algo}"
    --nodocker
    --runs "${RUNS}"
    -k "${COUNT}")
  if ${FORCE_OPP}; then
    CMD+=(--force)
  fi
  (cd "${BIGANN_ROOT}" && "${CMD[@]}")
  summarize_algo "${algo}"
  echo "✔️  ${algo} completed"
  echo
done

if ! ${SKIP_PLOT}; then
  progress "Generating recall/QPS plot"
  (cd "${BIGANN_ROOT}" && python3 plot.py --dataset sift1m-filter --neurips23track filter)
fi

printf '\n🎯 Benchmark sweep complete. Results live under %s/results.\n' "${BIGANN_ROOT}"

progress "All tasks finished"