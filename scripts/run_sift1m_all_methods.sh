#!/usr/bin/env bash
# Run fusedann (EFANNA + Parlay) and all NeurIPS'23 filter opponents on the
# SIFT1M dataset, ensuring every method shares identical queries, filters, and
# ground truth. Prints a consolidated summary for quick comparison.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BENCH_SCRIPT="${SCRIPT_DIR}/run_sift1m_filter_benchmarks.sh"
DEFAULT_OPPONENTS="faiss,faissplus,parlayivf,pyanns,puck"
SUMMARY_FILE="${ROOT_DIR}/third_party/big-ann-benchmarks/results/sift1m-filter-summary.csv"

if [[ ! -x "${BENCH_SCRIPT}" ]]; then
  echo "Missing ${BENCH_SCRIPT} (make sure you're running from the repo root)." >&2
  exit 1
fi

ARGS=("$@")
HAS_OPP=false
HAS_FUSED=false
OPP_VALUE="${DEFAULT_OPPONENTS}"

for ((idx = 0; idx < ${#ARGS[@]}; ++idx)); do
  arg="${ARGS[$idx]}"
  if [[ "${arg}" == "--opponents" ]]; then
    HAS_OPP=true
    next=$((idx + 1))
    if (( next < ${#ARGS[@]} )); then
      OPP_VALUE="${ARGS[$next]}"
    else
      echo "--opponents requires a value" >&2
      exit 1
    fi
  elif [[ "${arg}" == "--fused-backend" ]]; then
    HAS_FUSED=true
  fi
done

CMD=("${BENCH_SCRIPT}")
if ! ${HAS_FUSED}; then
  CMD+=("--fused-backend" "both")
fi
if ! ${HAS_OPP}; then
  CMD+=("--opponents" "${DEFAULT_OPPONENTS}")
fi
CMD+=("$@")

printf '\n🚀 Running fusedann + NeurIPS opponents on SIFT1M...\n'
"${CMD[@]}"

if [[ ! -f "${SUMMARY_FILE}" ]]; then
  echo "Expected summary file ${SUMMARY_FILE} was not created." >&2
  exit 1
fi

IFS=',' read -r -a OPP_LIST <<< "${OPP_VALUE}"
METHODS=("fusedann-efanna" "fusedann-parlay")
for opp in "${OPP_LIST[@]}"; do
  [[ -n "${opp}" ]] && METHODS+=("${opp}")
done

python3 - "${SUMMARY_FILE}" "${METHODS[@]}" <<'PY'
import csv
import sys
from collections import OrderedDict
from pathlib import Path

path = Path(sys.argv[1])
algos = list(dict.fromkeys(sys.argv[2:]))  # preserve order, drop duplicates

rows = []
if path.exists():
    with path.open(newline='') as src:
        reader = csv.DictReader(src)
        for row in reader:
            if row.get('dataset') == 'sift1m-filter' and row.get('algorithm') in algos:
                rows.append(row)

if not rows:
    print("[warn] No sift1m-filter entries found in summary file.")
    sys.exit(0)

print("\n📊 Consolidated results (Recall@K and QPS share identical queries + filters):")
header = f"{ 'Algorithm':<20} {'Run Name':<20} {'Recall':>10} {'QPS':>12}"
print(header)
print('-' * len(header))
for algo in algos:
    algo_rows = [row for row in rows if row['algorithm'] == algo]
    if not algo_rows:
        print(f"{algo:<20} {'(missing)':<20} {'-':>10} {'-':>12}")
        continue
    for row in algo_rows:
        recall = float(row['recall']) if row['recall'] not in {'inf', 'nan', ''} else float('nan')
        qps_val = row['qps']
        try:
            qps = float(qps_val)
            qps_fmt = 'inf' if qps == float('inf') else f"{qps:.2f}"
        except ValueError:
            qps_fmt = qps_val or '-'
        print(f"{algo:<20} {row['run_name']:<20} {recall:>10.4f} {qps_fmt:>12}")
print()
PY
