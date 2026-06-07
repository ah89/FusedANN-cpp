#!/usr/bin/env python3

import os
import re
import subprocess
import sys
from pathlib import Path


def run_and_parse_recall(cmd: list[str], env: dict[str, str]) -> float:
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = proc.stdout
    if proc.returncode != 0:
        print(out)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")

    m = re.search(r"FusedANN recall:\s*([0-9]*\.?[0-9]+)", out)
    if not m:
        print(out)
        raise RuntimeError("Could not parse recall from output")
    return float(m.group(1))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    exe = root / "build" / "fusedann_parlayann"
    if not exe.exists():
        raise RuntimeError(f"Missing executable: {exe}")

    data = root / "data" / "sift1m"
    base = data / "sift_base.fvecs"
    query = data / "sift_query.fvecs"
    base_attrs = data / "sift_base_attrs.bvecs"
    query_attrs = data / "sift_query_attrs.bvecs"
    gt = data / "raw" / "sift_groundtruth.ivecs"
    cache_dir = data / "fused_cache"

    for p in [base, query, base_attrs, query_attrs, gt]:
        if not p.exists():
            raise RuntimeError(f"Missing required file: {p}")

    cmd = [
        str(exe),
        str(base),
        str(base_attrs),
        str(query),
        str(query_attrs),
        str(gt),
        str(cache_dir),
    ]

    base_env = os.environ.copy()
    base_env.setdefault("PARLAY_NUM_THREADS", "8")
    # Keep settings fixed so both runs use the same cached graph.
    base_env.setdefault("FUSEDANN_PARLAY_GRAPH_DEGREE", "24")
    base_env.setdefault("FUSEDANN_PARLAY_BEAM_WIDTH", "42")
    base_env.setdefault("FUSEDANN_PARLAY_VISIT_LIMIT", "84")
    base_env.setdefault("FUSEDANN_PARLAY_RERANK_K", "15")
    base_env.setdefault("FUSEDANN_PARLAY_FINAL_CAND_MULT", "1.05")
    base_env.setdefault("FUSEDANN_PARLAY_DEGREE_LIMIT", "16")

    env0 = dict(base_env)
    env0["FUSEDANN_PARTITION_K"] = "0"
    recall0 = run_and_parse_recall(cmd, env0)

    env6 = dict(base_env)
    env6["FUSEDANN_PARTITION_K"] = "6"
    recall6 = run_and_parse_recall(cmd, env6)

    diff = abs(recall0 - recall6)
    eps = 1e-6

    print(f"recall(K=0)={recall0:.8f}")
    print(f"recall(K=6)={recall6:.8f}")
    print(f"abs_diff={diff:.8g} (eps={eps})")

    if diff > eps:
        print("FAIL: recall differs between partition_k=0 and 6")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}")
        raise
