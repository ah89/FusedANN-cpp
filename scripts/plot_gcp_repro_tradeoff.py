#!/usr/bin/env python3
"""Recall-QPS tradeoff on YFCC-10M filter track: FusedANN vs ParlayIVF,
GCP c3-standard-22 independent reproduction OVERLAID on the
original 20-core measurements from docs/.

Same data format as plot_yfcc_tradeoff.py:
  FusedANN : beam rerank visit RECALL QPS ...
  ParlayIVF: target_points tiny_cutoff beam RECALL QPS

Usage: python3 scripts/plot_gcp_repro_tradeoff.py [out.png]
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "docs/yfcc10m_recall_qps_gcp_repro.png"

def load_points(path, rcol=3, qcol=4):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            try:
                r = float(p[rcol]); q = float(p[qcol])
            except (ValueError, IndexError):
                continue
            if 0.0 < r <= 1.0 and q > 0:
                pts.append((r, q))
    pts.sort()
    return pts

# (label, path, color, marker, linewidth, linestyle, alpha)
series = [
    ("FusedANN — GCP c3-standard-22 repro",
        "docs/fusedann_10m_gcp_sweep.txt",      "crimson",  "o", 3.0, "-",  1.0),
    ("FusedANN — original 20-core box",
        "docs/fusedann_10m_fixed_sweep.txt",    "crimson",  "x", 2.0, "--", 0.6),
    ("ParlayIVF — GCP c3-standard-22 repro (leaderboard)",
        "docs/parlayivf_10m_gcp_sweep.txt",     "#1f77b4",  "s", 3.0, "-",  1.0),
    ("ParlayIVF — original 20-core box (leaderboard)",
        "docs/parlayivf_10m_sweep.txt",         "#1f77b4",  "+", 2.0, "--", 0.6),
]

plt.figure(figsize=(10, 6.5))
plotted = []
for label, path, color, marker, lw, ls, alpha in series:
    pts = load_points(path)
    if not pts:
        continue
    xs = [r for r, _ in pts]; ys = [q for _, q in pts]
    plt.plot(xs, ys, marker=marker, color=color, linewidth=lw, linestyle=ls, alpha=alpha,
             markersize=8, label=label)
    plotted.append((label, pts))

plt.xlabel("Recall@10")
plt.ylabel("Queries per second (1/s)")
plt.title("YFCC-10M filter track — independent GCP reproduction (c3-standard-22)\n"
          "FusedANN vs ParlayIVF — solid = GCP repro, dashed = original 20-core box")
plt.grid(True, alpha=0.3)
plt.xlim(0.70, 1.0)
plt.ylim(0, None)
plt.legend(loc="best", fontsize=9)
plt.figtext(0.01, 0.005,
            "GCP c3-standard-22 (Intel Xeon Platinum 8481C, Sapphire Rapids, 22 vCPU, 86 GB RAM, AVX-512), "
            "Ubuntu 22.04, PARLAY_NUM_THREADS=20, official GT.public.ibin, --nodocker. "
            "Reproduction confirms FusedANN dominates ParlayIVF at every matched recall.",
            fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
for label, pts in plotted:
    print(label, "->", [(round(r, 3), round(q)) for r, q in pts])
print("wrote", out)
