#!/usr/bin/env python3
"""Recall-QPS tradeoff on YFCC-10M filter track: FusedANN vs ParlayIVF,
BOTH measured on the same machine (same cores, same data, official GT) — an
apples-to-apples head-to-head, not a comparison against other hardware.

Both sweep files have whitespace columns with recall in column index 3 and
QPS in column index 4:
  FusedANN : beam rerank visit RECALL QPS coverage
  ParlayIVF: target_points tiny_cutoff beam RECALL QPS

Usage: python3 scripts/plot_yfcc_tradeoff.py [out.png]
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "docs/yfcc10m_recall_qps.png"

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

series = [
    ("FusedANN — uint8 + G=100 (seeded routing)", "docs/fusedann_10m_fixed_sweep.txt", "crimson", "o", 3.0, "-", 1.0),
    ("ParlayIVF (IVF², leaderboard config)", "docs/parlayivf_10m_sweep.txt", "#1f77b4", "s", 2.6, "-", 1.0),
]

plt.figure(figsize=(10, 6.5))
plotted = []
for label, path, color, marker, lw, ls, alpha in series:
    pts = load_points(path)
    if not pts:
        continue
    xs = [r for r, _ in pts]; ys = [q for _, q in pts]
    plt.plot(xs, ys, marker=marker, color=color, linewidth=lw, linestyle=ls, alpha=alpha,
             markersize=7, label=label)
    plotted.append((label, pts))

plt.xlabel("Recall@10")
plt.ylabel("Queries per second (1/s)")
plt.title("YFCC-10M filter track — FusedANN vs ParlayIVF (same 20-core box, official GT)")
plt.grid(True, alpha=0.3)
plt.xlim(0.70, 1.0)
plt.ylim(0, None)
plt.legend(loc="best", fontsize=10)
plt.figtext(0.01, 0.005,
            "Both measured on the same machine: full 10M base, 100K public queries, official GT.public.ibin, "
            "PARLAY_NUM_THREADS=20. Up and to the right is better.",
            fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
for label, pts in plotted:
    print(label, "->", [(round(r, 3), round(q)) for r, q in pts])
print("wrote", out)
