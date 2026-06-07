#!/usr/bin/env python3
"""Recall-QPS tradeoff for FusedANN on the SIFT1M categorical-filter benchmark
(10-way percentile attribute, recall@10 vs the attribute-filtered ground truth),
measured on this machine. Single method: SIFT1M uses categorical *equality*
filtering, so the bag-of-tags ParlayIVF filter track does not apply.

Data columns: beam rerank visit RECALL QPS  (recall idx 3, qps idx 4).
Usage: python3 scripts/plot_sift_tradeoff.py [out.png]
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "docs/sift1m_recall_qps.png"

def load_points(path, rcol=3, qcol=4):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("DONE"):
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

pts = load_points("docs/sift1m_sweep.txt")

plt.figure(figsize=(10, 6.5))
if pts:
    xs = [r for r, _ in pts]; ys = [q for _, q in pts]
    plt.plot(xs, ys, marker="o", color="crimson", linewidth=3.0, markersize=7,
             label="FusedANN — uint8 (this work)")
    for r, q in pts:
        plt.annotate(f"{r:.3f}", (r, q), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7, color="gray")

plt.xlabel("Recall@10")
plt.ylabel("Queries per second (1/s)")
plt.title("SIFT1M categorical-filter track — FusedANN recall–QPS (20-core box)")
plt.grid(True, alpha=0.3)
plt.xlim(0.88, 1.0)
plt.ylim(0, None)
plt.legend(loc="best", fontsize=10)
plt.figtext(0.01, 0.005,
            "SIFT1M base (1M×128), 10K queries, 10-way percentile categorical attribute; "
            "recall@10 vs attribute-filtered GT, PARLAY_NUM_THREADS=20. Up and to the right is better.",
            fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
print("FusedANN SIFT1M ->", [(round(r, 3), round(q)) for r, q in pts])
print("wrote", out)
