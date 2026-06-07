#!/usr/bin/env python3
"""Run ParlayANN IVF^2 (the NeurIPS'23 filter-track 'parlayivf') natively on our
YFCC-10M files, on THIS machine, for an apples-to-apples comparison with FusedANN.

Builds the IVF^2 index with the leaderboard config, then sweeps query args and
reports recall@10 (vs official GT.public.ibin) and QPS for each operating point.
Writes points to logs/parlayivf_10m_sweep.txt.
"""
import os, sys, time, struct
import numpy as np

DATA = "data/yfcc10m"
BASE = f"{DATA}/base.10M.u8bin"
BASE_META = f"{DATA}/base.metadata.10M.spmat"
QVEC = f"{DATA}/query.public.100K.u8bin"
QMETA = f"{DATA}/query.metadata.public.100K.spmat"
GT = f"{DATA}/GT.public.ibin"
INDEX_DIR = os.environ.get("PARLAYIVF_INDEX_DIR", "build/parlayivf_idx_10m")
OUT = os.environ.get("PARLAYIVF_OUT", "logs/parlayivf_10m_sweep.txt")

import wrapper as wp

def load_u8bin(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("II", f.read(8))
        a = np.frombuffer(f.read(n * d), dtype=np.uint8).reshape(n, d)
    return np.ascontiguousarray(a)

def load_spmat_rows(path):
    with open(path, "rb") as f:
        rows, cols, nnz = struct.unpack("qqq", f.read(24))
        indptr = np.frombuffer(f.read((rows + 1) * 8), dtype=np.int64)
        indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
    return rows, indptr, indices

def load_ibin(path):
    with open(path, "rb") as f:
        n, k = struct.unpack("II", f.read(8))
        ids = np.frombuffer(f.read(n * k * 4), dtype=np.int32).reshape(n, k)
    return ids

print("loading queries + filters + GT ...", flush=True)
Xq = load_u8bin(QVEC)
nq = Xq.shape[0]
qrows, qindptr, qindices = load_spmat_rows(QMETA)
filters = [None] * nq
for qi in range(nq):
    tags = [int(t) for t in qindices[qindptr[qi]:qindptr[qi + 1]]]
    filters[qi] = wp.QueryFilter(*tags)
gt = load_ibin(GT)
print(f"  nq={nq} dim={Xq.shape[1]} gt={gt.shape}", flush=True)

# ---- Build IVF^2 index (leaderboard YFCC-10M config) ----
os.makedirs(INDEX_DIR, exist_ok=True)
cluster_size, cutoff, max_iter = 5000, 10000, 10
weight_classes = (100000, 400000)
bp = [wp.BuildParams(8, 200, 1.175), wp.BuildParams(10, 200, 1.175), wp.BuildParams(12, 200, 1.175)]
max_degree = (8, 10, 12)

index = wp.init_squared_ivf_index("Euclidian", "uint8")
index.set_max_iter(max_iter)
index.set_bitvector_cutoff(cutoff)
for i, b in enumerate(bp):
    index.set_build_params(b, i)
print("building IVF^2 index (fit_from_filename) ...", flush=True)
t0 = time.time()
index.fit_from_filename(BASE, BASE_META, cutoff, cluster_size, INDEX_DIR, weight_classes, False)
print(f"  built in {time.time()-t0:.1f}s", flush=True)

def recall_at_10(res):
    hit = 0
    for i in range(nq):
        hit += len(np.intersect1d(res[i], gt[i], assume_unique=False))
    return hit / (nq * 10.0)

# ---- Query-arg sweep (leaderboard sets + a few faster points) ----
sweep = [
    dict(target_points=15000, tiny_cutoff=100000, beam_widths=[90, 90, 90]),
    dict(target_points=15000, tiny_cutoff=60000,  beam_widths=[90, 90, 90]),
    dict(target_points=15000, tiny_cutoff=100000, beam_widths=[60, 60, 60]),
    dict(target_points=15000, tiny_cutoff=60000,  beam_widths=[50, 50, 50]),
    dict(target_points=15000, tiny_cutoff=100000, beam_widths=[40, 40, 40]),
    dict(target_points=7500,  tiny_cutoff=35000,  beam_widths=[55, 55, 55]),
    dict(target_points=5000,  tiny_cutoff=30000,  beam_widths=[40, 40, 40], search_limits=[500, 500, 500]),
    dict(target_points=3000,  tiny_cutoff=20000,  beam_widths=[30, 30, 30], search_limits=[500, 500, 500]),
    dict(target_points=2000,  tiny_cutoff=15000,  beam_widths=[20, 20, 20], search_limits=[500, 500, 500]),
]

os.makedirs("logs", exist_ok=True)
with open(OUT, "w") as fout:
    fout.write("# target_points tiny_cutoff beam recall qps\n")
    for qa in sweep:
        tp = qa["target_points"]; tc = qa["tiny_cutoff"]; bw = qa["beam_widths"]
        sl = qa.get("search_limits", list(weight_classes) + [3_000_000])
        index.set_target_points(tp); index.set_tiny_cutoff(tc); index.set_sq_target_points(tp)
        for i in range(3):
            index.set_query_params(wp.QueryParams(10, bw[i], 1.35, sl[i], max_degree[i]), i)
        t0 = time.time()
        res, _ = index.batch_filter_search(Xq, filters, nq, 10)
        dt = time.time() - t0
        try:
            index.reset()
        except Exception:
            pass
        res = np.asarray(res)
        rec = recall_at_10(res)
        qps = nq / dt
        line = f"{tp} {tc} {bw[0]} {rec:.4f} {qps:.2f}"
        print("  ", line, f"(search {dt:.2f}s)", flush=True)
        fout.write(line + "\n"); fout.flush()
print("DONE ->", OUT, flush=True)
