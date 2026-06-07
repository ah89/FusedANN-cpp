import numpy as np
import os

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def ibin_read(fname):
    with open(fname, "rb") as f:
        n, d = np.fromfile(f, dtype='uint32', count=2)
        print(f"GT shape: {n} x {d}")
        # The file format is: n (uint32), d (uint32), indices (n*d int32), distances (n*d float32)
        # We only need indices.
        indices = np.fromfile(f, dtype='int32', count=n * d)
        return indices.reshape(n, d)

gt_path = "data/yfcc100M/GT.public.ibin"
if os.path.exists(gt_path):
    try:
        gt = ibin_read(gt_path)
        print(f"Max index in GT: {np.max(gt)}")
        print(f"Min index in GT: {np.min(gt)}")
    except Exception as e:
        print(f"Error reading GT: {e}")
else:
    print("GT file not found")
