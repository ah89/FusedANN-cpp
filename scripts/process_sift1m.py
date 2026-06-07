#!/usr/bin/env python3
"""Generate attribute bitmaps for the SIFT1M dataset.

This script reads the standard ANN-Benchmarks SIFT1M files (`*.fvecs`) and
derives categorical attribute bitmaps expected by `fusedann_efanna.cpp`.
Each descriptor is reduced to a scalar score, bucketed into one of ten
percentile-based categories, and written as a one-hot uint8 vector (`*.bvecs`).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import numpy as np
import shutil

ATTR_CATEGORIES = 10
ATTR_DIM = 10


def read_fvecs(path: pathlib.Path) -> np.ndarray:
    """Load an fvecs file into a float32 NumPy array."""
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    dim = int(data[0])
    if dim <= 0:
        raise ValueError(f"Invalid dimension {dim} reported by {path}.")

    try:
        vectors = data.reshape(-1, dim + 1)
    except ValueError as exc:
        raise ValueError(f"{path} size is not a multiple of vector stride.") from exc

    dims = vectors[:, 0]
    if not np.all(dims == dim):
        unique = np.unique(dims)
        raise ValueError(
            f"Inconsistent vector dimensions found in {path}. First dim={dim}, unique dims={unique}."
        )

    # Reinterpret the payload as float32 without copying, then copy to detach from memmap.
    float_view = vectors[:, 1:].view(np.float32)
    return np.array(float_view, copy=True, dtype=np.float32)


def write_bvecs(path: pathlib.Path, vectors: np.ndarray) -> None:
    """Persist uint8 vectors in bvecs format (dim header + payload)."""
    if vectors.dtype != np.uint8:
        raise ValueError("write_bvecs expects uint8 vectors.")

    if vectors.ndim != 2:
        raise ValueError("write_bvecs expects a 2-D array.")

    n, dim = vectors.shape
    dtype = np.dtype([("dim", np.int32), ("vec", np.uint8, (dim,))])
    packed = np.empty(n, dtype=dtype)
    packed["dim"] = dim
    packed["vec"] = vectors
    packed.tofile(path)


def compute_percentile_bins(scores: np.ndarray, categories: int) -> np.ndarray:
    percentiles = np.linspace(0, 100, categories + 1)
    boundaries = np.percentile(scores, percentiles[1:-1])
    return boundaries.astype(np.float32, copy=True)


def assign_categories(scores: np.ndarray, bins: np.ndarray) -> np.ndarray:
    raw = np.digitize(scores, bins, right=False)
    return raw.clip(0, bins.size)


def one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    eye = np.eye(depth, dtype=np.uint8)
    return eye[indices]


def generate_attribute_bitmaps(base_vectors: np.ndarray,
                               query_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_scores = base_vectors.mean(axis=1)
    bins = compute_percentile_bins(base_scores, ATTR_CATEGORIES)
    base_categories = assign_categories(base_scores, bins)
    base_attrs = one_hot(base_categories, ATTR_DIM)

    query_scores = query_vectors.mean(axis=1)
    query_categories = assign_categories(query_scores, bins)
    query_attrs = one_hot(query_categories, ATTR_DIM)

    return base_attrs, query_attrs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SIFT1M attribute bitmaps.")
    parser.add_argument(
        "--raw-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/sift1m/raw"),
        help="Path containing the downloaded SIFT1M *.fvecs files.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/sift1m"),
        help="Directory where the attribute *.bvecs files will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing attribute files if they are present.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    required_files = {
        "sift_base.fvecs": raw_dir / "sift_base.fvecs",
        "sift_query.fvecs": raw_dir / "sift_query.fvecs",
    }

    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        print(f"Missing required input files: {joined}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "sift_base_attrs.bvecs": out_dir / "sift_base_attrs.bvecs",
        "sift_query_attrs.bvecs": out_dir / "sift_query_attrs.bvecs",
    }

    if not args.force:
        existing = [name for name, path in outputs.items() if path.exists()]
        if existing:
            joined = ", ".join(existing)
            print(
                f"Refusing to overwrite existing attribute files ({joined}). Use --force to regenerate.",
                file=sys.stderr,
            )
            return 1

    base_path = required_files["sift_base.fvecs"]
    query_path = required_files["sift_query.fvecs"]

    base_out = outputs["sift_base_attrs.bvecs"]
    query_out = outputs["sift_query_attrs.bvecs"]

    base_vectors = read_fvecs(base_path)
    query_vectors = read_fvecs(query_path)
    if base_vectors.size == 0 or query_vectors.size == 0:
        raise ValueError("Failed to read SIFT vectors for attribute generation.")

    base_attrs, query_attrs = generate_attribute_bitmaps(base_vectors, query_vectors)

    write_bvecs(base_out, base_attrs)
    write_bvecs(query_out, query_attrs)

    unique_base = np.unique(base_attrs.view([('vec', base_attrs.dtype, (ATTR_DIM,))]))
    print(f"Wrote base attributes ({ATTR_DIM} dims) to {base_out}. Unique combinations: {unique_base.size}")

    unique_query = np.unique(query_attrs.view([('vec', query_attrs.dtype, (ATTR_DIM,))]))
    print(f"Wrote query attributes ({ATTR_DIM} dims) to {query_out}. Unique combinations: {unique_query.size}")

    del base_vectors
    del query_vectors
    del base_attrs
    del query_attrs

    # For convenience, copy the original fvecs into out_dir if not already there.
    for name, src in required_files.items():
        dst = out_dir / name
        if dst.exists():
            continue
        try:
            relative = os.path.relpath(src, dst.parent)
            dst.symlink_to(relative)
        except (AttributeError, NotImplementedError, OSError):
            shutil.copy2(src, dst)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
