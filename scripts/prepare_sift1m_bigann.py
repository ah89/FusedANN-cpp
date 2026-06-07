#!/usr/bin/env python3
"""Convert the local SIFT1M assets into the big-ann benchmark format."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import numpy as np

ATTR_DIM = 10  # must match scripts/process_sift1m.py


def read_fvecs(path: pathlib.Path) -> np.ndarray:
    with path.open("rb") as handle:
        dim_arr = np.fromfile(handle, dtype=np.int32, count=1)
    if dim_arr.size != 1:
        raise ValueError(f"Unable to read the dimension header from {path}.")
    dim = int(dim_arr[0])
    record_dtype = np.dtype([("dim", np.int32), ("vec", np.float32, (dim,))])
    data = np.fromfile(path, dtype=record_dtype)
    if data.size == 0:
        raise ValueError(f"{path} appears empty.")
    if not np.all(data["dim"] == dim):
        raise ValueError(f"{path} has inconsistent dimensions in the header column.")
    return np.array(data["vec"], copy=True)


def read_bvecs(path: pathlib.Path) -> np.ndarray:
    if path.stat().st_size == 0:
        raise ValueError(f"{path} appears empty.")

    with path.open("rb") as handle:
        dim_arr = np.fromfile(handle, dtype=np.int32, count=1)
    if dim_arr.size != 1:
        raise ValueError(f"Unable to read the dimension header from {path}.")
    dim = int(dim_arr[0])

    dtype = np.dtype([("dim", np.int32), ("vec", np.uint8, (dim,))])
    data = np.fromfile(path, dtype=dtype)
    if not np.all(data["dim"] == dim):
        raise ValueError(f"{path} has inconsistent dimensions in the header column.")
    return np.array(data["vec"], copy=True)


def read_ivecs(path: pathlib.Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} appears empty.")

    dim = raw[0]
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(f"{path} size is not divisible by its stride {stride}.")
    reshaped = raw.reshape(-1, stride)
    if not np.all(reshaped[:, 0] == dim):
        raise ValueError(f"{path} has inconsistent dimensions in the header column.")
    return reshaped[:, 1:].astype(np.int32)


def write_fbin(path: pathlib.Path, data: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.array(data.shape, dtype=np.uint32).tofile(handle)
        data.astype(np.float32, copy=False).tofile(handle)


def write_ibin(path: pathlib.Path, ids: np.ndarray, dists: np.ndarray) -> None:
    if ids.shape != dists.shape:
        raise ValueError("ids and dists must share the same shape.")
    with path.open("wb") as handle:
        np.array(ids.shape, dtype=np.uint32).tofile(handle)
        np.ascontiguousarray(ids, dtype=np.int32).tofile(handle)
        np.ascontiguousarray(dists, dtype=np.float32).tofile(handle)


def dense_to_csr(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("dense_to_csr expects a 2-D array.")
    rows, cols = np.nonzero(matrix)
    data = matrix[rows, cols].astype(np.float32, copy=False)
    counts = np.bincount(rows, minlength=matrix.shape[0])
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return data, cols.astype(np.int32, copy=False), indptr


def write_spmat(path: pathlib.Path, matrix: np.ndarray) -> None:
    data, indices, indptr = dense_to_csr(matrix)
    header = np.array([matrix.shape[0], matrix.shape[1], data.size], dtype=np.int64)
    with path.open("wb") as handle:
        header.tofile(handle)
        indptr.tofile(handle)
        indices.tofile(handle)
        data.tofile(handle)


def decode_categories(attrs: np.ndarray) -> np.ndarray:
    if attrs.ndim != 2 or attrs.shape[1] != ATTR_DIM:
        raise ValueError("Attribute bitmap has unexpected shape.")
    labels = attrs.argmax(axis=1)
    # Require that every argmax entry corresponds to a non-zero bit; this guards against
    # accidental all-zero rows that would make filtering ambiguous.
    if not np.all(attrs[np.arange(attrs.shape[0]), labels] > 0):
        raise ValueError("Attribute vectors must be one-hot encoded.")
    return labels.astype(np.int32, copy=False)


def brute_force_knn(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    base_indices: np.ndarray,
    k: int,
    block_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if base_vectors.shape[0] == 0 or query_vectors.shape[0] == 0:
        raise ValueError("brute_force_knn expects non-empty base and query sets.")
    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, base_vectors.shape[0])
    base = np.ascontiguousarray(base_vectors, dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    base_norms = np.sum(base * base, axis=1, dtype=np.float32)

    ids = np.empty((queries.shape[0], k), dtype=np.int32)
    dists = np.empty((queries.shape[0], k), dtype=np.float32)

    for start in range(0, queries.shape[0], block_size):
        end = min(start + block_size, queries.shape[0])
        block = queries[start:end]
        block_norms = np.sum(block * block, axis=1, keepdims=True, dtype=np.float32)
        dots = block @ base.T
        block_dists = block_norms + base_norms[None, :] - 2.0 * dots
        np.maximum(block_dists, 0.0, out=block_dists)

        kth = min(k - 1, block_dists.shape[1] - 1)
        part_idx = np.argpartition(block_dists, kth=kth, axis=1)[:, :k]
        part_dists = np.take_along_axis(block_dists, part_idx, axis=1)
        order = np.argsort(part_dists, axis=1)
        sorted_idx = np.take_along_axis(part_idx, order, axis=1)
        sorted_dists = np.take_along_axis(part_dists, order, axis=1)

        ids[start:end] = base_indices[sorted_idx]
        dists[start:end] = sorted_dists

    return ids, dists


def compute_filtered_groundtruth(
    base: np.ndarray,
    queries: np.ndarray,
    base_attrs: np.ndarray,
    query_attrs: np.ndarray,
    k: int = 10,
    block_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    base_labels = decode_categories(base_attrs)
    query_labels = decode_categories(query_attrs)

    gt_ids = np.full((queries.shape[0], k), -1, dtype=np.int32)
    gt_dists = np.full((queries.shape[0], k), np.inf, dtype=np.float32)

    for attr in range(ATTR_DIM):
        base_idx = np.nonzero(base_labels == attr)[0]
        query_idx = np.nonzero(query_labels == attr)[0]
        if query_idx.size == 0:
            continue
        if base_idx.size < k:
            raise ValueError(
                f"Attribute bucket {attr} has only {base_idx.size} base vectors (< k={k})."
            )

        ids, dists = brute_force_knn(
            base[base_idx],
            queries[query_idx],
            base_idx,
            k=k,
            block_size=block_size,
        )
        gt_ids[query_idx] = ids
        gt_dists[query_idx] = dists

    missing = np.nonzero(gt_ids[:, 0] < 0)[0]
    if missing.size:
        raise ValueError(f"Queries without attribute matches found: {missing.size}")
    return gt_ids, gt_dists


def convert_dataset(
    sift_root: pathlib.Path,
    bigann_dir: pathlib.Path,
) -> None:
    base = read_fvecs(sift_root / "sift_base.fvecs")
    queries = read_fvecs(sift_root / "sift_query.fvecs")
    base_attrs = read_bvecs(sift_root / "sift_base_attrs.bvecs")
    query_attrs = read_bvecs(sift_root / "sift_query_attrs.bvecs")

    if base.shape[0] != base_attrs.shape[0]:
        raise ValueError("Base attribute count does not match vector count.")
    if queries.shape[0] != query_attrs.shape[0]:
        raise ValueError("Query attribute count does not match vector count.")
    if base_attrs.shape[1] != ATTR_DIM or query_attrs.shape[1] != ATTR_DIM:
        raise ValueError("Unexpected attribute dimensionality for SIFT1M.")

    print("Computing attribute-filtered groundtruth (Euclidean)")
    gt, gt_dists = compute_filtered_groundtruth(base, queries, base_attrs, query_attrs)

    bigann_dir.mkdir(parents=True, exist_ok=True)

    write_fbin(bigann_dir / "sift1m.base.fbin", base)
    write_fbin(bigann_dir / "sift1m.query.public.10K.fbin", queries)
    write_ibin(bigann_dir / "sift1m.GT.public.ibin", gt, gt_dists)
    write_spmat(bigann_dir / "sift1m.base.metadata.spmat", base_attrs)
    write_spmat(bigann_dir / "sift1m.query.metadata.public.spmat", query_attrs)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit big-ann formatted SIFT1M files.")
    parser.add_argument(
        "--sift-root",
        type=pathlib.Path,
        default=pathlib.Path("data/sift1m"),
        help="Directory containing sift_base/query.{fvecs,bvecs} and groundtruth.",
    )
    parser.add_argument(
        "--bigann-root",
        type=pathlib.Path,
        default=pathlib.Path("third_party/big-ann-benchmarks"),
        help="Path to the big-ann-benchmarks checkout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    sift_root = args.sift_root
    bigann_root = args.bigann_root / "data" / "sift1m-filter"
    required = [
        sift_root / "sift_base.fvecs",
        sift_root / "sift_query.fvecs",
        sift_root / "sift_base_attrs.bvecs",
        sift_root / "sift_query_attrs.bvecs",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing SIFT1M prerequisite files:\n{joined}")

    convert_dataset(sift_root, bigann_root)
    print(f"Wrote big-ann assets to {bigann_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
