#!/usr/bin/env python3
"""Validate that the SIFT1M assets used by fusedann and big-ann share the same shapes."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from prepare_sift1m_bigann import (  # type: ignore  # noqa: E402
    ATTR_DIM,
    convert_dataset,
    read_bvecs,
    read_fvecs,
)


def read_fbin_header(path: pathlib.Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"{path} missing fbin header")
    return int(header[0]), int(header[1])


def read_ibin_header(path: pathlib.Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"{path} is missing the 2xuint32 header")
    return int(header[0]), int(header[1])


def read_spmat_header(path: pathlib.Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int64, count=3)
    if header.size != 3:
        raise ValueError(f"{path} is missing the 3xint64 header")
    rows, cols, _ = header
    return int(rows), int(cols)


def verify_shapes(sift_root: pathlib.Path, bigann_root: pathlib.Path) -> list[str]:
    errors: list[str] = []

    # Content vectors
    base = read_fvecs(sift_root / "sift_base.fvecs")
    queries = read_fvecs(sift_root / "sift_query.fvecs")
    base_rows, base_dim = read_fbin_header(bigann_root / "sift1m.base.fbin")
    query_rows, query_dim = read_fbin_header(bigann_root / "sift1m.query.public.10K.fbin")

    if base.shape != (base_rows, base_dim):
        errors.append(
            f"Base vector shape mismatch: fusedann {base.shape} vs big-ann {(base_rows, base_dim)}"
        )
    if queries.shape != (query_rows, query_dim):
        errors.append(
            f"Query vector shape mismatch: fusedann {queries.shape} vs big-ann {(query_rows, query_dim)}"
        )

    # Attributes
    base_attrs = read_bvecs(sift_root / "sift_base_attrs.bvecs")
    query_attrs = read_bvecs(sift_root / "sift_query_attrs.bvecs")
    base_rows, base_dim = read_spmat_header(bigann_root / "sift1m.base.metadata.spmat")
    query_rows, query_dim = read_spmat_header(bigann_root / "sift1m.query.metadata.public.spmat")

    if base_attrs.shape[1] != ATTR_DIM or query_attrs.shape[1] != ATTR_DIM:
        errors.append(
            f"Local attribute bitmaps have dim {base_attrs.shape[1]}/{query_attrs.shape[1]} instead of {ATTR_DIM}"
        )
    if base_attrs.shape[0] != base_rows:
        errors.append(
            f"Base attribute row mismatch: fusedann {base_attrs.shape[0]} vs big-ann {base_rows}"
        )
    if query_attrs.shape[0] != query_rows:
        errors.append(
            f"Query attribute row mismatch: fusedann {query_attrs.shape[0]} vs big-ann {query_rows}"
        )
    if base_dim != ATTR_DIM or query_dim != ATTR_DIM:
        errors.append(
            f"Big-ann metadata files expect dim {base_dim}/{query_dim} instead of {ATTR_DIM}"
        )

    # Ground truth
    base_gt_rows, base_gt_dim = read_ibin_header(bigann_root / "sift1m.GT.public.ibin")
    gt_path = sift_root / "sift_groundtruth.ivecs"
    if not gt_path.exists():
        raw_candidate = sift_root / "raw" / "sift_groundtruth.ivecs"
        if raw_candidate.exists():
            gt_path = raw_candidate
    if not gt_path.exists():
        errors.append("Missing sift_groundtruth.ivecs in data/sift1m or data/sift1m/raw")
        local_gt = np.array([], dtype=np.int32)
    else:
        local_gt = np.fromfile(gt_path, dtype=np.int32)
        if local_gt.size == 0:
            errors.append("Local sift_groundtruth.ivecs appears empty")
        else:
            local_k = local_gt[0]
            local_queries = local_gt.size // (local_k + 1)
            if local_queries != queries.shape[0]:
                errors.append(
                    f"Local ground truth rows {local_queries} do not match query count {queries.shape[0]}"
                )
            if base_gt_rows != queries.shape[0]:
                errors.append(
                    f"Big-ann GT rows {base_gt_rows} do not match query count {queries.shape[0]}"
                )
            if local_k < base_gt_dim:
                errors.append(
                    f"Local ground truth stores k={local_k}, which is < big-ann requirement {base_gt_dim}"
                )

    return errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Verify SIFT1M asset parity")
    parser.add_argument(
        "--sift-root",
        type=pathlib.Path,
        default=REPO_ROOT / "data" / "sift1m",
        help="Directory containing local fusedann SIFT1M files",
    )
    parser.add_argument(
        "--bigann-root",
        type=pathlib.Path,
        default=REPO_ROOT / "third_party" / "big-ann-benchmarks" / "data" / "sift1m-filter",
        help="Directory containing big-ann formatted SIFT1M files",
    )
    parser.add_argument(
        "--autofix",
        action="store_true",
        help="Regenerate big-ann files from local assets when mismatches are detected",
    )
    args = parser.parse_args(argv)

    errors = verify_shapes(args.sift_root, args.bigann_root)
    if errors and args.autofix:
        print("⚠️  Shape mismatch detected; regenerating big-ann dataset...")
        convert_dataset(args.sift_root, args.bigann_root)
        errors = verify_shapes(args.sift_root, args.bigann_root)

    if errors:
        print("❌ SIFT1M assets are inconsistent:")
        for issue in errors:
            print(f"  - {issue}")
        print("Run scripts/prepare_sift1m_bigann.py to regenerate the big-ann files.")
        return 1

    print("✅ SIFT1M content, attributes, and filtered GT match between fusedann and big-ann.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
