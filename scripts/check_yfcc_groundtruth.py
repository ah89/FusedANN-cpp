#!/usr/bin/env python3
"""Validate fused-space ground truth for sparse YFCC datasets.

The script checks that:
  * every ground-truth id is within [0, nb)
  * IDs are unique per query (duplicates are flagged)
  * for sparse attributes, each GT doc contains all query tags
  * each query has the requested number of valid entries

Usage example:
  python scripts/check_yfcc_groundtruth.py \
      --dataset-dir data/yfcc100M_sampled_1000000 \
      --gt fused_cache/fused_groundtruth.ivecs
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.sparse import csr_matrix

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_io import read_sparse_matrix, ivecs_read


@dataclass
class ValidationStats:
    total_queries: int
    k: int
    negatives: int = 0
    out_of_range: int = 0
    attr_mismatches: int = 0
    duplicates: int = 0

    def report(self) -> None:
        denom = max(1, self.total_queries * self.k)
        print("\n📋 Validation summary")
        print(f"   Queries checked: {self.total_queries:,}")
        print(f"   Target K: {self.k}")
        print(f"   Negative placeholders: {self.negatives} ({self.negatives / denom * 100:.2f}% of slots)")
        print(f"   Out-of-range ids: {self.out_of_range} ({self.out_of_range / denom * 100:.2f}% of slots)")
        print(f"   Attribute mismatches: {self.attr_mismatches} ({self.attr_mismatches / denom * 100:.2f}% of slots)")
        print(f"   Duplicate ids: {self.duplicates} ({self.duplicates / denom * 100:.2f}% of slots)")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate fused-space ground truth")
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/yfcc100M_sampled_1000000"),
        help="Directory containing base/query metadata and fused_cache",
    )
    parser.add_argument(
        "--gt",
        type=pathlib.Path,
        default=None,
        help="Path to fused_groundtruth.ivecs (defaults to dataset_dir/fused_cache/fused_groundtruth.ivecs)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional cap on number of queries to validate (0 = all)",
    )
    return parser.parse_args(argv)


def load_csr(path: pathlib.Path, label: str) -> csr_matrix:
    print(f"➡️  Loading {label} matrix from {path} ...")
    csr = read_sparse_matrix(str(path), do_mmap=False)
    csr.sort_indices()
    print(
        f"   rows={csr.shape[0]:,}, cols={csr.shape[1]:,}, nnz={csr.nnz:,} (avg {csr.nnz / csr.shape[0]:.2f} entries/row)"
    )
    return csr


def row_view(csr: csr_matrix, row: int) -> np.ndarray:
    start = csr.indptr[row]
    end = csr.indptr[row + 1]
    return csr.indices[start:end]


def contains_all(doc_tags: np.ndarray, query_tags: np.ndarray) -> bool:
    if query_tags.size == 0:
        return True
    if doc_tags.size < query_tags.size:
        return False
    i = j = 0
    while i < doc_tags.size and j < query_tags.size:
        if doc_tags[i] < query_tags[j]:
            i += 1
        elif doc_tags[i] > query_tags[j]:
            return False
        else:
            i += 1
            j += 1
    return j == query_tags.size


def validate_groundtruth(
    base_meta: csr_matrix,
    query_meta: csr_matrix,
    gt: np.ndarray,
    limit: Optional[int] = None,
) -> ValidationStats:
    nq, k = gt.shape
    max_queries = nq if not limit or limit <= 0 else min(nq, limit)
    stats = ValidationStats(total_queries=max_queries, k=k)
    valid_counts = np.zeros(max_queries, dtype=np.int32)

    for qi in range(max_queries):
        query_tags = np.unique(row_view(query_meta, qi))
        seen = set()
        for doc_id in gt[qi]:
            if doc_id < 0:
                stats.negatives += 1
                continue
            if doc_id >= base_meta.shape[0]:
                stats.out_of_range += 1
                continue
            if doc_id in seen:
                stats.duplicates += 1
                continue
            seen.add(int(doc_id))
            doc_tags = row_view(base_meta, int(doc_id))
            if not contains_all(doc_tags, query_tags):
                stats.attr_mismatches += 1
                continue
            valid_counts[qi] += 1

    describe_distribution("Valid GT per query", valid_counts)
    zero_share = np.mean(valid_counts == 0) * 100.0
    print(f"   Queries with zero valid GT: {zero_share:.2f}%")
    return stats


def describe_distribution(label: str, values: np.ndarray) -> None:
    percentiles = np.percentile(values, [0, 25, 50, 75, 100])
    print(
        f"\n{label}: min={percentiles[0]:.0f}, p25={percentiles[1]:.0f}, median={percentiles[2]:.0f}, "
        f"p75={percentiles[3]:.0f}, max={percentiles[4]:.0f}"
    )


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir
    base_meta = load_csr(dataset_dir / "base.metadata.1000000.spmat", "base metadata")
    query_meta = load_csr(dataset_dir / "query.metadata.public.100K.spmat", "query metadata")

    gt_path = args.gt
    if gt_path is None:
        gt_path = dataset_dir / "fused_cache" / "fused_groundtruth.ivecs"
    else:
        if not gt_path.is_absolute():
            gt_path = dataset_dir / gt_path
    if not gt_path.exists():
        print(f"🔥 Groundtruth file {gt_path} not found")
        return 1

    print(f"➡️  Loading groundtruth from {gt_path} ...")
    gt = ivecs_read(str(gt_path))
    print(f"   shape={gt.shape}")

    stats = validate_groundtruth(base_meta, query_meta, gt, args.max_queries)
    stats.report()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
