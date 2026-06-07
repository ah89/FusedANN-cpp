#!/usr/bin/env python3
"""Utility to inspect YFCC sparse metadata and explain low recall in fused ANN runs.

The script focuses on three angles:
1. Attribute coverage: how many tags per base/query vector and how often query tags
   do not exist in the sampled base.
2. Exact-match availability: for each query, how many base documents satisfy the
   strict sparse filter (doc tags must contain every query tag).
3. Cached fused-space ground-truth quality: how many valid neighbours exist per
   query according to `fused_groundtruth.ivecs`.

By default it targets the 1M-sampled dataset produced by `scripts/run_pipeline.sh`
(`data/yfcc100M_sampled_1000000`). Use `--dataset-dir` to point elsewhere.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_io import read_sparse_matrix, ivecs_read


@dataclass
class InvertedIndex:
    csc_indptr: np.ndarray
    csc_indices: np.ndarray

    @property
    def tag_freq(self) -> np.ndarray:
        return np.diff(self.csc_indptr)

    def docs_for_tag(self, tag_id: int) -> np.ndarray:
        start = self.csc_indptr[tag_id]
        end = self.csc_indptr[tag_id + 1]
        return self.csc_indices[start:end]


def describe_distribution(label: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"{label}: no data")
        return
    percentiles = np.percentile(values, [0, 10, 25, 50, 75, 90, 99, 100])
    stats = (
        f"min={percentiles[0]:.0f}, p10={percentiles[1]:.0f}, p25={percentiles[2]:.0f}, "
        f"median={percentiles[3]:.0f}, p75={percentiles[4]:.0f}, "
        f"p90={percentiles[5]:.0f}, p99={percentiles[6]:.0f}, max={percentiles[7]:.0f}"
    )
    print(f"{label}: {stats}")


def doc_match_count(tags: np.ndarray, inverted: InvertedIndex) -> int:
    if tags.size == 0:
        return inverted.csc_indptr[-1]  # nb of docs; any doc matches empty filter
    unique_tags = np.unique(tags)
    doc_lists: List[np.ndarray] = []
    for tag in unique_tags:
        start = inverted.csc_indptr[tag]
        end = inverted.csc_indptr[tag + 1]
        if start == end:
            return 0
        doc_lists.append(inverted.csc_indices[start:end])
    doc_lists.sort(key=len)
    current = doc_lists[0]
    for arr in doc_lists[1:]:
        current = np.intersect1d(current, arr, assume_unique=True)
        if current.size == 0:
            break
    return int(current.size)


def analyze_sparse_matches(
    query_csr: csr_matrix,
    inverted: InvertedIndex,
    max_queries: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    nq = query_csr.shape[0]
    limit = nq if max_queries <= 0 else min(nq, max_queries)
    match_counts = np.zeros(limit, dtype=np.int64)
    missing_tag_counts = np.zeros(limit, dtype=np.int64)
    zero_examples: List[int] = []
    freqs = inverted.tag_freq

    for qi in range(limit):
        start = query_csr.indptr[qi]
        end = query_csr.indptr[qi + 1]
        tags = query_csr.indices[start:end]
        unique_tags = np.unique(tags)
        missing = (freqs[unique_tags] == 0).sum()
        missing_tag_counts[qi] = missing
        if missing == unique_tags.size and unique_tags.size > 0:
            match_counts[qi] = 0
            if len(zero_examples) < 5:
                zero_examples.append(qi)
            continue
        count = doc_match_count(unique_tags, inverted)
        match_counts[qi] = count
        if count == 0 and len(zero_examples) < 5:
            zero_examples.append(qi)
    return match_counts, missing_tag_counts, zero_examples


def load_csr(path: pathlib.Path, label: str) -> csr_matrix:
    print(f"➡️  Loading {label} sparse matrix from {path} ...")
    csr = read_sparse_matrix(str(path), do_mmap=False)
    csr.sort_indices()
    print(
        f"   rows={csr.shape[0]:,}, cols={csr.shape[1]:,}, nnz={csr.nnz:,} (avg {csr.nnz / csr.shape[0]:.1f} tags/row)"
    )
    return csr


def build_inverted_index(base_csr: csr_matrix) -> InvertedIndex:
    print("➡️  Building column-major inverted index (CSC)...")
    base_csc = base_csr.tocsc(copy=True)
    base_csc.sort_indices()
    print("   done.")
    return InvertedIndex(base_csc.indptr, base_csc.indices)


def load_fused_groundtruth(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        print(f"⚠️  Fused groundtruth file {path} not found; skipping GT stats.")
        return np.array([])
    print(f"➡️  Reading fused groundtruth from {path} ...")
    gt = ivecs_read(str(path))
    print(f"   gt shape={gt.shape}")
    return gt


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose low recall on YFCC sparse datasets")
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/yfcc100M_sampled_1000000"),
        help="Directory containing sampled base/query files and fused_cache",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1000,
        help="Limit number of queries inspected for match counts (<=0 means all queries)",
    )
    parser.add_argument(
        "--gt-path",
        type=pathlib.Path,
        default=None,
        help="Optional override for fused groundtruth path",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir
    base_meta = dataset_dir / "base.metadata.1000000.spmat"
    query_meta = dataset_dir / "query.metadata.public.100K.spmat"
    if not base_meta.exists() or not query_meta.exists():
        print("🔥 Could not find base/query metadata files under", dataset_dir)
        return 1

    base_csr = load_csr(base_meta, "base metadata")
    query_csr = load_csr(query_meta, "query metadata")

    print("\n📊 Attribute cardinalities")
    describe_distribution("Base tags per vector", np.diff(base_csr.indptr))
    describe_distribution("Query tags per vector", np.diff(query_csr.indptr))

    inverted = build_inverted_index(base_csr)
    tag_freq = inverted.tag_freq
    print(
        f"\n📚 Attribute coverage: {np.count_nonzero(tag_freq):,} / {tag_freq.size:,} tags appear in the base"
    )
    if np.count_nonzero(tag_freq) < tag_freq.size:
        orphan = tag_freq.size - np.count_nonzero(tag_freq)
        print(f"   → {orphan:,} tags occur only in queries/private data")

    print("\n🔎 Evaluating strict sparse matches (doc must contain every query tag)...")
    match_counts, missing_tags, zero_examples = analyze_sparse_matches(
        query_csr, inverted, args.max_queries
    )
    describe_distribution("Matches per query", match_counts)
    describe_distribution("Missing query tags (per query)", missing_tags)
    zero_share = (match_counts == 0).mean() if match_counts.size else 0.0
    print(f"   Queries with zero exact matches: {zero_share * 100:.2f}% of inspected queries")

    if zero_examples:
        print("   Sample queries with zero matches:")
        for qi in zero_examples:
            start = query_csr.indptr[qi]
            end = query_csr.indptr[qi + 1]
            tags = np.unique(query_csr.indices[start:end])
            freq_snapshot = tag_freq[tags]
            print(
                f"     • q={qi}: {tags.size} tags; min base freq={freq_snapshot.min() if freq_snapshot.size else 0}, "
                f"median freq={np.median(freq_snapshot) if freq_snapshot.size else 0}"
            )

    gt_path = args.gt_path
    if gt_path is None:
        gt_path = dataset_dir / "fused_cache" / "fused_groundtruth.ivecs"
    fused_gt = load_fused_groundtruth(gt_path)
    if fused_gt.size:
        valid_counts = np.sum(fused_gt >= 0, axis=1)
        print("\n🎯 Cached fused groundtruth coverage")
        describe_distribution("Valid GT neighbors per query", valid_counts)
        zero_gt = (valid_counts == 0).mean() * 100.0
        lt5 = (valid_counts < 5).mean() * 100.0
        print(f"   Queries with 0 GT neighbors: {zero_gt:.2f}%")
        print(f"   Queries with <5 GT neighbors: {lt5:.2f}%")

    print("\nDone. Use these stats to pick fallback knobs or relax attribute constraints.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
