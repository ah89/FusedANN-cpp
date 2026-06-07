# fusedann-cpp Agent Guide

## Architecture Overview
Two binaries (`fusedann_efanna`, `fusedann_parlayann`) share core logic in `fusedann_common.h`: data I/O, α/β auto-tuning, fusion transforms, and recall. ParlayANN supports sparse/filtering + caching; EFANNA is dense-only.

**Data flow**: Raw vectors (`.fvecs`/`.u8bin`) + attributes (dense `.bvecs` or sparse `.spmat`) → hashing/PCA → fused vectors → graph build → search → recall.

**Fusion math** (`Transformer::SingleTransform`): content scaled by `1/β`, attributes repeated/padded and shifted by `−α/β`.

## Build Commands
```bash
# EFANNA (dense only)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEFANNA_ROOT=third_party/efanna
cmake --build build --target fusedann_efanna

# ParlayANN (dense + sparse)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_PARLAYANN=ON \
  -DPARLAYANN_ROOT=third_party/parlayann -DPARLAYLIB_INCLUDE_DIR=third_party/parlayann/parlaylib
cmake --build build --target fusedann_parlayann
```

**Pipeline scripts** (handle setup + execution):
- `scripts/run_sift1m_pipeline.sh [--backend efanna|parlay] [--fresh]`
- `scripts/run_bigann_filter_pipeline.sh [--fresh]` (YFCC, ParlayANN only)

## Environment Variables (Key Knobs)
| Variable | Purpose | Default |
|----------|---------|---------|
| `PARLAY_NUM_THREADS` | Parallelism | All cores |
| `FUSEDANN_PARTITION_K` | K-Means partitions (0=off) | 0 |
| `FUSEDANN_PARTITION_NPROBE` | Multi-probe count | 1 |
| `FUSEDANN_BITMAP_HASH_DIM` | Sparse→dense hash buckets | min(vocab,256) |
| `FUSEDANN_BITMAP_PCA_DIM` | PCA output dimension | 10 |
| `FUSEDANN_PCA_TARGET_DIM` | Dense PCA target | 500 |
| `FUSEDANN_PARLAY_GRAPH_DEGREE` | Vamana graph degree | 128 |
| `FUSEDANN_PARLAY_BEAM_WIDTH` | Search beam width | 64 |
| `FUSEDANN_PARLAY_RERANK_K` | Reranking pool size | 100 |
| `FUSEDANN_ALPHA_MULT/BETA_MULT` | Post-scale α/β | 1.0 |
| `FUSEDANN_PER_CLUSTER_ALPHA_BETA` | Per-partition α/β | 0 (off) |

## Caching & Cache Invalidation
ParlayANN caches to `<dataset>/fused_cache/` (or `--cache-dir`):
- `alpha_beta.txt`, `per_cluster_alpha_beta.bin` – α/β values
- `fused_groundtruth.ivecs` – ground truth reranked in fused space
- `vamana_graph.bin` + `.meta` – graph with shape/version hash
- `partition_graph_k*_c*.{bin,meta}` – per-partition graphs

**When to invalidate**: Delete `fused_cache/` after changing:
- PCA dimensions (`FUSEDANN_BITMAP_PCA_DIM`, `FUSEDANN_PCA_TARGET_DIM`)
- Hash dimensions (`FUSEDANN_BITMAP_HASH_DIM`)
- α/β heuristics (constants in `fusedann_common.h`)
- Graph params (`FUSEDANN_PARLAY_GRAPH_DEGREE`, `BUILD_PASSES`)
- Preprocessing logic in `scripts/process_sift1m.py`

## CLI Usage
```bash
# EFANNA: 5 positional args
./build/fusedann_efanna base.fvecs base_attrs.bvecs query.fvecs query_attrs.bvecs gt.ivecs

# ParlayANN: 5 args + optional cache + flags
./build/fusedann_parlayann base.u8bin base.spmat query.u8bin query.spmat gt.ibin [cache_dir] \
  [--emit-diagnostics jsonl] [--diagnostics-out path.jsonl] [--alpha-beta-cache-only]
```

## Code Conventions
- **Emoji logs** (🔍📊✅) – preserve for pipeline readability
- **Row-major contiguous** – fused buffers must stay aligned for `FloatMatrixView`/ParlayANN
- **Centralize in `fusedann_common.h`** – reuse `load_dataset`, `auto_alpha_beta`, `AttrGroupMap` helpers; avoid ad-hoc loaders
- **No automated tests** – validate via recall/QPS from pipeline runs

## Key Files Reference
| File | Purpose |
|------|---------|
| `fusedann_common.h` | Shared I/O, transforms, α/β auto-tune, constants |
| `fusedann_parlayann.cpp` | ParlayANN backend + caching + diagnostics |
| `fusedann_efanna.cpp` | EFANNA backend (dense only) |
| `scripts/process_sift1m.py` | Generate SIFT1M attribute bitmaps (10 bins) |
| `scripts/run_sift1m_pipeline.sh` | End-to-end SIFT1M automation |
| `scripts/run_bigann_filter_pipeline.sh` | YFCC filter-track automation |
