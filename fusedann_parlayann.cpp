#include "fusedann_common.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <sstream>
#include <mutex>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "algorithms/utils/beamSearch.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/graph.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"

namespace fs = std::filesystem;
namespace pann = parlayANN;
using namespace fusedann;

static constexpr uint32_t PARLAY_GRAPH_DEGREE = 128;
static constexpr uint32_t PARLAY_BEAM_WIDTH = 64;
static constexpr double PARLAY_VAMANA_ALPHA = 1.20;
static constexpr int PARLAY_VAMANA_PASSES = 2;
static constexpr int FUSEDANN_PARTITION_K = 0;
static constexpr int PARLAY_KMEANS_MAX_ITER = 50;
static constexpr double PARLAY_QUERY_CUT = 1.10;
static constexpr long PARLAY_VISIT_LIMIT = 128;
static constexpr long PARLAY_DEGREE_LIMIT = 16;
static constexpr double PARLAY_BATCH_FACTOR = 0.10;
static constexpr double PARLAY_RERANK_FACTOR = 1.0;
static constexpr size_t PARLAY_RERANK_CANDIDATES = 100;
static constexpr uint64_t GRAPH_CACHE_VERSION = 1;

enum class DiagnosticsMode {
  None,
  Jsonl,
  Tsv,
};

struct DiagnosticsState {
  DiagnosticsMode mode = DiagnosticsMode::None;
  std::ostream *stream = nullptr;
  std::unique_ptr<std::ofstream> owned_stream;
  size_t k = DEFAULT_K;
  double recall_sum = 0.0;
  size_t recall_count = 0;
};

struct QueryDiagnosticsSummary {
  size_t fused_count = 0;
  size_t expected_count = 0;
  size_t true_positive = 0;
  size_t false_positive = 0;
  size_t false_negative = 0;
  bool has_groundtruth = false;
};

struct PartitionPCA {
  size_t input_dim = 0;
  size_t output_dim = 0;
  std::vector<float> mean;
  std::vector<float> components; // row-major [output_dim][input_dim]
};

static inline void build_query_attr_raw_float_and_bitmap_bytes(const Dataset &ds,
                                                               size_t query_id,
                                                               std::vector<uint8_t> &scratch_bitmap_u8,
                                                               std::vector<float> &out_raw_float,
                                                               const uint8_t *&out_bitmap_bytes)
{
  const size_t raw_dim = ds.attr_dim;
  if (raw_dim == 0)
    throw std::runtime_error("Attribute dimension is zero; cannot build query attributes.");

  // Sparse (.spmat) path: build hashed bitmap for this single row.
  if (!ds.q_attr_indptr.empty() && !ds.q_attr_indices.empty())
  {
    scratch_bitmap_u8.assign(raw_dim, 0);
    out_raw_float.assign(raw_dim, 0.0f);

    const int64_t begin = ds.q_attr_indptr[query_id];
    const int64_t end = ds.q_attr_indptr[query_id + 1];
    if (begin > end)
      throw std::runtime_error("Query sparse indptr must be non-decreasing.");
    for (int64_t pos = begin; pos < end; ++pos)
    {
      const int32_t tag = ds.q_attr_indices[static_cast<size_t>(pos)];
      if (tag < 0 || static_cast<size_t>(tag) >= ds.attr_vocab)
        throw std::runtime_error("Query sparse tag out of range during hashed materialization.");
      const size_t bucket = static_cast<size_t>(static_cast<uint32_t>(tag)) % raw_dim;
      scratch_bitmap_u8[bucket] = 1;
      out_raw_float[bucket] = 1.0f;
    }
    out_bitmap_bytes = scratch_bitmap_u8.data();
    return;
  }

  // Dense (.bvecs) path: use raw bytes, but convert to float online.
  if (ds.q_bitmaps.empty())
    throw std::runtime_error("Query bitmap buffer is empty; cannot build dense query attributes.");
  const uint8_t *row = ds.q_bitmaps.data() + query_id * raw_dim;
  out_raw_float.resize(raw_dim);
  for (size_t j = 0; j < raw_dim; ++j)
    out_raw_float[j] = static_cast<float>(row[j]);
  out_bitmap_bytes = row;
}

static inline void project_query_attr_dense_pca(const Dataset &ds,
                                                const float *raw_in,
                                                size_t raw_dim,
                                                float *proj_out,
                                                size_t proj_dim)
{
  if (proj_dim == 0)
    return;

  if (!ds.attributes_pca)
  {
    if (raw_dim != proj_dim)
      throw std::runtime_error("PCA disabled but raw_dim != proj_dim; cannot project.");
    std::memcpy(proj_out, raw_in, sizeof(float) * proj_dim);
    return;
  }

  if (ds.attr_pca_input_dim != raw_dim || ds.attr_pca_output_dim != proj_dim ||
      ds.attr_pca_mean.size() != raw_dim ||
      ds.attr_pca_components.size() != proj_dim * raw_dim)
  {
    throw std::runtime_error("Dense PCA model missing or dimension mismatch; cannot project query attributes online.");
  }

  const float *mean = ds.attr_pca_mean.data();
  const float *components = ds.attr_pca_components.data();
  for (size_t c = 0; c < proj_dim; ++c)
  {
    const float *comp = components + c * raw_dim;
    double dot = 0.0;
    for (size_t j = 0; j < raw_dim; ++j)
    {
      const double centered = static_cast<double>(raw_in[j]) - static_cast<double>(mean[j]);
      dot += centered * static_cast<double>(comp[j]);
    }
    proj_out[c] = static_cast<float>(dot);
  }
}

static inline void project_attribute_delta(const Dataset &ds,
                                           const float *raw_delta,
                                           size_t raw_dim,
                                           float *proj_out,
                                           size_t proj_dim)
{
  if (proj_dim == 0)
    return;

  if (!ds.attributes_pca)
  {
    if (raw_dim != proj_dim)
      throw std::runtime_error("PCA disabled but raw_dim != proj_dim; cannot project attribute delta.");
    std::memcpy(proj_out, raw_delta, sizeof(float) * proj_dim);
    return;
  }

  if (ds.attr_pca_input_dim != raw_dim || ds.attr_pca_output_dim != proj_dim ||
      ds.attr_pca_components.size() != proj_dim * raw_dim)
  {
    throw std::runtime_error("Dense PCA model missing or dimension mismatch; cannot project attribute delta.");
  }

  const float *components = ds.attr_pca_components.data();
  for (size_t c = 0; c < proj_dim; ++c)
  {
    const float *comp = components + c * raw_dim;
    double dot = 0.0;
    for (size_t j = 0; j < raw_dim; ++j)
      dot += static_cast<double>(raw_delta[j]) * static_cast<double>(comp[j]);
    proj_out[c] = static_cast<float>(dot);
  }
}


static inline void fuse_single_query(const float *content_vec,
                                     const float *attr_vec,
                                     size_t dim,
                                     size_t attr_dim,
                                     float alpha,
                                     float beta,
                                     float *out_fused)
{
  Transformer::SingleTransform(const_cast<float *>(content_vec),
                               const_cast<float *>(attr_vec),
                               1,
                               dim,
                               attr_dim,
                               alpha,
                               beta,
                               out_fused);
}

DiagnosticsMode parse_diagnostics_mode(const std::string &value) {
  if (value == "none")
    return DiagnosticsMode::None;
  if (value == "jsonl")
    return DiagnosticsMode::Jsonl;
  if (value == "tsv")
    return DiagnosticsMode::Tsv;
  throw std::invalid_argument("Unknown diagnostics mode: " + value);
}

std::string format_double_full(double value) {
  if (!std::isfinite(value))
    return "null";
  std::ostringstream oss;
  oss << std::setprecision(17) << value;
  return oss.str();
}

std::string format_double_fixed5(double value, const char *nan_token = "null") {
  if (!std::isfinite(value))
    return nan_token;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(5) << value;
  return oss.str();
}

static inline std::string bytes_hex_prefix(const uint8_t *data, size_t len, size_t max_bytes = 8)
{
  if (!data || len == 0)
    return "";
  const size_t n = std::min(len, max_bytes);
  static const char *kHex = "0123456789abcdef";
  std::string out;
  out.reserve(n * 2);
  for (size_t i = 0; i < n; ++i)
  {
    const uint8_t b = data[i];
    out.push_back(kHex[(b >> 4) & 0xF]);
    out.push_back(kHex[b & 0xF]);
  }
  return out;
}

static inline double l2_distance_u8_as_float(const uint8_t *a, const uint8_t *b, size_t len)
{
  if (!a || !b || len == 0)
    return std::numeric_limits<double>::quiet_NaN();
  double sum = 0.0;
  for (size_t i = 0; i < len; ++i)
  {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += d * d;
  }
  return std::sqrt(sum);
}

static inline uint64_t fnv1a64(const uint8_t *data, size_t len);

static inline int32_t bitmap_hash_partition_id(const uint8_t *bitmap_bytes,
                                               size_t bitmap_nbytes,
                                               int partition_k)
{
  if (partition_k <= 0)
    return -1;
  if (!bitmap_bytes || bitmap_nbytes == 0)
    return -1;
  const uint64_t h = fnv1a64(bitmap_bytes, bitmap_nbytes);
  return static_cast<int32_t>(h % static_cast<uint64_t>(partition_k));
}

struct Candidate {
  double score = std::numeric_limits<double>::quiet_NaN();        // fused-space L2 distance (stage 1 ranking)
  double content_dist = std::numeric_limits<double>::quiet_NaN(); // raw content L2 distance (stage 3 ranking)
  double attr_dist = std::numeric_limits<double>::quiet_NaN();    // PCA-projected attribute L2 distance
  int32_t id = -1;
  int32_t content_partition_id = -1; // partition used during ANN search (if applicable)
  bool attr_filter_passed = false;   // true if hard attribute filter passed (stage 2)
};

struct DiagnosticEntry {
  int32_t id = -1;
  double final_score = std::numeric_limits<double>::quiet_NaN();
  double content_dist = std::numeric_limits<double>::quiet_NaN();
  double attr_dist = std::numeric_limits<double>::quiet_NaN();

  // Raw bitmap diagnostics (canonical bytes used for hashing + dist_attr_raw).
  double dist_attr_raw = std::numeric_limits<double>::quiet_NaN();
  uint64_t raw_bitmap_hash = 0;
  uint32_t bitmap_nbytes = 0;

  // Partition diagnostics.
  int32_t content_partition_id = -1; // partition probed/used by the ANN pipeline for this query/candidate
  int32_t bitmap_partition_id = -1;  // partition assignment for this base id (from bitmap-keyed assignment)

  // Sparse filter clarity.
  bool attr_filter_passed = false;
  int32_t attr_filter_missing = -1;
};

static inline void build_doc_attr_raw_bitmap_bytes(const Dataset &ds,
                                                   int32_t doc_id,
                                                   std::vector<uint8_t> &scratch_bitmap_u8,
                                                   const uint8_t *&out_bitmap_bytes)
{
  const size_t raw_dim = ds.attr_dim;
  if (raw_dim == 0)
    throw std::runtime_error("Attribute dimension is zero; cannot build doc attributes.");

  // Sparse (.spmat) path: build hashed bitmap for this single row from CSR tags.
  if (!ds.attr_indptr.empty() && !ds.attr_indices.empty())
  {
    scratch_bitmap_u8.assign(raw_dim, 0);
    const int64_t begin = ds.attr_indptr[doc_id];
    const int64_t end = ds.attr_indptr[doc_id + 1];
    if (begin > end)
      throw std::runtime_error("Base sparse indptr must be non-decreasing.");
    for (int64_t pos = begin; pos < end; ++pos)
    {
      const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
      if (tag < 0 || static_cast<size_t>(tag) >= ds.attr_vocab)
        throw std::runtime_error("Base sparse tag out of range during hashed materialization.");
      const size_t bucket = static_cast<size_t>(static_cast<uint32_t>(tag)) % raw_dim;
      scratch_bitmap_u8[bucket] = 1;
    }
    out_bitmap_bytes = scratch_bitmap_u8.data();
    return;
  }

  // Dense (.bvecs) path: use raw bytes.
  if (ds.bitmaps.empty())
    throw std::runtime_error("Base bitmap buffer is empty; cannot build dense doc attributes.");
  out_bitmap_bytes = ds.bitmaps.data() + static_cast<size_t>(doc_id) * raw_dim;
}

DiagnosticEntry compute_diagnostic_entry(const Dataset &ds,
                                         size_t query_index,
                                         int32_t doc_id,
                                         double attr_score_weight,
                                         double content_score_weight,
                                         bool use_fused_distance,
                                         const std::vector<float> *fused_base,
                                         const std::vector<float> *fused_queries) {
  DiagnosticEntry entry;
  entry.id = doc_id;
  if (doc_id < 0 || static_cast<size_t>(doc_id) >= ds.nb)
    return entry;

  const float *base_vec = ds.xb.data() + static_cast<size_t>(doc_id) * ds.dim;
  const float *query_vec = ds.xq.data() + query_index * ds.dim;
  entry.content_dist = l2_distance(base_vec, query_vec, ds.dim);

  if (ds.attributes_sparse)
  {
    const int64_t doc_begin = ds.attr_indptr[doc_id];
    const int64_t doc_end = ds.attr_indptr[doc_id + 1];
    const int64_t q_begin = ds.q_attr_indptr[query_index];
    const int64_t q_end = ds.q_attr_indptr[query_index + 1];

    size_t missing = 0;
    int64_t i = doc_begin;
    int64_t j = q_begin;
    while (i < doc_end && j < q_end)
    {
      const int32_t a = ds.attr_indices[static_cast<size_t>(i)];
      const int32_t b = ds.q_attr_indices[static_cast<size_t>(j)];
      if (a == b)
      {
        ++i;
        ++j;
      }
      else if (a < b)
      {
        ++i; // doc has an extra tag not present in the query; ignore for distance
      }
      else
      {
        ++missing; // query tag missing from the doc
        ++j;
      }
    }
    missing += static_cast<size_t>(q_end - j);
    entry.attr_dist = static_cast<double>(missing);
    entry.attr_filter_missing = static_cast<int32_t>(
      missing > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        ? std::numeric_limits<int32_t>::max()
        : missing);
    entry.attr_filter_passed = (missing == 0);
  }
  else
  {
    const size_t attr_dim = ds.attr_fused_dim ? ds.attr_fused_dim : ds.attr_dim;
    if (attr_dim > 0 && !ds.bitmaps_float.empty() && !ds.q_bitmaps_float.empty()) {
      const float *base_attr = ds.bitmaps_float.data() + static_cast<size_t>(doc_id) * attr_dim;
      const float *query_attr = ds.q_bitmaps_float.data() + query_index * attr_dim;
      entry.attr_dist = l2_distance(base_attr, query_attr, attr_dim);
    }
  }

  if (use_fused_distance && fused_base && fused_queries &&
      !fused_base->empty() && !fused_queries->empty())
  {
    const float *fused_base_vec = fused_base->data() + static_cast<size_t>(doc_id) * ds.dim;
    const float *fused_query_vec = fused_queries->data() + query_index * ds.dim;
    entry.final_score = l2_distance(fused_base_vec, fused_query_vec, ds.dim);
  }
  else
  {
    entry.final_score = attr_score_weight * entry.attr_dist + content_score_weight * entry.content_dist;
  }

  return entry;
}

// Emits per-query diagnostics in the requested schema (JSONL or TSV) and tracks recall@k.
static int32_t get_partition_id_for_doc(const std::vector<int32_t> *base_partition_id,
                                       int partition_k,
                                       int32_t doc_id)
{
  if (partition_k <= 0 || base_partition_id == nullptr)
    return -1;
  if (doc_id < 0 || static_cast<size_t>(doc_id) >= base_partition_id->size())
    return -1;
  const int32_t pid = (*base_partition_id)[static_cast<size_t>(doc_id)];
  if (pid < 0 || pid >= partition_k)
    return -1;
  return pid;
}

QueryDiagnosticsSummary emit_query_diagnostics(DiagnosticsState &state,
                                              size_t query_id,
                                              uint64_t query_raw_bitmap_hash,
                                              uint32_t query_bitmap_nbytes,
                                              int32_t query_content_partition_id,
                                              int32_t query_bitmap_partition_id,
                                              bool query_exact_bitmap_lookup_succeeded,
                                              const std::vector<DiagnosticEntry> &fused,
                                              const std::vector<DiagnosticEntry> &expected,
                                              const int32_t *expected_ids_row,
                                              const std::vector<int32_t> *candidate_order,
                                              const std::vector<int32_t> *base_partition_id,
                                              int partition_k,
                                              double alpha_real,
                                              double beta_real,
                                              double alpha_mult,
                                              double beta_mult,
                                              bool per_cluster_alpha_beta) {
  QueryDiagnosticsSummary summary;
  summary.fused_count = fused.size();
  summary.expected_count = expected.size();
  summary.has_groundtruth = (expected_ids_row != nullptr);

  std::unordered_map<int32_t, size_t> fused_rank;
  fused_rank.reserve(fused.size());
  for (size_t i = 0; i < fused.size(); ++i)
    fused_rank[fused[i].id] = i + 1;

  std::unordered_map<int32_t, size_t> candidate_rank;
  if (candidate_order && !candidate_order->empty()) {
    candidate_rank.reserve(candidate_order->size());
    for (size_t i = 0; i < candidate_order->size(); ++i)
      candidate_rank[(*candidate_order)[i]] = i + 1;
  }

  for (const auto &entry : expected) {
    if (fused_rank.find(entry.id) != fused_rank.end())
      summary.true_positive += 1;
  }
  summary.false_positive = (summary.fused_count >= summary.true_positive)
                               ? summary.fused_count - summary.true_positive
                               : 0;
  summary.false_negative = (summary.expected_count >= summary.true_positive)
                               ? summary.expected_count - summary.true_positive
                               : 0;

  if (state.mode != DiagnosticsMode::None && state.stream != nullptr) {
    std::ostream &out = *state.stream;
    const size_t k = state.k;

    switch (state.mode) {
      case DiagnosticsMode::Jsonl: {
        out << "{\"query_id\": " << query_id
            << ", \"k\": " << k
            << ", \"fused_count\": " << summary.fused_count
            << ", \"expected_count\": " << summary.expected_count
            << ", \"tp\": " << summary.true_positive
            << ", \"fp\": " << summary.false_positive
            << ", \"fn\": " << summary.false_negative
            << ", \"query_raw_bitmap_hash\": " << query_raw_bitmap_hash
            << ", \"query_bitmap_nbytes\": " << query_bitmap_nbytes
            << ", \"query_content_partition_id\": " << query_content_partition_id
            << ", \"query_bitmap_partition_id\": " << query_bitmap_partition_id
            << ", \"query_exact_bitmap_lookup_succeeded\": " << (query_exact_bitmap_lookup_succeeded ? "true" : "false")
            << ", \"per_cluster_alpha_beta\": " << (per_cluster_alpha_beta ? "true" : "false")
            << ", \"alpha_real\": " << format_double_fixed5(alpha_real, "null")
            << ", \"beta_real\": " << format_double_fixed5(beta_real, "null")
            << ", \"alpha_mult\": " << format_double_fixed5(alpha_mult, "null")
            << ", \"beta_mult\": " << format_double_fixed5(beta_mult, "null")
            << ", \"alpha_applied\": " << format_double_fixed5(alpha_real * alpha_mult, "null")
            << ", \"beta_applied\": " << format_double_fixed5(beta_real * beta_mult, "null")
            << ", \"fusedann\": [";
        for (size_t i = 0; i < fused.size(); ++i) {
          if (i > 0)
            out << ", ";
          out << "{\"id\": " << fused[i].id
              << ", \"partition_id\": " << get_partition_id_for_doc(base_partition_id, partition_k, fused[i].id)
              << ", \"content_partition_id\": " << fused[i].content_partition_id
              << ", \"bitmap_partition_id\": " << fused[i].bitmap_partition_id
              << ", \"raw_bitmap_hash\": " << fused[i].raw_bitmap_hash
              << ", \"bitmap_nbytes\": " << fused[i].bitmap_nbytes
              << ", \"dist_attr_raw\": " << format_double_fixed5(fused[i].dist_attr_raw)
              << ", \"score\": " << format_double_fixed5(fused[i].final_score)
              << ", \"dist_content\": " << format_double_fixed5(fused[i].content_dist)
              << ", \"dist_attr\": " << format_double_fixed5(fused[i].attr_dist)
              << ", \"attr_filter_passed\": " << (fused[i].attr_filter_passed ? "true" : "false")
              << ", \"attr_filter_missing\": " << fused[i].attr_filter_missing
              << ", \"rank\": " << (i + 1) << "}";
        }
        out << "], \"expected\": [";
        for (size_t i = 0; i < expected.size(); ++i) {
          if (i > 0)
            out << ", ";
          out << "{\"id\": " << expected[i].id
              << ", \"partition_id\": " << get_partition_id_for_doc(base_partition_id, partition_k, expected[i].id)
              << ", \"content_partition_id\": " << expected[i].content_partition_id
              << ", \"bitmap_partition_id\": " << expected[i].bitmap_partition_id
              << ", \"raw_bitmap_hash\": " << expected[i].raw_bitmap_hash
              << ", \"bitmap_nbytes\": " << expected[i].bitmap_nbytes
              << ", \"dist_attr_raw\": " << format_double_fixed5(expected[i].dist_attr_raw)
              << ", \"score\": " << format_double_fixed5(expected[i].final_score)
              << ", \"dist_content\": " << format_double_fixed5(expected[i].content_dist)
              << ", \"dist_attr\": " << format_double_fixed5(expected[i].attr_dist)
              << ", \"attr_filter_passed\": " << (expected[i].attr_filter_passed ? "true" : "false")
              << ", \"attr_filter_missing\": " << expected[i].attr_filter_missing
              << ", \"fusedann_rank\": ";
          auto it = fused_rank.find(expected[i].id);
          if (it != fused_rank.end())
            out << it->second;
          else {
            auto cand_it = candidate_rank.find(expected[i].id);
            if (cand_it != candidate_rank.end())
              out << cand_it->second;
            else
              out << "null";
          }
          out << "}";
        }
        out << "]}\n";
        break;
      }
      case DiagnosticsMode::Tsv: {
        out << "# query_id: " << query_id << "\n";
        out << "# k: " << k << "\n";
        out << "# fused_count: " << summary.fused_count << "\n";
        out << "# expected_count: " << summary.expected_count << "\n";
        out << "# true_positive: " << summary.true_positive << "\n";
        out << "# false_positive: " << summary.false_positive << "\n";
        out << "# false_negative: " << summary.false_negative << "\n";
        for (size_t i = 0; i < fused.size(); ++i) {
          out << "FUSEDANN\t" << (i + 1)
              << "\t" << fused[i].id
              << "\t" << format_double_fixed5(fused[i].final_score, "nan")
              << "\t" << format_double_fixed5(fused[i].content_dist, "nan")
              << "\t" << format_double_fixed5(fused[i].attr_dist, "nan")
              << "\n";
        }
        out << "\n";
        for (size_t i = 0; i < expected.size(); ++i) {
          auto it = fused_rank.find(expected[i].id);
          int fused_rank_value = -1;
          if (it != fused_rank.end()) {
            fused_rank_value = static_cast<int>(it->second);
          } else {
            auto cand_it = candidate_rank.find(expected[i].id);
            if (cand_it != candidate_rank.end())
              fused_rank_value = static_cast<int>(cand_it->second);
          }
          out << "EXPECTED\t" << (i + 1)
              << "\t" << expected[i].id
            << "\t" << format_double_fixed5(expected[i].final_score, "nan")
              << "\t" << format_double_fixed5(expected[i].content_dist, "nan")
              << "\t" << format_double_fixed5(expected[i].attr_dist, "nan")
              << "\t" << fused_rank_value << "\n";
        }
        out << "\n";
        break;
      }
      case DiagnosticsMode::None:
        break;
    }
  }

  return summary;
}

void emit_diagnostics_for_queries(DiagnosticsState &state,
                                  const Dataset &ds,
                                  const std::vector<int32_t> &ann_results,
                                  const std::vector<int32_t> &groundtruth,
                                  const std::vector<std::vector<Candidate>> &all_candidates,
                                  const std::vector<std::vector<int32_t>> *candidate_orders,
                                  double attr_score_weight,
                                  double content_score_weight,
                                  bool use_fused_distance,
                                  const std::vector<float> *fused_base,
                                  const std::vector<float> *fused_queries,
                                  const std::vector<int32_t> *base_partition_id,
                                  int partition_k,
                                  const std::vector<int32_t> *query_primary_partition_id,
                                  double alpha_real,
                                  double beta_real,
                                  double alpha_mult,
                                  double beta_mult,
                                  bool per_cluster_alpha_beta,
                                  const std::vector<bool> *query_exact_bitmap_lookup_succeeded = nullptr) {
  if (state.mode == DiagnosticsMode::None || state.stream == nullptr)
    return;

  const size_t nq = ds.nq;
  const size_t k = state.k;
  std::vector<DiagnosticEntry> fused_entries;
  std::vector<DiagnosticEntry> expected_entries;
  fused_entries.reserve(k);
  expected_entries.reserve(k);

  for (size_t qi = 0; qi < nq; ++qi) {
    fused_entries.clear();
    expected_entries.clear();

    std::vector<uint8_t> scratch_query_bitmap;
    std::vector<float> scratch_query_raw_float;
    const uint8_t *query_bitmap_bytes = nullptr;
    build_query_attr_raw_float_and_bitmap_bytes(ds,
                                                qi,
                                                scratch_query_bitmap,
                                                scratch_query_raw_float,
                                                query_bitmap_bytes);
    const size_t query_bitmap_nbytes = ds.attr_dim;
    const uint64_t query_raw_hash = fnv1a64(query_bitmap_bytes, query_bitmap_nbytes);

    const int32_t query_primary_pid =
        (query_primary_partition_id && qi < query_primary_partition_id->size())
            ? (*query_primary_partition_id)[qi]
            : -1;

    const int32_t query_bitmap_pid = bitmap_hash_partition_id(query_bitmap_bytes,
                                  query_bitmap_nbytes,
                                  partition_k);

    std::unordered_map<uint64_t, int32_t> hash_to_bitmap_pid;
    hash_to_bitmap_pid.reserve(2 * k);
    auto derive_bitmap_pid = [&](uint64_t raw_hash, int32_t fallback_pid) -> int32_t
    {
      const auto it = hash_to_bitmap_pid.find(raw_hash);
      if (it == hash_to_bitmap_pid.end())
      {
        hash_to_bitmap_pid.emplace(raw_hash, fallback_pid);
        return fallback_pid;
      }
      if (it->second != fallback_pid && fallback_pid >= 0)
      {
        std::cerr << "[invariant] query=" << qi
                  << " raw_bitmap_hash=" << raw_hash
                  << " has conflicting doc partition_id: "
                  << it->second << " vs " << fallback_pid << "\n";
      }
      return it->second;
    };

    const auto &cands_for_q = all_candidates.empty() ? std::vector<Candidate>{} : all_candidates[qi];
    for (size_t i = 0; i < k; ++i) {
      const size_t idx = qi * DEFAULT_K + i;
      if (idx >= ann_results.size())
        break;
      int32_t doc = ann_results[idx];
      if (doc < 0 || static_cast<size_t>(doc) >= ds.nb)
        continue;

      DiagnosticEntry entry;
      entry.id = doc;

      // If we stored component distances, prefer them to avoid recomputation.
      auto it = std::find_if(cands_for_q.begin(), cands_for_q.end(),
                             [doc](const Candidate &c){ return c.id == doc; });
      if (it != cands_for_q.end())
      {
        entry.final_score = it->score;
        entry.content_dist = it->content_dist;
        entry.attr_dist = it->attr_dist;
        entry.content_partition_id = it->content_partition_id;
      }
      else
      {
        entry = compute_diagnostic_entry(ds,
                                         qi,
                                         doc,
                                         attr_score_weight,
                                         content_score_weight,
                                         use_fused_distance,
                                         fused_base,
                                         fused_queries);
        entry.content_partition_id = query_primary_pid;
      }

      std::vector<uint8_t> scratch_doc_bitmap;
      const uint8_t *doc_bitmap_bytes = nullptr;
      build_doc_attr_raw_bitmap_bytes(ds, doc, scratch_doc_bitmap, doc_bitmap_bytes);
      entry.bitmap_nbytes = static_cast<uint32_t>(ds.attr_dim);
      entry.raw_bitmap_hash = fnv1a64(doc_bitmap_bytes, ds.attr_dim);
      entry.dist_attr_raw = l2_distance_u8_as_float(query_bitmap_bytes, doc_bitmap_bytes, ds.attr_dim);
      const int32_t doc_pid = get_partition_id_for_doc(base_partition_id, partition_k, doc);
      entry.bitmap_partition_id = derive_bitmap_pid(entry.raw_bitmap_hash, doc_pid);
      fused_entries.push_back(entry);
    }

    const int32_t *gt_row = groundtruth.empty()
                                ? nullptr
                                : (groundtruth.data() + qi * DEFAULT_K);
    if (gt_row) {
      for (size_t i = 0; i < k; ++i) {
        int32_t doc = gt_row[i];
        if (doc < 0 || static_cast<size_t>(doc) >= ds.nb)
          continue;

        DiagnosticEntry expected_entry = compute_diagnostic_entry(ds,
                                                                  qi,
                                                                  doc,
                                                                  attr_score_weight,
                                                                  content_score_weight,
                                                                  use_fused_distance,
                                                                  fused_base,
                                                                  fused_queries);
        expected_entry.content_partition_id = get_partition_id_for_doc(base_partition_id, partition_k, doc);
        std::vector<uint8_t> scratch_doc_bitmap;
        const uint8_t *doc_bitmap_bytes = nullptr;
        build_doc_attr_raw_bitmap_bytes(ds, doc, scratch_doc_bitmap, doc_bitmap_bytes);
        expected_entry.bitmap_nbytes = static_cast<uint32_t>(ds.attr_dim);
        expected_entry.raw_bitmap_hash = fnv1a64(doc_bitmap_bytes, ds.attr_dim);
        expected_entry.dist_attr_raw = l2_distance_u8_as_float(query_bitmap_bytes, doc_bitmap_bytes, ds.attr_dim);
        const int32_t doc_pid = get_partition_id_for_doc(base_partition_id, partition_k, doc);
        expected_entry.bitmap_partition_id = derive_bitmap_pid(expected_entry.raw_bitmap_hash, doc_pid);
        expected_entries.push_back(expected_entry);
      }
    }

    const std::vector<int32_t> *candidate_order =
      (candidate_orders && qi < candidate_orders->size())
        ? &(*candidate_orders)[qi]
        : nullptr;

    // Determine if this query had a successful exact bitmap lookup
    const bool exact_lookup_ok =
      (query_exact_bitmap_lookup_succeeded && qi < query_exact_bitmap_lookup_succeeded->size())
        ? (*query_exact_bitmap_lookup_succeeded)[qi]
        : true; // default to true if not provided (backwards compat)

    QueryDiagnosticsSummary summary =
      emit_query_diagnostics(state,
                             qi,
                             query_raw_hash,
                             static_cast<uint32_t>(query_bitmap_nbytes),
                             query_primary_pid,
                             query_bitmap_pid,
                             exact_lookup_ok,
                             fused_entries,
                             expected_entries,
                             gt_row,
                             candidate_order,
                             base_partition_id,
                             partition_k,
                             alpha_real,
                             beta_real,
                             alpha_mult,
                             beta_mult,
                             per_cluster_alpha_beta);
    if (summary.has_groundtruth) {
      state.recall_sum += static_cast<double>(summary.true_positive) / static_cast<double>(k);
      state.recall_count += 1;
    }
  }

  if (state.recall_count > 0) {
    const double avg_recall = state.recall_sum / static_cast<double>(state.recall_count);
    std::cerr << std::fixed << std::setprecision(4)
              << "[diagnostics] avg_recall_at_" << k << "=" << avg_recall
              << " over " << state.recall_count << " queries\n";
    std::cerr.unsetf(std::ios::floatfield);
  }
}

struct PartitionIndexAssets
{
  int cluster_id = -1;
  double alpha = 0.0;
  double beta = 0.0;
  size_t attr_dim_in = 0;
  size_t attr_dim_out = 0;
  std::vector<size_t> base_ids;
  std::vector<uint32_t> id_map;
  std::vector<float> attr_projected;
  std::vector<float> fused_base;
  std::vector<float> centroid_raw;
  std::vector<float> centroid_projected;
  PartitionPCA pca;
  std::unique_ptr<pann::Graph<uint32_t>> graph;
  uint32_t start_point = 0;
};

namespace
{

  long parse_env_long(const char *name, long fallback, long min_value = 1)
  {
    if (const char *raw = std::getenv(name))
    {
      char *end = nullptr;
      errno = 0;
      long parsed = std::strtol(raw, &end, 10);
      if (end != raw && errno == 0 && parsed >= min_value)
      {
        return parsed;
      }
      std::cerr << "⚠️  Ignoring invalid value for " << name << " (" << raw
                << ")\n";
    }
    return fallback;
  }

  long parse_env_long_any(const char *name, long fallback)
  {
    if (const char *raw = std::getenv(name))
    {
      char *end = nullptr;
      errno = 0;
      long parsed = std::strtol(raw, &end, 10);
      if (end != raw && errno == 0)
      {
        return parsed;
      }
      std::cerr << "⚠️  Ignoring invalid value for " << name << " (" << raw
                << ")\n";
    }
    return fallback;
  }

  double parse_env_double(const char *name, double fallback, double min_value = 0.0)
  {
    if (const char *raw = std::getenv(name))
    {
      char *end = nullptr;
      errno = 0;
      double parsed = std::strtod(raw, &end);
      if (end != raw && errno == 0 && parsed > min_value)
      {
        return parsed;
      }
      std::cerr << "⚠️  Ignoring invalid value for " << name << " (" << raw
                << ")\n";
    }
    return fallback;
  }

  bool parse_env_flag(const char *name, bool fallback)
  {
    if (const char *raw = std::getenv(name))
    {
      if (*raw == '\0')
        return fallback;
      std::string value(raw);
      for (auto &ch : value)
      {
        if (ch >= 'A' && ch <= 'Z')
          ch = static_cast<char>(ch + ('a' - 'A'));
      }
      if (value == "1" || value == "true" || value == "yes" || value == "on")
        return true;
      if (value == "0" || value == "false" || value == "no" || value == "off")
        return false;
    }
    return fallback;
  }

  struct SparseFilterIndex
  {
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> doc_ids;
  };

  SparseFilterIndex build_sparse_filter_index(const Dataset &ds)
  {
    if (!ds.attributes_sparse)
      throw std::runtime_error("build_sparse_filter_index called on dense dataset");
    SparseFilterIndex index;
    index.offsets.assign(ds.attr_vocab + 1, 0);

    for (size_t doc = 0; doc < ds.nb; ++doc)
    {
      const int64_t begin = ds.attr_indptr[doc];
      const int64_t end = ds.attr_indptr[doc + 1];
      for (int64_t pos = begin; pos < end; ++pos)
      {
        const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
        if (tag < 0 || static_cast<size_t>(tag) >= ds.attr_vocab)
          throw std::runtime_error("Sparse attribute tag out of range");
        index.offsets[static_cast<size_t>(tag) + 1] += 1;
      }
    }

    for (size_t i = 0; i < ds.attr_vocab; ++i)
    {
      index.offsets[i + 1] += index.offsets[i];
    }

    index.doc_ids.resize(index.offsets.back());
    std::vector<uint64_t> cursor = index.offsets;

    for (size_t doc = 0; doc < ds.nb; ++doc)
    {
      const int64_t begin = ds.attr_indptr[doc];
      const int64_t end = ds.attr_indptr[doc + 1];
      for (int64_t pos = begin; pos < end; ++pos)
      {
        const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
        uint64_t &write_pos = cursor[static_cast<size_t>(tag)];
        index.doc_ids[write_pos] = static_cast<uint32_t>(doc);
        ++write_pos;
      }
    }

    return index;
  }

  bool doc_matches_sparse(const Dataset &ds,
                          size_t doc_idx,
                          const int32_t *query_tags,
                          size_t query_tag_count)
  {
    static const long debug_doc_id = parse_env_long_any("FUSEDANN_DEBUG_DOC_ID", -1);

    if (query_tag_count == 0)
      return true;
    const int64_t begin = ds.attr_indptr[doc_idx];
    const int64_t end = ds.attr_indptr[doc_idx + 1];
    const size_t doc_tag_count = static_cast<size_t>(end - begin);
    if (doc_tag_count < query_tag_count)
    {
      if (static_cast<long>(doc_idx) == debug_doc_id)
      {
        std::cout << "[debug] doc_matches_sparse miss (doc=" << doc_idx
                  << ") — doc tags fewer than query tags (" << doc_tag_count
                  << " < " << query_tag_count << ")\n";
      }
      return false;
    }
    const int32_t *doc_tags = ds.attr_indices.data() + begin;
    const bool match = std::includes(doc_tags, doc_tags + doc_tag_count,
                                     query_tags, query_tags + query_tag_count);
    if (static_cast<long>(doc_idx) == debug_doc_id)
    {
      std::cout << "[debug] doc_matches_sparse(" << doc_idx << ") => "
                << (match ? "true" : "false")
                << " | doc_tag_count=" << doc_tag_count
                << " | query_tag_count=" << query_tag_count << "\n";
    }
    return match;
  }

  std::vector<uint32_t> gather_sparse_candidates(const SparseFilterIndex &index,
                                                 const int32_t *query_tags,
                                                 size_t query_tag_count)
  {
    std::vector<uint32_t> result;
    if (query_tag_count == 0)
      return result;

    std::vector<int32_t> sorted(query_tags, query_tags + query_tag_count);
    std::sort(sorted.begin(), sorted.end());

    const int32_t first_tag = sorted[0];
    if (first_tag < 0 || static_cast<size_t>(first_tag) + 1 > index.offsets.size())
      return result;
    const uint64_t first_begin = index.offsets[static_cast<size_t>(first_tag)];
    const uint64_t first_end = index.offsets[static_cast<size_t>(first_tag) + 1];
    result.assign(index.doc_ids.begin() + first_begin,
                  index.doc_ids.begin() + first_end);

    for (size_t qi = 1; qi < query_tag_count && !result.empty(); ++qi)
    {
      const int32_t tag = sorted[qi];
      if (tag < 0 || static_cast<size_t>(tag) + 1 > index.offsets.size())
      {
        result.clear();
        break;
      }
      const uint64_t begin = index.offsets[static_cast<size_t>(tag)];
      const uint64_t end = index.offsets[static_cast<size_t>(tag) + 1];
      const auto tag_begin = index.doc_ids.begin() + begin;
      const auto tag_end = index.doc_ids.begin() + end;
      std::vector<uint32_t> next;
      next.reserve(std::min(result.size(), static_cast<size_t>(end - begin)));
      std::set_intersection(result.begin(), result.end(),
                            tag_begin, tag_end,
                            std::back_inserter(next));
      result.swap(next);
    }

    return result;
  }

  static PartitionPCA compute_partition_pca(const float *data,
                                            size_t rows,
                                            size_t dim,
                                            size_t target_dim)
  {
    PartitionPCA res;
    if (rows == 0 || dim == 0)
      return res;

    target_dim = std::max<size_t>(1, std::min({target_dim, rows, dim}));
    res.input_dim = dim;
    res.output_dim = target_dim;
    res.mean.assign(dim, 0.0f);

    for (size_t r = 0; r < rows; ++r)
    {
      const float *row_ptr = data + r * dim;
      for (size_t j = 0; j < dim; ++j)
        res.mean[j] += row_ptr[j];
    }
    const double inv_rows = 1.0 / static_cast<double>(rows);
    for (size_t j = 0; j < dim; ++j)
      res.mean[j] = static_cast<float>(res.mean[j] * inv_rows);

    std::vector<double> centered(rows * dim, 0.0);
    for (size_t r = 0; r < rows; ++r)
    {
      const float *src = data + r * dim;
      double *dst = centered.data() + r * dim;
      for (size_t j = 0; j < dim; ++j)
        dst[j] = static_cast<double>(src[j]) - static_cast<double>(res.mean[j]);
    }

    std::vector<std::vector<double>> components;
    components.reserve(target_dim);
    std::vector<double> vec(dim, 0.0);
    std::vector<double> y(dim, 0.0);
    std::vector<double> projections(rows, 0.0);
    std::mt19937_64 rng(20240221);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    constexpr size_t MAX_PCA_ITER = 30;
    constexpr double CONVERGENCE = 1e-6;

    for (size_t c = 0; c < target_dim; ++c)
    {
      for (size_t j = 0; j < dim; ++j)
        vec[j] = dist(rng);
      for (const auto &prev : components)
      {
        double dot = 0.0;
        for (size_t j = 0; j < dim; ++j)
          dot += vec[j] * prev[j];
        for (size_t j = 0; j < dim; ++j)
          vec[j] -= dot * prev[j];
      }
      double norm = 0.0;
      for (double v : vec)
        norm += v * v;
      norm = std::sqrt(norm);
      if (norm < 1e-12)
        break;
      for (double &v : vec)
        v /= norm;

      for (size_t iter = 0; iter < MAX_PCA_ITER; ++iter)
      {
        std::fill(projections.begin(), projections.end(), 0.0);
        for (size_t r = 0; r < rows; ++r)
        {
          const double *src = centered.data() + r * dim;
          double dot = 0.0;
          for (size_t j = 0; j < dim; ++j)
            dot += src[j] * vec[j];
          projections[r] = dot;
        }

        std::fill(y.begin(), y.end(), 0.0);
        for (size_t r = 0; r < rows; ++r)
        {
          const double *src = centered.data() + r * dim;
          const double coeff = projections[r];
          for (size_t j = 0; j < dim; ++j)
            y[j] += src[j] * coeff;
        }

        for (const auto &prev : components)
        {
          double dot = 0.0;
          for (size_t j = 0; j < dim; ++j)
            dot += y[j] * prev[j];
          for (size_t j = 0; j < dim; ++j)
            y[j] -= dot * prev[j];
        }

        double y_norm = 0.0;
        for (double v : y)
          y_norm += v * v;
        y_norm = std::sqrt(y_norm);
        if (y_norm < 1e-12)
          break;

        double diff = 0.0;
        for (size_t j = 0; j < dim; ++j)
        {
          const double updated = y[j] / y_norm;
          diff += std::fabs(updated - vec[j]);
          vec[j] = updated;
        }
        if (diff < CONVERGENCE * dim)
          break;
      }

      components.emplace_back(vec);
    }

    res.components.assign(res.output_dim * dim, 0.0f);
    for (size_t c = 0; c < res.output_dim; ++c)
    {
      for (size_t j = 0; j < dim; ++j)
        res.components[c * dim + j] = static_cast<float>(components[c][j]);
    }

    return res;
  }

  static void project_partition_attributes(const PartitionPCA &pca,
                                           const float *src,
                                           size_t rows,
                                           std::vector<float> &dst)
  {
    if (pca.output_dim == 0 || rows == 0)
    {
      dst.clear();
      return;
    }
    dst.resize(rows * pca.output_dim, 0.0f);
    std::vector<double> centered(pca.input_dim, 0.0);
    for (size_t r = 0; r < rows; ++r)
    {
      const float *row_ptr = src + r * pca.input_dim;
      for (size_t j = 0; j < pca.input_dim; ++j)
        centered[j] = static_cast<double>(row_ptr[j]) - static_cast<double>(pca.mean[j]);
      for (size_t c = 0; c < pca.output_dim; ++c)
      {
        const float *comp = pca.components.data() + c * pca.input_dim;
        double dot = 0.0;
        for (size_t j = 0; j < pca.input_dim; ++j)
          dot += centered[j] * static_cast<double>(comp[j]);
        dst[r * pca.output_dim + c] = static_cast<float>(dot);
      }
    }
  }

  static Dataset slice_dataset(const Dataset &ds,
                               const std::vector<size_t> &indices)
  {
    Dataset sub;
    sub.nb = indices.size();
    sub.dim = ds.dim;
    sub.attr_dim = ds.attr_dim;
    sub.attr_fused_dim = ds.attr_fused_dim;
    sub.attributes_sparse = ds.attributes_sparse;
    sub.attr_vocab = ds.attr_vocab;

    sub.xb.resize(sub.nb * sub.dim);
    for (size_t i = 0; i < sub.nb; ++i)
    {
      const size_t idx = indices[i];
      std::memcpy(sub.xb.data() + i * sub.dim,
                  ds.xb.data() + idx * sub.dim,
                  sizeof(float) * sub.dim);
    }

    if (!ds.bitmaps.empty())
    {
      sub.bitmaps.resize(sub.nb * sub.attr_dim);
      for (size_t i = 0; i < sub.nb; ++i)
      {
        const size_t idx = indices[i];
        std::memcpy(sub.bitmaps.data() + i * sub.attr_dim,
                    ds.bitmaps.data() + idx * sub.attr_dim,
                    sizeof(uint8_t) * sub.attr_dim);
      }
    }

    const size_t float_attr_dim = sub.attr_fused_dim ? sub.attr_fused_dim : sub.attr_dim;
    if (!ds.bitmaps_float.empty() && float_attr_dim > 0)
    {
      sub.bitmaps_float.resize(sub.nb * float_attr_dim);
      for (size_t i = 0; i < sub.nb; ++i)
      {
        const size_t idx = indices[i];
        std::memcpy(sub.bitmaps_float.data() + i * float_attr_dim,
                    ds.bitmaps_float.data() + idx * float_attr_dim,
                    sizeof(float) * float_attr_dim);
      }
    }

    if (sub.attributes_sparse)
    {
      sub.attr_indptr.assign(sub.nb + 1, 0);
      sub.attr_indices.clear();
      size_t cursor = 0;
      if (ds.attr_indptr.size() != ds.nb + 1)
        throw std::runtime_error("Sparse attribute indptr shape mismatch (slice).");
      for (size_t i = 0; i < sub.nb; ++i)
      {
        const size_t idx = indices[i];
        const int64_t begin = ds.attr_indptr[idx];
        const int64_t end = ds.attr_indptr[idx + 1];
        if (begin > end)
          throw std::runtime_error("Sparse attribute indptr must be non-decreasing (slice).");
        const size_t begin_idx = static_cast<size_t>(begin);
        const size_t end_idx = static_cast<size_t>(end);
        const size_t len = end_idx - begin_idx;
        sub.attr_indices.insert(sub.attr_indices.end(),
                                ds.attr_indices.begin() + begin_idx,
                                ds.attr_indices.begin() + end_idx);
        cursor += len;
        sub.attr_indptr[i + 1] = cursor;
      }
    }

    return sub;
  }

  constexpr size_t DEFAULT_FILTER_BRUTE_FORCE_LIMIT = 50000;

  // Storage/compute type for fused vectors in the ParlayANN point. Default uint8
  // (1 byte/dim, SIMD-friendly, ~4x faster distance than float) — the fused
  // vectors are scalar-quantized into it, exactly as ParlayIVF quantizes its base.
  // Override with -DFUSEDANN_POINT_T=float to disable quantization.
#ifndef FUSEDANN_POINT_T
#define FUSEDANN_POINT_T uint8_t
#endif
  using ParlayPoint = pann::Euclidian_Point<FUSEDANN_POINT_T>;
  using PointRange = pann::PointRange<ParlayPoint>;

  std::vector<int32_t> compute_sparse_filtered_groundtruth(
      const Dataset &ds,
      const SparseFilterIndex &index,
      const std::vector<float> &fused_base,
      const std::vector<float> &fused_queries,
      size_t dim,
      size_t k)
  {
    std::vector<int32_t> gt(ds.nq * k, -1);
    std::vector<std::pair<float, int32_t>> scored;

    for (size_t qi = 0; qi < ds.nq; ++qi)
    {
      scored.clear();
      const int64_t tag_begin = ds.q_attr_indptr.empty() ? 0 : ds.q_attr_indptr[qi];
      const int64_t tag_end = ds.q_attr_indptr.empty() ? 0 : ds.q_attr_indptr[qi + 1];
      const size_t tag_count = static_cast<size_t>(tag_end - tag_begin);
      const int32_t *query_tags = ds.q_attr_indices.data() + tag_begin;

      const auto matches = gather_sparse_candidates(index, query_tags, tag_count);
      const float *query_vec = fused_queries.data() + qi * dim;
      scored.reserve(matches.size());

      for (uint32_t doc : matches)
      {
        const float *base_vec = fused_base.data() + static_cast<size_t>(doc) * dim;
        const float dist = l2_distance_sq(query_vec, base_vec, dim);
        scored.emplace_back(dist, static_cast<int32_t>(doc));
      }

      if (scored.size() > k)
      {
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](const auto &lhs, const auto &rhs)
                          {
                            return lhs.first < rhs.first;
                          });
        scored.resize(k);
      }
      else
      {
        std::sort(scored.begin(), scored.end(),
                  [](const auto &lhs, const auto &rhs)
                  {
                    return lhs.first < rhs.first;
                  });
      }

      for (size_t kk = 0; kk < k; ++kk)
      {
        gt[qi * k + kk] = (kk < scored.size()) ? scored[kk].second : -1;
      }
    }

    return gt;
  }

  // Content-space filtered ground truth: for each query, the true top-k base
  // docs by *content* L2 among docs that satisfy the (subset) filter. This is
  // the honest filter-track metric, independent of the fusion / alpha-beta.
  std::vector<int32_t> compute_content_filtered_groundtruth(
      const Dataset &ds,
      const SparseFilterIndex &index,
      size_t k)
  {
    std::vector<int32_t> gt(ds.nq * k, -1);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      std::vector<std::pair<float, int32_t>> scored;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
      for (size_t qi = 0; qi < ds.nq; ++qi)
      {
        scored.clear();
        const int64_t tag_begin = ds.q_attr_indptr.empty() ? 0 : ds.q_attr_indptr[qi];
        const int64_t tag_end = ds.q_attr_indptr.empty() ? 0 : ds.q_attr_indptr[qi + 1];
        const size_t tag_count = static_cast<size_t>(tag_end - tag_begin);
        const int32_t *query_tags = ds.q_attr_indices.data() + tag_begin;
        const auto matches = gather_sparse_candidates(index, query_tags, tag_count);
        const float *query_vec = ds.xq.data() + qi * ds.dim;
        scored.reserve(matches.size());
        for (uint32_t doc : matches)
        {
          const float *base_vec = ds.xb.data() + static_cast<size_t>(doc) * ds.dim;
          scored.emplace_back(static_cast<float>(l2_distance_sq(query_vec, base_vec, ds.dim)),
                              static_cast<int32_t>(doc));
        }
        if (scored.size() > k)
        {
          std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                            [](const auto &a, const auto &b) { return a.first < b.first; });
          scored.resize(k);
        }
        else
        {
          std::sort(scored.begin(), scored.end(),
                    [](const auto &a, const auto &b) { return a.first < b.first; });
        }
        for (size_t kk = 0; kk < k; ++kk)
          gt[qi * k + kk] = (kk < scored.size()) ? scored[kk].second : -1;
      }
    }
    return gt;
  }

} // namespace

struct FloatMatrixView
{
  const float *data = nullptr;
  size_t rows = 0;
  size_t dim = 0;

  struct Row
  {
    const float *ptr;
    float operator[](size_t j) const { return ptr[j]; }
  };

  size_t size() const { return rows; }
  long dimension() const { return static_cast<long>(dim); }
  Row operator[](size_t i) const { return Row{data + i * dim}; }
};

struct CachePaths
{
  fs::path dir;
  fs::path alpha_beta;
  fs::path per_cluster_alpha_beta;
  fs::path fused_gt;
  fs::path graph_bin;
  fs::path graph_meta;
};

static fs::path partition_assignments_path(const CachePaths &paths, int partition_k)
{
  return paths.dir / ("partition_assignments_k" + std::to_string(partition_k) + ".bin");
}

static inline void fnv1a64_update(uint64_t &h, const uint8_t *data, size_t len)
{
  for (size_t i = 0; i < len; ++i)
  {
    h ^= static_cast<uint64_t>(data[i]);
    h *= 1099511628211ULL;
  }
}

static CachePaths resolve_cache_paths(const std::string &base_path,
                                      const char *override_dir)
{
  CachePaths paths;
  if (override_dir && *override_dir)
  {
    paths.dir = fs::path(override_dir);
  }
  else
  {
    fs::path base = fs::path(base_path).lexically_normal();
    fs::path parent = base.has_parent_path() ? base.parent_path() : fs::current_path();
    paths.dir = parent / "fused_cache";
  }
  std::error_code ec;
  fs::create_directories(paths.dir, ec);
  if (ec)
  {
    throw std::runtime_error("Unable to create cache directory: " + paths.dir.string());
  }
  paths.alpha_beta = paths.dir / "alpha_beta.txt";
  paths.per_cluster_alpha_beta = paths.dir / "per_cluster_alpha_beta.bin";
  paths.fused_gt = paths.dir / "fused_groundtruth.ivecs";
  paths.graph_bin = paths.dir / "vamana_graph.bin";
  paths.graph_meta = paths.dir / "vamana_graph.meta";
  return paths;
}

static inline uint64_t fnv1a64(const uint8_t *data, size_t len)
{
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < len; ++i)
  {
    h ^= static_cast<uint64_t>(data[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

struct PartitionAssignmentsHeader
{
  uint32_t magic = 0x44544950; // 'PITD'
  uint32_t version = 1;
  uint32_t partition_k = 0;
  uint32_t reserved = 0;
  uint64_t nb = 0;
  uint64_t base_hash = 0;
  uint64_t attr_hash = 0;
  uint64_t checksum = 0;
};

static uint64_t compute_partition_assignments_checksum(const PartitionAssignmentsHeader &hdr,
                                                       const std::vector<int32_t> &assignments)
{
  uint64_t h = 1469598103934665603ULL;
  fnv1a64_update(h, reinterpret_cast<const uint8_t *>(&hdr.partition_k), sizeof(hdr.partition_k));
  fnv1a64_update(h, reinterpret_cast<const uint8_t *>(&hdr.nb), sizeof(hdr.nb));
  fnv1a64_update(h, reinterpret_cast<const uint8_t *>(&hdr.base_hash), sizeof(hdr.base_hash));
  fnv1a64_update(h, reinterpret_cast<const uint8_t *>(&hdr.attr_hash), sizeof(hdr.attr_hash));
  if (!assignments.empty())
  {
    fnv1a64_update(h,
                   reinterpret_cast<const uint8_t *>(assignments.data()),
                   assignments.size() * sizeof(int32_t));
  }
  return h;
}

static bool load_partition_assignments(const fs::path &path,
                                       int partition_k,
                                       size_t nb,
                                       uint64_t base_hash,
                                       uint64_t attr_hash,
                                       std::vector<int32_t> &out_assignments)
{
  out_assignments.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;

  PartitionAssignmentsHeader hdr;
  in.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
  if (!in)
    return false;

  if (hdr.magic != 0x44544950 || hdr.version != 1)
    return false;
  if (hdr.partition_k != static_cast<uint32_t>(partition_k))
    return false;
  if (hdr.nb != static_cast<uint64_t>(nb))
    return false;
  if (hdr.base_hash != base_hash || hdr.attr_hash != attr_hash)
    return false;

  out_assignments.resize(nb);
  in.read(reinterpret_cast<char *>(out_assignments.data()), nb * sizeof(int32_t));
  if (!in)
  {
    out_assignments.clear();
    return false;
  }

  PartitionAssignmentsHeader chk = hdr;
  chk.checksum = 0;
  const uint64_t expected = hdr.checksum;
  const uint64_t got = compute_partition_assignments_checksum(chk, out_assignments);
  if (expected != got)
  {
    out_assignments.clear();
    return false;
  }
  return true;
}

static void write_partition_assignments(const fs::path &path,
                                        int partition_k,
                                        size_t nb,
                                        uint64_t base_hash,
                                        uint64_t attr_hash,
                                        const std::vector<int32_t> &assignments)
{
  if (assignments.size() != nb)
    throw std::runtime_error("Partition assignments size mismatch.");

  PartitionAssignmentsHeader hdr;
  hdr.partition_k = static_cast<uint32_t>(partition_k);
  hdr.nb = static_cast<uint64_t>(nb);
  hdr.base_hash = base_hash;
  hdr.attr_hash = attr_hash;
  hdr.checksum = 0;
  hdr.checksum = compute_partition_assignments_checksum(hdr, assignments);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    throw std::runtime_error("Unable to write partition assignments cache: " + path.string());

  out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
  out.write(reinterpret_cast<const char *>(assignments.data()), assignments.size() * sizeof(int32_t));
  if (!out)
    throw std::runtime_error("Failed writing partition assignments cache: " + path.string());
}

struct PerClusterAlphaBetaHeader
{
  uint32_t magic = 0x41424350; // 'PCBA'
  uint32_t version = 3;
  uint32_t partition_k = 0;
  uint32_t bitmap_hash_dim = 0;
  uint32_t bitmap_pca_dim = 0;
  uint32_t attr_dim = 0;
  uint32_t reserved = 0;
  uint64_t checksum = 0;
};

static bool load_per_cluster_alpha_beta(const fs::path &path,
                                        int partition_k,
                                        size_t bitmap_hash_dim,
                                        size_t bitmap_pca_dim,
                                        size_t attr_dim,
                                        std::vector<double> &cluster_alpha,
                                        std::vector<double> &cluster_beta)
{
  cluster_alpha.clear();
  cluster_beta.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;

  PerClusterAlphaBetaHeader hdr;
  if (!in.read(reinterpret_cast<char *>(&hdr), sizeof(hdr)))
    return false;
  if (hdr.magic != 0x41424350 || hdr.version != 3)
    return false;
  if (hdr.partition_k != static_cast<uint32_t>(partition_k))
    return false;
  if (hdr.bitmap_hash_dim != static_cast<uint32_t>(bitmap_hash_dim))
    return false;
  if (hdr.bitmap_pca_dim != static_cast<uint32_t>(bitmap_pca_dim))
    return false;
  if (hdr.attr_dim != static_cast<uint32_t>(attr_dim))
    return false;

  const size_t k = static_cast<size_t>(partition_k);
  cluster_alpha.resize(k);
  cluster_beta.resize(k);
  const size_t bytes = sizeof(double) * k * 2;
  std::vector<uint8_t> payload(bytes);
  if (!in.read(reinterpret_cast<char *>(payload.data()), static_cast<std::streamsize>(bytes)))
    return false;
  if (fnv1a64(payload.data(), payload.size()) != hdr.checksum)
    return false;

  std::memcpy(cluster_alpha.data(), payload.data(), sizeof(double) * k);
  std::memcpy(cluster_beta.data(), payload.data() + sizeof(double) * k, sizeof(double) * k);
  return true;
}

static void write_per_cluster_alpha_beta(const fs::path &path,
                                         int partition_k,
                                         size_t bitmap_hash_dim,
                                         size_t bitmap_pca_dim,
                                         size_t attr_dim,
                                         const std::vector<double> &cluster_alpha,
                                         const std::vector<double> &cluster_beta)
{
  const size_t k = static_cast<size_t>(partition_k);
  if (cluster_alpha.size() != k || cluster_beta.size() != k)
    throw std::runtime_error("Per-cluster alpha/beta arrays have wrong size.");

  const size_t bytes = sizeof(double) * k * 2;
  std::vector<uint8_t> payload(bytes);
  std::memcpy(payload.data(), cluster_alpha.data(), sizeof(double) * k);
  std::memcpy(payload.data() + sizeof(double) * k, cluster_beta.data(), sizeof(double) * k);

  PerClusterAlphaBetaHeader hdr;
  hdr.partition_k = static_cast<uint32_t>(partition_k);
  hdr.bitmap_hash_dim = static_cast<uint32_t>(bitmap_hash_dim);
  hdr.bitmap_pca_dim = static_cast<uint32_t>(bitmap_pca_dim);
  hdr.attr_dim = static_cast<uint32_t>(attr_dim);
  hdr.checksum = fnv1a64(payload.data(), payload.size());

  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Failed to open per-cluster alpha/beta cache for writing: " + path.string());
  out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
  out.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

struct PartitionGraphCacheMeta
{
  uint32_t magic = 0x47435050; // 'PPCG'
  uint32_t version = 4;
  int32_t cluster_id = -1;
  uint32_t reserved = 0;
  size_t nb_group = 0;
  size_t dim = 0;
  size_t attr_dim_out = 0;
  uint32_t degree = 0;
  int passes = 0;
  uint64_t base_hash = 0;
  uint64_t attr_hash = 0;
  double alpha_real = 0.0;
  double beta_real = 0.0;
  double alpha_mult = 1.0;
  double beta_mult = 1.0;
  uint32_t start_id = 0;
};

static bool read_partition_graph_meta(const fs::path &path, PartitionGraphCacheMeta &meta)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  return in.read(reinterpret_cast<char *>(&meta), sizeof(meta)).good();
}

static void write_partition_graph_meta(const fs::path &path, const PartitionGraphCacheMeta &meta)
{
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Failed to open partition graph meta for writing: " + path.string());
  out.write(reinterpret_cast<const char *>(&meta), sizeof(meta));
}

static bool partition_graph_meta_matches(const PartitionGraphCacheMeta &a, const PartitionGraphCacheMeta &b)
{
  return a.magic == b.magic && a.version == b.version &&
         a.cluster_id == b.cluster_id &&
         a.nb_group == b.nb_group &&
         a.dim == b.dim &&
         a.attr_dim_out == b.attr_dim_out &&
         a.degree == b.degree && a.passes == b.passes &&
         a.base_hash == b.base_hash && a.attr_hash == b.attr_hash &&
         a.alpha_real == b.alpha_real && a.beta_real == b.beta_real &&
         a.alpha_mult == b.alpha_mult && a.beta_mult == b.beta_mult;
}

static void invalidate_cache_file(const fs::path &path)
{
  std::error_code ec;
  fs::remove(path, ec);
  if (ec && ec.value() != static_cast<int>(std::errc::no_such_file_or_directory))
  {
    std::cout << "   (failed to delete " << path << ": " << ec.message() << ")\n";
  }
}

static bool load_alpha_beta(const fs::path &path, double &alpha, double &beta)
{
  std::ifstream in(path);
  if (!in)
  {
    std::cout << "📭 α/β cache missing at " << path << "\n";
    return false;
  }
  double a = 0.0;
  double b = 0.0;
  if (in >> a >> b)
  {
    alpha = a;
    beta = b;
    return true;
  }
  return false;
}

static void write_alpha_beta(const fs::path &path, double alpha, double beta)
{
  std::ofstream out(path);
  if (out)
  {
    out << alpha << " " << beta << "\n";
  }
}

struct BitmapHash
{
  uint64_t lo = 0;
  uint64_t hi = 0;

  bool operator==(const BitmapHash &other) const noexcept
  {
    return lo == other.lo && hi == other.hi;
  }
};

struct BitmapHashHasher
{
  size_t operator()(const BitmapHash &h) const noexcept
  {
    return static_cast<size_t>(h.lo ^ (h.hi >> 1));
  }
};

static inline uint64_t rotl64(uint64_t x, uint8_t r)
{
  return (x << r) | (x >> (64U - r));
}

static inline uint64_t fmix64(uint64_t k)
{
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

// MurmurHash3_x64_128 (public domain reference implementation)
static BitmapHash hash_bitmap_bytes(const uint8_t *data, size_t len)
{
  if (!data || len == 0)
    return {};

  const uint64_t c1 = 0x87c37b91114253d5ULL;
  const uint64_t c2 = 0x4cf5ad432745937fULL;
  uint64_t h1 = 0x9368e53c2f6af274ULL;
  uint64_t h2 = 0x586dcd208f7cd3fdULL;

  const size_t nblocks = len / 16;
  for (size_t i = 0; i < nblocks; ++i)
  {
    uint64_t k1 = 0;
    uint64_t k2 = 0;
    std::memcpy(&k1, data + i * 16 + 0, sizeof(uint64_t));
    std::memcpy(&k2, data + i * 16 + 8, sizeof(uint64_t));

    k1 *= c1;
    k1 = rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;

    h1 = rotl64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

    h2 = rotl64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  const uint8_t *tail = data + nblocks * 16;
  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (len & 15)
  {
  case 15:
    k2 ^= static_cast<uint64_t>(tail[14]) << 48;
    [[fallthrough]];
  case 14:
    k2 ^= static_cast<uint64_t>(tail[13]) << 40;
    [[fallthrough]];
  case 13:
    k2 ^= static_cast<uint64_t>(tail[12]) << 32;
    [[fallthrough]];
  case 12:
    k2 ^= static_cast<uint64_t>(tail[11]) << 24;
    [[fallthrough]];
  case 11:
    k2 ^= static_cast<uint64_t>(tail[10]) << 16;
    [[fallthrough]];
  case 10:
    k2 ^= static_cast<uint64_t>(tail[9]) << 8;
    [[fallthrough]];
  case 9:
    k2 ^= static_cast<uint64_t>(tail[8]) << 0;
    k2 *= c2;
    k2 = rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;
    [[fallthrough]];
  case 8:
    k1 ^= static_cast<uint64_t>(tail[7]) << 56;
    [[fallthrough]];
  case 7:
    k1 ^= static_cast<uint64_t>(tail[6]) << 48;
    [[fallthrough]];
  case 6:
    k1 ^= static_cast<uint64_t>(tail[5]) << 40;
    [[fallthrough]];
  case 5:
    k1 ^= static_cast<uint64_t>(tail[4]) << 32;
    [[fallthrough]];
  case 4:
    k1 ^= static_cast<uint64_t>(tail[3]) << 24;
    [[fallthrough]];
  case 3:
    k1 ^= static_cast<uint64_t>(tail[2]) << 16;
    [[fallthrough]];
  case 2:
    k1 ^= static_cast<uint64_t>(tail[1]) << 8;
    [[fallthrough]];
  case 1:
    k1 ^= static_cast<uint64_t>(tail[0]) << 0;
    k1 *= c1;
    k1 = rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;
    break;
  default:
    break;
  }

  h1 ^= len;
  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  return BitmapHash{h1, h2};
}

// Helper for hashing paths
static uint64_t hash_path(const std::string &p)
{
  std::hash<std::string> h;
  return static_cast<uint64_t>(h(p));
}

// Graph metadata for caching
struct GraphCacheMeta
{
  uint32_t magic = 0x4743414D; // 'GCAM'
  uint32_t version = 2;
  size_t nb = 0;
  size_t dim = 0;
  size_t attr_dim = 0;
  uint32_t degree = 0;
  int passes = 0;
  uint64_t base_hash = 0;
  uint64_t attr_hash = 0;
  double alpha_real = 0.0;
  double beta_real = 0.0;
  double alpha_mult = 1.0;
  double beta_mult = 1.0;
  uint32_t start_id = 0;
};

static bool read_graph_metadata(const fs::path &path, GraphCacheMeta &meta)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  return in.read(reinterpret_cast<char *>(&meta), sizeof(meta)).good();
}

static void write_graph_metadata(const fs::path &path, const GraphCacheMeta &meta)
{
  std::ofstream out(path, std::ios::binary);
  if (out)
  {
    out.write(reinterpret_cast<const char *>(&meta), sizeof(meta));
  }
}

static bool graph_metadata_matches(const GraphCacheMeta &a, const GraphCacheMeta &b)
{
  // α/β are persisted to text (alpha_beta.txt) at limited precision, so the
  // reloaded value can differ from the graph meta's stored value by a few ULPs.
  // Compare them with a small relative tolerance to avoid needless rebuilds.
  auto close = [](double x, double y)
  { return std::fabs(x - y) <= 1e-3 * (1.0 + std::fabs(x) + std::fabs(y)); };
  return a.magic == b.magic && a.version == b.version &&
         a.nb == b.nb && a.dim == b.dim && a.attr_dim == b.attr_dim &&
         a.degree == b.degree && a.passes == b.passes &&
         a.base_hash == b.base_hash && a.attr_hash == b.attr_hash &&
         close(a.alpha_real, b.alpha_real) && close(a.beta_real, b.beta_real) &&
         close(a.alpha_mult, b.alpha_mult) && close(a.beta_mult, b.beta_mult);
}

static bool load_fused_groundtruth(const fs::path &path, size_t nq, size_t k, size_t nb,
                                   std::vector<int32_t> &out)
{
  if (!fs::exists(path))
    return false;
  size_t rows = 0, dim = 0;
  try
  {
    load_ivecs(path.string(), out, rows, dim);
    if (rows != nq || dim != k)
    {
      std::cout << "⚠️  Cached ground truth shape mismatch (" << rows << "x" << dim
                << " vs " << nq << "x" << k << ")\n";
      return false;
    }
    // Validate indices
    for (int32_t idx : out)
    {
      if (idx < -1 && idx != -1)
        return false; // -1 is allowed for missing
      if (idx >= 0 && static_cast<size_t>(idx) >= nb)
        return false;
    }
    return true;
  }
  catch (...)
  {
    return false;
  }
}

static void write_fused_groundtruth(const fs::path &path, const std::vector<int32_t> &data,
                                    size_t nq, size_t k)
{
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return;
  for (size_t i = 0; i < nq; ++i)
  {
    int32_t d = static_cast<int32_t>(k);
    out.write(reinterpret_cast<const char *>(&d), sizeof(d));
    out.write(reinterpret_cast<const char *>(data.data() + i * k), k * sizeof(int32_t));
  }
}

int main(int argc, char **argv)
{
  std::vector<std::string> positional;
  std::string diagnostics_mode_arg = "none";
  std::string diagnostics_out_path;
  bool alpha_beta_cache_only = false;

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--emit-diagnostics")
    {
      if (i + 1 >= argc)
      {
        std::cerr << "--emit-diagnostics requires a value (none|jsonl|tsv).\n";
        return EXIT_FAILURE;
      }
      diagnostics_mode_arg = argv[++i];
    }
    else if (arg == "--diagnostics-out")
    {
      if (i + 1 >= argc)
      {
        std::cerr << "--diagnostics-out requires a path argument.\n";
        return EXIT_FAILURE;
      }
      diagnostics_out_path = argv[++i];
    }
    else if (arg == "--alpha-beta-cache-only")
    {
      alpha_beta_cache_only = true;
    }
    else if (arg.rfind("--", 0) == 0)
    {
      std::cerr << "Unknown flag: " << arg << "\n";
      return EXIT_FAILURE;
    }
    else
    {
      positional.push_back(arg);
    }
  }

  if (positional.size() < 5 || positional.size() > 6)
  {
    std::cerr << "Usage: " << argv[0]
              << " <sift_base.fvecs> <sift_base_attrs.bvecs>"
                 " <sift_query.fvecs> <sift_query_attrs.bvecs>"
                 " <sift_groundtruth.ivecs> [cache_dir]"
                 " [--emit-diagnostics none|jsonl|tsv] [--diagnostics-out PATH]"
                 " [--alpha-beta-cache-only]\n";
    return EXIT_FAILURE;
  }

  const std::string base_path = positional[0];
  const std::string base_attr_path = positional[1];
  const std::string query_path = positional[2];
  const std::string query_attr_path = positional[3];
  const std::string gt_path = positional[4];
  std::string cache_override_arg = (positional.size() == 6) ? positional[5] : std::string();
  const char *cache_override = cache_override_arg.empty() ? nullptr : cache_override_arg.c_str();

  DiagnosticsState diagnostics;
  try
  {
    diagnostics.mode = parse_diagnostics_mode(diagnostics_mode_arg);
  }
  catch (const std::exception &ex)
  {
    std::cerr << ex.what() << "\n";
    return EXIT_FAILURE;
  }
  if (diagnostics.mode != DiagnosticsMode::None)
  {
    if (!diagnostics_out_path.empty())
    {
      diagnostics.owned_stream = std::make_unique<std::ofstream>(diagnostics_out_path, std::ios::app);
      if (!diagnostics.owned_stream->is_open())
      {
        std::cerr << "Failed to open diagnostics output path: " << diagnostics_out_path << "\n";
        return EXIT_FAILURE;
      }
      diagnostics.stream = diagnostics.owned_stream.get();
    }
    else
    {
      diagnostics.stream = &std::cout;
    }
  }

  try
  {
    int stage_counter = 1;
    auto log_stage = [&stage_counter](const std::string &desc)
    {
      std::cout << "⏱️  Stage " << stage_counter++
                << ": " << desc << "\n";
      std::cout.flush();
    };

    log_stage("Loading dataset and attributes...");
    Dataset ds = load_dataset(base_path, base_attr_path,
                              query_path, query_attr_path,
                              gt_path);

    const bool diagnostics_enabled = (diagnostics.mode != DiagnosticsMode::None);
    std::vector<std::vector<int32_t>> diagnostics_candidate_orders;
    if (diagnostics_enabled)
      diagnostics_candidate_orders.assign(ds.nq, {});
    std::vector<std::vector<Candidate>> all_query_candidates(ds.nq);

    CachePaths cache = resolve_cache_paths(base_path, cache_override);

    // User-requested cache-only mode is treated as best-effort: if the cache is
    // missing/mismatched we warn and fall back to auto estimation.
    bool alpha_beta_cache_only_effective = alpha_beta_cache_only;

    auto load_alpha_beta_or_throw = [&](double &alpha, double &beta) -> bool
    {
      if (load_alpha_beta(cache.alpha_beta, alpha, beta))
      {
        std::cout << "📦 Loaded α/β from " << cache.alpha_beta << "\n";
        return true;
      }
      if (alpha_beta_cache_only_effective)
      {
        std::cout << "⚠️  --alpha-beta-cache-only requested, but α/β cache missing at "
                  << cache.alpha_beta << ". Falling back to auto estimation.\n";
        alpha_beta_cache_only_effective = false;
      }
      return false;
    };

    if (alpha_beta_cache_only)
    {
      std::cout << "📎 α/β cache-only requested; will fall back to auto estimation if cache is missing.\n";
    }

    long beam_width = parse_env_long("FUSEDANN_PARLAY_BEAM_WIDTH",
                                     static_cast<long>(PARLAY_BEAM_WIDTH));

    double alpha_mult = parse_env_double("FUSEDANN_ALPHA_MULT", 1.0, 0.0);
    double beta_mult = parse_env_double("FUSEDANN_BETA_MULT", 1.0, 0.0);
    if (!std::isfinite(alpha_mult) || alpha_mult <= 0.0)
    {
      std::cerr << "⚠️  Ignoring non-finite/invalid FUSEDANN_ALPHA_MULT; using 1.0\n";
      alpha_mult = 1.0;
    }
    if (!std::isfinite(beta_mult) || beta_mult <= 0.0)
    {
      std::cerr << "⚠️  Ignoring non-finite/invalid FUSEDANN_BETA_MULT; using 1.0\n";
      beta_mult = 1.0;
    }
    if (alpha_mult != 1.0 || beta_mult != 1.0)
    {
      std::cout << std::fixed << std::setprecision(4)
                << "🧪 Applying α/β multipliers: α×" << alpha_mult
                << ", β×" << beta_mult << std::defaultfloat << "\n";
    }
    const long graph_degree_env = parse_env_long(
        "FUSEDANN_PARLAY_GRAPH_DEGREE",
        static_cast<long>(PARLAY_GRAPH_DEGREE));
    const long max_degree_allowed = static_cast<long>(
        std::max<size_t>(static_cast<size_t>(1), ds.nb > 1 ? ds.nb - 1 : ds.nb));
    const uint32_t graph_degree = static_cast<uint32_t>(std::clamp(
        graph_degree_env, 1L, std::max(1L, max_degree_allowed)));
    const long build_passes_env = parse_env_long("FUSEDANN_PARLAY_BUILD_PASSES",
                                                 PARLAY_VAMANA_PASSES, 1);
    const int build_passes = static_cast<int>(
        std::max<long>(1, build_passes_env));
    const long visit_limit_env = parse_env_long("FUSEDANN_PARLAY_VISIT_LIMIT", PARLAY_VISIT_LIMIT);
    const long degree_limit_env = parse_env_long("FUSEDANN_PARLAY_DEGREE_LIMIT", PARLAY_DEGREE_LIMIT);
    // NOTE: current ParlayANN removed the beam-search "cut" parameter from
    // QueryParams, so this knob is parsed for backward compatibility but no
    // longer forwarded to the search. Retained so existing scripts/env do not break.
    [[maybe_unused]] const double query_cut = parse_env_double("FUSEDANN_PARLAY_QUERY_CUT",
                                              PARLAY_QUERY_CUT, 0.0);
    const double batch_factor = parse_env_double(
        "FUSEDANN_PARLAY_BATCH_FACTOR", PARLAY_BATCH_FACTOR, 0.0);
    const double rerank_factor = parse_env_double(
        "FUSEDANN_PARLAY_RERANK_FACTOR", PARLAY_RERANK_FACTOR, 0.0);
    const long rerank_k_env = parse_env_long("FUSEDANN_PARLAY_RERANK_K",
                                             static_cast<long>(PARLAY_RERANK_CANDIDATES),
                                             static_cast<long>(DEFAULT_K));
    size_t rerank_k = static_cast<size_t>(
        std::max<long>(rerank_k_env, static_cast<long>(DEFAULT_K)));
    if (beam_width > 0 && rerank_k >= static_cast<size_t>(beam_width))
    {
      const long new_beam = static_cast<long>(rerank_k + 1);
      if (new_beam > beam_width)
      {
        std::cout << "⚠️  Increasing beam width from " << beam_width
                  << " to " << new_beam
                  << " to cover rerank_k=" << rerank_k << "\n";
        beam_width = new_beam;
      }
    }
    const double final_mult = parse_env_double(
        "FUSEDANN_PARLAY_FINAL_CAND_MULT", 2.0, 1.0);
    size_t final_candidate_cap = std::max<size_t>(
        rerank_k,
        static_cast<size_t>(std::ceil(static_cast<double>(rerank_k) * final_mult)));
    if (beam_width > 0 && final_candidate_cap >= static_cast<size_t>(beam_width))
    {
      const long new_beam = static_cast<long>(final_candidate_cap + 1);
      if (new_beam > beam_width)
      {
        std::cout << "⚠️  Increasing beam width from " << beam_width
                  << " to " << new_beam
                  << " to admit " << final_candidate_cap << " candidates per query\n";
        beam_width = new_beam;
      }
    }

    std::cout << "ℹ️  ParlayANN search params: graph_degree=" << graph_degree
          << " build_passes=" << build_passes
          << " beam_width=" << beam_width
          << " visit_limit_env=" << visit_limit_env
          << " degree_limit_env=" << degree_limit_env
          << " final_candidate_cap=" << final_candidate_cap
          << " rerank_k=" << rerank_k
          << " (scores use fused distance: attr+content)\n";

    std::vector<Candidate> diagnostics_sort_buffer;
    if (diagnostics_enabled)
      diagnostics_sort_buffer.reserve(final_candidate_cap);

    auto capture_candidate_order = [&](size_t query_index,
                                       const std::vector<Candidate> &scores)
    {
      if (!diagnostics_enabled)
        return;
      auto &diag_ids = diagnostics_candidate_orders[query_index];
      diag_ids.clear();
      diagnostics_sort_buffer = scores;
      std::sort(diagnostics_sort_buffer.begin(), diagnostics_sort_buffer.end(),
                [](const Candidate &lhs, const Candidate &rhs)
                { return lhs.score < rhs.score; });
      diag_ids.reserve(diagnostics_sort_buffer.size());
      for (const auto &entry : diagnostics_sort_buffer)
        diag_ids.push_back(entry.id);
    };

    long partition_k = parse_env_long("FUSEDANN_PARTITION_K", FUSEDANN_PARTITION_K);
    if (partition_k > 0)
    {
      std::cout << "🧩 Partitioning enabled (K=" << partition_k << ").\n";
      if (ds.bitmaps_float.empty())
      {
        throw std::runtime_error("Attribute float buffer is empty; cannot partition without fused attributes.");
      }
    }

    const double attr_score_scale = parse_env_double(
        "FUSEDANN_PARLAY_ATTR_SCORE_SCALE", 1.0, -1.0);
    const double content_score_scale = parse_env_double(
        "FUSEDANN_PARLAY_CONTENT_SCORE_SCALE", 1.0, -1.0);
    // Default: use fused distance for stage-1 ranking (2-stage pipeline)
    // Set FUSEDANN_PARLAY_USE_FUSED_DISTANCE=0 to revert to weighted attr+content scoring
    const bool use_fused_distance = parse_env_flag(
        "FUSEDANN_PARLAY_USE_FUSED_DISTANCE", true);
    double attr_score_weight = 0.0;
    double content_score_weight = 0.0;

    const bool per_cluster_alpha_beta = parse_env_flag("FUSEDANN_PER_CLUSTER_ALPHA_BETA", false);
    const bool per_cluster_alpha_beta_debug = parse_env_flag("FUSEDANN_PER_CLUSTER_ALPHA_BETA_DEBUG", false);
    const long debug_query_id = parse_env_long_any("FUSEDANN_DEBUG_QUERY_ID", -1);
    const bool sparse_centroid_policy = ds.attributes_sparse && partition_k > 0;
    const bool partition_local_alpha_beta =
        per_cluster_alpha_beta || sparse_centroid_policy;

    if (ds.attributes_sparse && partition_k > 0)
      throw std::runtime_error(
          "Sparse FUSEDANN_PARTITION_K (centroid-residual) was removed; "
          "use FUSEDANN_TAG_GROUPS for filtered sparse search.");

    if (ds.attributes_sparse && partition_k == 0)
    {
      log_stage("Building sparse attribute groups and caches...");
      AttrGroupMap groups = build_sparse_attribute_groups(ds);

      double alpha = 0.0;
      double beta = 0.0;
      log_stage("Loading or estimating α/β heuristics...");
      if (!load_alpha_beta_or_throw(alpha, beta))
      {
        std::cout << "\n🧪 Estimating α/β heuristics (auto mode)...\n";
        AlphaBetaStats stats = auto_alpha_beta(ds, groups);
        alpha = stats.alpha;
        beta = stats.beta;
        write_alpha_beta(cache.alpha_beta, alpha, beta);
        std::cout << std::fixed << std::setprecision(4)
                  << "\n🧮 Auto α/β stats:\n"
                  << "   μ_max ≈ " << stats.mu_max
                  << " | Á_f ≈ " << stats.cluster_radius
                  << " | â_min ≈ " << stats.min_attr_distance << "\n"
                  << "   → α = " << stats.alpha
                  << ", β = " << stats.beta
                  << "   (cached at " << cache.alpha_beta << ")\n";
      }

      const double alpha_used = alpha * alpha_mult;
      const double beta_used = beta * beta_mult;
      std::cout << std::fixed << std::setprecision(4)
                << "🧮 α/β: real(α=" << alpha << ", β=" << beta
                << ") applied(α=" << alpha_used << ", β=" << beta_used << ")\n"
                << std::defaultfloat;

      // The per-attribute expansion / tag-partitioned path builds its own fused
      // vectors (per group) and uses the official/content GT directly, so it does
      // NOT need the single-point fused_base / fused_queries_gt / fused_gt below.
      // Skipping them in expand mode frees ~n·d floats (≈7.7 GB at 10M) — enough
      // headroom that stronger-α builds no longer thrash into swap.
      const bool expand_attrs = parse_env_flag("FUSEDANN_EXPAND_ATTRS", false);
      std::vector<float> fused_base;        // single-graph (non-expand) path only
      std::vector<float> fused_queries_gt;  // offline GT only
      if (!expand_attrs)
      {
        log_stage("Fusing base vectors for sparse attributes...");
        std::cout << "\n⚙️  Fusing base vectors...\n";
        if (ds.attr_fused_dim == 0)
          throw std::runtime_error("Sparse attribute PCA produced zero-dimensional projection; cannot fuse vectors.");
        fused_base.resize(ds.nb * ds.dim);
        Transformer::SingleTransform(ds.xb.data(), ds.bitmaps_float.data(), ds.nb, ds.dim, ds.attr_fused_dim,
                                     static_cast<float>(alpha_used), static_cast<float>(beta_used), fused_base.data());

        log_stage("Fusing query vectors for ground truth...");
        std::cout << "⚙️  Fusing query vectors (offline/GT)...\n";
        fused_queries_gt.resize(ds.nq * ds.dim);
        Transformer::SingleTransform(ds.xq.data(), ds.q_bitmaps_float.data(), ds.nq, ds.dim, ds.attr_fused_dim,
                                     static_cast<float>(alpha_used), static_cast<float>(beta_used), fused_queries_gt.data());
      }

      log_stage("Building sparse filter index...");
      SparseFilterIndex filter_index = build_sparse_filter_index(ds);

      std::vector<int32_t> fused_gt;
      if (!expand_attrs)
      {
      bool force_soft_gt = (std::getenv("FUSEDANN_FORCE_SOFT_GT") != nullptr);

      log_stage("Loading or computing fused-space ground truth (sparse)...");
      if (!force_soft_gt && load_fused_groundtruth(cache.fused_gt, ds.nq, DEFAULT_K, ds.nb, fused_gt))
      {
        std::cout << "📦 Loaded fused-space ground truth from " << cache.fused_gt << "\n";
      }
      else
      {
        if (!force_soft_gt && !ds.groundtruth.empty() && ds.gt_k >= DEFAULT_K)
        {
          std::cout << "📦 Using provided filtered ground truth from dataset (scanning first "
                    << ds.gt_k << " entries for " << DEFAULT_K << " valid neighbors).\n";
          fused_gt.resize(ds.nq * DEFAULT_K);
          for (size_t i = 0; i < ds.nq; ++i)
          {
            size_t found = 0;
            for (size_t j = 0; j < ds.gt_k && found < DEFAULT_K; ++j)
            {
              int32_t id = ds.groundtruth[i * ds.gt_k + j];
              if (id >= 0 && static_cast<size_t>(id) < ds.nb)
              {
                fused_gt[i * DEFAULT_K + found] = id;
                found++;
              }
            }
            while (found < DEFAULT_K)
            {
              fused_gt[i * DEFAULT_K + found] = -1;
              found++;
            }
          }
          write_fused_groundtruth(cache.fused_gt, fused_gt, ds.nq, DEFAULT_K);
          std::cout << "💾 Cached dataset ground truth at " << cache.fused_gt << "\n";
        }
        else
        {
          if (force_soft_gt)
          {
            std::cout << "⚠️  Forcing computation of Soft Ground Truth (ignoring provided GT file)...\n";
          }
          std::cout << "⏳ Computing attribute-filtered fused-space ground truth (sparse)...\n";
          fused_gt = compute_sparse_filtered_groundtruth(ds,
                                                         filter_index,
                                                         fused_base,
                                                         fused_queries_gt,
                                                         ds.dim,
                                                         DEFAULT_K);
          if (!force_soft_gt)
          {
            write_fused_groundtruth(cache.fused_gt, fused_gt, ds.nq, DEFAULT_K);
            std::cout << "💾 Stored fused-space ground truth at " << cache.fused_gt << "\n";
          }
          else
          {
            std::cout << "⚠️  Skipping cache write for forced Soft GT to avoid polluting cache.\n";
          }
        }
      }
      } // end if (!expand_attrs): single-point fused GT only used by single-graph path

      // ====================================================================
      // PER-ATTRIBUTE EXPANSION (FUSEDANN_EXPAND_ATTRS=1)
      // Index each (doc, tag) as its own fused node = fuse(content, e[tag]),
      // where e[tag] is a distinct unit embedding for that single tag. A query
      // with m tags fuses to m points (one per tag); searching each lands next
      // to docs that actually carry that tag, so valid superset docs are no
      // longer pushed away. Fixes bag-of-tags subset filtering (BigANN/YFCC).
      // Cost: graph grows from n_docs to nnz nodes (see README "Results").
      // ====================================================================
      if (expand_attrs)
      {
        if (ds.attr_indptr.size() != ds.nb + 1 || ds.q_attr_indptr.size() != ds.nq + 1)
          throw std::runtime_error("FUSEDANN_EXPAND_ATTRS requires sparse CSR attributes (base + query).");

        // ==================================================================
        // TAG-PARTITIONED SUBSPACES (FUSEDANN_TAG_GROUPS=G > 1)
        // Split tags into G groups (tag % G). Each group is its own cached
        // Vamana sub-index over the (doc, tag-in-group) nodes, with attribute =
        // group-local offset one-hot (dim FUSEDANN_GROUP_ATTR_DIM). A query tag
        // is searched ONLY in its group's subspace (filter-first routing), so
        // per-query work ≈ 1/G. Groups are processed sequentially (build →
        // search → free), so peak memory ≈ one subspace, which also makes large
        // scales (10M) feasible without materializing all nnz nodes at once.
        // ==================================================================
        const int tag_groups = static_cast<int>(parse_env_long("FUSEDANN_TAG_GROUPS", 1));
        if (tag_groups > 1)
        {
          size_t gdim = static_cast<size_t>(parse_env_long(
              "FUSEDANN_GROUP_ATTR_DIM",
              static_cast<long>(ds.attr_fused_dim ? ds.attr_fused_dim : 16)));
          gdim = std::max<size_t>(1, std::min<size_t>(gdim, ds.dim));
          // Per-tag attribute embedding within a subspace: distinct random unit
          // vector per tag (default, collision-free) or the group-local offset
          // one-hot (FUSEDANN_GROUP_ONEHOT=1, dim buckets → collisions at small dim).
          const bool grp_onehot = parse_env_flag("FUSEDANN_GROUP_ONEHOT", false);
          log_stage("Tag-partitioned expansion (G groups)...");
          std::cout << "   • tag_groups=" << tag_groups << ", group_attr_dim=" << gdim
                    << ", embed=" << (grp_onehot ? "offset-onehot" : "per-tag-random")
                    << ", total (doc,tag) nodes=" << ds.attr_indices.size() << "\n";

          // SCALE FIX (default on): the per-tag embedding is a UNIT vector (‖e‖=1),
          // but raw YFCC content has ‖v‖≈1820, so in fuse=(v-αe)/β the tag term (≈α)
          // is ~0.1% of the content signal — the graph search becomes blind to tags
          // and returns content-nearest (wrong-tag) docs that fail the subset filter.
          // L2-normalizing content makes ‖v‖=1 so the tag embedding is on-scale and
          // same-tag docs form tight clusters. This is REQUIRED together with seeded
          // routing (below): normalization alone (global-start search) only reaches
          // ~82% coverage, but normalization + same-tag seed reaches ~99.9% coverage
          // and ~0.93 recall@10 on 10M. Disable with FUSEDANN_NORMALIZE_CONTENT=0.
          // IMPORTANT: normalization is applied ONLY when fusing the per-(doc,tag)
          // base/query nodes below (so the unit tag embedding is on-scale for graph
          // routing). ds.xb / ds.xq are left RAW so the final content-rerank uses the
          // SAME raw-L2 metric as the official ground truth. Normalizing the rerank
          // too (the earlier in-place approach) reorders near-equal-norm neighbors and
          // caps recall@10 at ~0.95 regardless of search budget.
          const bool normalize_content = parse_env_flag("FUSEDANN_NORMALIZE_CONTENT", true);
          if (normalize_content)
            std::cout << "   • L2-normalizing content for FUSION/routing only (rerank stays on raw content)\n";

          // Bucket base (doc,tag) and query (qi,tag) entries by group.
          std::vector<std::vector<std::pair<uint32_t, int32_t>>> grp_nodes(tag_groups);
          std::vector<std::vector<std::pair<uint32_t, int32_t>>> grp_qnodes(tag_groups);
          for (size_t doc = 0; doc < ds.nb; ++doc)
            for (int64_t p = ds.attr_indptr[doc]; p < ds.attr_indptr[doc + 1]; ++p)
            {
              const int32_t tag = ds.attr_indices[static_cast<size_t>(p)];
              grp_nodes[fusedann::tag_group_of(tag, tag_groups)].emplace_back(static_cast<uint32_t>(doc), tag);
            }
          for (size_t qi = 0; qi < ds.nq; ++qi)
            for (int64_t p = ds.q_attr_indptr[qi]; p < ds.q_attr_indptr[qi + 1]; ++p)
            {
              const int32_t tag = ds.q_attr_indices[static_cast<size_t>(p)];
              grp_qnodes[fusedann::tag_group_of(tag, tag_groups)].emplace_back(static_cast<uint32_t>(qi), tag);
            }

          std::vector<std::vector<uint32_t>> cand_docs(ds.nq); // accumulated candidate doc ids per query
          double search_seconds = 0.0;
          const auto qphase_start = std::chrono::steady_clock::now();

          for (int g = 0; g < tag_groups; ++g)
          {
            const size_t ng = grp_nodes[g].size();
            if (ng == 0 || grp_qnodes[g].empty()) continue; // skip groups no query touches

            // Build this subspace's fused base nodes.
            std::vector<float> fb(ng * ds.dim);
            std::vector<uint32_t> node_doc(ng);
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
              std::vector<float> emb(gdim);
              std::vector<float> cbuf(ds.dim); // per-thread normalized-content scratch (fusion only)
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
              for (size_t i = 0; i < ng; ++i)
              {
                const uint32_t doc = grp_nodes[g][i].first;
                if (grp_onehot) fusedann::make_group_onehot(grp_nodes[g][i].second, tag_groups, gdim, emb.data());
                else fusedann::make_tag_embedding(grp_nodes[g][i].second, gdim, emb.data());
                const float *csrc = ds.xb.data() + static_cast<size_t>(doc) * ds.dim;
                if (normalize_content)
                {
                  double s = 0.0;
                  for (size_t j = 0; j < ds.dim; ++j) s += static_cast<double>(csrc[j]) * csrc[j];
                  const float inv = (s > 1e-24) ? static_cast<float>(1.0 / std::sqrt(s)) : 1.0f;
                  for (size_t j = 0; j < ds.dim; ++j) cbuf[j] = csrc[j] * inv;
                  csrc = cbuf.data();
                }
                fuse_single_query(csrc, emb.data(), ds.dim, gdim,
                                  static_cast<float>(alpha_used), static_cast<float>(beta_used),
                                  fb.data() + i * ds.dim);
                node_doc[i] = doc;
              }
            }

            FloatMatrixView gv{fb.data(), ng, ds.dim};
            PointRange gpr(gv);

            // Build or load this group's sub-graph (cached per group).
            GraphCacheMeta gm{};
            gm.nb = ng; gm.dim = ds.dim; gm.attr_dim = gdim; gm.degree = graph_degree;
            gm.passes = build_passes;
            gm.base_hash = hash_path(fs::absolute(base_path).string());
            gm.attr_hash = hash_path(fs::absolute(base_attr_path).string())
                         ^ (static_cast<uint64_t>(g + 1) * 0x9E3779B97F4A7C15ull);
            gm.alpha_real = alpha; gm.beta_real = beta; gm.alpha_mult = alpha_mult; gm.beta_mult = beta_mult;
            const std::string suffix = "vamana_g" + std::to_string(tag_groups) + "_p" + std::to_string(g);
            const fs::path gbin = cache.dir / (suffix + ".bin");
            const fs::path gmeta = cache.dir / (suffix + ".meta");
            std::unique_ptr<pann::Graph<uint32_t>> gg;
            uint32_t gstart = 0;
            GraphCacheMeta gc{};
            if (fs::exists(gbin) && fs::exists(gmeta) && read_graph_metadata(gmeta, gc) &&
                graph_metadata_matches(gc, gm))
            {
              std::string s = gbin.string();
              gg = std::make_unique<pann::Graph<uint32_t>>(s.data());
              gstart = gc.start_id;
            }
            else
            {
              const long bb = parse_env_long("FUSEDANN_PARLAY_BUILD_BEAM", beam_width);
              pann::BuildParams bp(static_cast<long>(graph_degree), bb, PARLAY_VAMANA_ALPHA, build_passes);
              gg = std::make_unique<pann::Graph<uint32_t>>(graph_degree, ng);
              pann::stats<uint32_t> bs(ng);
              pann::knn_index<PointRange, PointRange, uint32_t> idx(bp);
              idx.build_index(*gg, gpr, gpr, bs);
              gstart = idx.get_start();
              GraphCacheMeta tw = gm; tw.start_id = gstart;
              std::string s = gbin.string();
              gg->save(s.data());
              write_graph_metadata(gmeta, tw);
            }

            // Build this group's query nodes and search.
            const size_t nqg = grp_qnodes[g].size();
            std::vector<float> qfb(nqg * ds.dim);
            std::vector<uint32_t> qp_query(nqg);
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
              std::vector<float> emb(gdim);
              std::vector<float> cbuf(ds.dim); // per-thread normalized-content scratch (fusion only)
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
              for (size_t i = 0; i < nqg; ++i)
              {
                const uint32_t qi = grp_qnodes[g][i].first;
                if (grp_onehot) fusedann::make_group_onehot(grp_qnodes[g][i].second, tag_groups, gdim, emb.data());
                else fusedann::make_tag_embedding(grp_qnodes[g][i].second, gdim, emb.data());
                const float *csrc = ds.xq.data() + static_cast<size_t>(qi) * ds.dim;
                if (normalize_content)
                {
                  double s = 0.0;
                  for (size_t j = 0; j < ds.dim; ++j) s += static_cast<double>(csrc[j]) * csrc[j];
                  const float inv = (s > 1e-24) ? static_cast<float>(1.0 / std::sqrt(s)) : 1.0f;
                  for (size_t j = 0; j < ds.dim; ++j) cbuf[j] = csrc[j] * inv;
                  csrc = cbuf.data();
                }
                fuse_single_query(csrc, emb.data(), ds.dim, gdim,
                                  static_cast<float>(alpha_used), static_cast<float>(beta_used),
                                  qfb.data() + i * ds.dim);
                qp_query[i] = qi;
              }
            }
            FloatMatrixView qv{qfb.data(), nqg, ds.dim};
            PointRange qpr(qv, gpr.params);
            const long ev = std::max<long>(beam_width, std::min<long>(visit_limit_env, static_cast<long>(ng)));
            const long ed = std::max<long>(1, std::min<long>(degree_limit_env, static_cast<long>(gg->max_degree())));
            pann::QueryParams qpp(static_cast<long>(final_candidate_cap), beam_width, ev, ed, rerank_factor, batch_factor);

            // SEEDED ROUTING (default on; FUSEDANN_SEED_ROUTING=0 to disable):
            // start each query's beam search from a node that actually carries the
            // query's tag, so the search begins INSIDE the correct same-tag cluster
            // rather than at a global medoid. Paired with tag-dominant fusion
            // (FUSEDANN_NORMALIZE_CONTENT=1) this is what makes per-group routing
            // reliable — it lifts coverage from ~82% toward ~100%.
            const bool seed_routing = parse_env_flag("FUSEDANN_SEED_ROUTING", true);
            // The tag -> representative-node map is part of the index (depends only
            // on base nodes), NOT per-query work, so build it OUTSIDE the timed
            // search region — same way the graph load / query fusion are excluded.
            std::unordered_map<int32_t, uint32_t> tag_to_node;
            if (seed_routing)
            {
              tag_to_node.reserve(ng * 2);
              for (uint32_t i2 = 0; i2 < ng; ++i2)
                tag_to_node.emplace(grp_nodes[g][i2].second, i2); // first node per tag
            }
            const auto ss = std::chrono::steady_clock::now();
            if (seed_routing)
            {
              // NOTE: a query carrying two tags that fall in the SAME group appears
              // twice in grp_qnodes[g] with the same global qi, so we must NOT write
              // cand_docs[qi] directly from the parallel loop (data race). Collect
              // per query-node into a private buffer, then merge sequentially.
              std::vector<std::vector<uint32_t>> per_q(nqg);
              parlay::parallel_for(0, nqg, [&](size_t p)
              {
                const int32_t qtag = grp_qnodes[g][p].second;
                auto it = tag_to_node.find(qtag);
                const uint32_t seed = (it != tag_to_node.end()) ? it->second : gstart;
                parlay::sequence<uint32_t> sp = {seed};
                auto [pairElts, dc] = pann::beam_search(qpr[p], *gg, gpr, sp, qpp);
                (void)dc;
                auto &beamElts = pairElts.first;
                const size_t topn = std::min<size_t>(beamElts.size(), final_candidate_cap);
                per_q[p].reserve(topn);
                for (size_t k = 0; k < topn; ++k)
                  if (beamElts[k].first < ng) per_q[p].push_back(node_doc[beamElts[k].first]);
              });
              for (size_t p = 0; p < nqg; ++p)
              {
                auto &dst = cand_docs[qp_query[p]];
                dst.insert(dst.end(), per_q[p].begin(), per_q[p].end());
              }
            }
            else
            {
              pann::stats<uint32_t> qs(nqg);
              auto res = pann::searchAll(qpr, *gg, gpr, qs, gstart, qpp);
              for (size_t p = 0; p < nqg; ++p)
              {
                const auto &nb = res[p];
                const size_t topn = std::min(nb.size(), final_candidate_cap);
                auto &dst = cand_docs[qp_query[p]];
                for (size_t k = 0; k < topn; ++k)
                  if (nb[k] < ng) dst.push_back(node_doc[nb[k]]);
              }
            }
            const auto se = std::chrono::steady_clock::now();
            search_seconds += std::chrono::duration<double>(se - ss).count();
            std::cout << "   • group " << g << ": nodes=" << ng << " query_nodes=" << nqg
                      << (seed_routing ? " (seeded)" : "") << "\n";
          } // groups

          // Shared post-processing: dedup docs, exact subset filter, content rerank.
          const auto pp_start = std::chrono::steady_clock::now();
          std::vector<int32_t> ann_results(ds.nq * DEFAULT_K, -1);
          size_t queries_with_results = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
          {
            std::vector<uint8_t> visited(ds.nb, 0);
            std::vector<uint32_t> touched;
            std::vector<Candidate> filtered;
            size_t lqwr = 0;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
            for (size_t qi = 0; qi < ds.nq; ++qi)
            {
              touched.clear(); filtered.clear();
              const int64_t tb = ds.q_attr_indptr[qi], te = ds.q_attr_indptr[qi + 1];
              const size_t tcount = static_cast<size_t>(te - tb);
              const int32_t *qtags = ds.q_attr_indices.data() + tb;
              const float *qvec = ds.xq.data() + static_cast<size_t>(qi) * ds.dim;
              for (uint32_t doc : cand_docs[qi])
              {
                if (doc >= ds.nb || visited[doc]) continue;
                visited[doc] = 1; touched.push_back(doc);
                if (!doc_matches_sparse(ds, doc, qtags, tcount)) continue;
                Candidate c;
                c.id = static_cast<int32_t>(doc);
                c.content_dist = l2_distance(ds.xb.data() + static_cast<size_t>(doc) * ds.dim, qvec, ds.dim);
                filtered.push_back(c);
              }
              for (uint32_t d : touched) visited[d] = 0;
              if (filtered.size() > DEFAULT_K)
              {
                std::partial_sort(filtered.begin(), filtered.begin() + DEFAULT_K, filtered.end(),
                                  [](const Candidate &a, const Candidate &b) { return a.content_dist < b.content_dist; });
                filtered.resize(DEFAULT_K);
              }
              else
              {
                std::sort(filtered.begin(), filtered.end(),
                          [](const Candidate &a, const Candidate &b) { return a.content_dist < b.content_dist; });
              }
              if (!filtered.empty()) ++lqwr;
              for (size_t k = 0; k < DEFAULT_K; ++k)
                ann_results[qi * DEFAULT_K + k] = (k < filtered.size()) ? filtered[k].id : -1;
            }
#ifdef _OPENMP
#pragma omp atomic
#endif
            queries_with_results += lqwr;
          }
          const auto qphase_end = std::chrono::steady_clock::now();
          const double postproc_seconds = std::chrono::duration<double>(qphase_end - pp_start).count();
          const double query_seconds = search_seconds + postproc_seconds; // excludes one-time build/load
          const double total_seconds = std::chrono::duration<double>(qphase_end - qphase_start).count();

          std::vector<int32_t> gt;
          const bool force_soft_gt_grp = (std::getenv("FUSEDANN_FORCE_SOFT_GT") != nullptr);
          if (!force_soft_gt_grp && !ds.groundtruth.empty() && ds.gt_k >= DEFAULT_K)
          {
            gt.assign(ds.nq * DEFAULT_K, -1);
            for (size_t i = 0; i < ds.nq; ++i)
            {
              size_t f = 0;
              for (size_t j = 0; j < ds.gt_k && f < DEFAULT_K; ++j)
              {
                const int32_t id = ds.groundtruth[i * ds.gt_k + j];
                if (id >= 0 && static_cast<size_t>(id) < ds.nb) gt[i * DEFAULT_K + f++] = id;
              }
            }
            std::cout << "📦 Using provided (official) ground truth.\n";
          }
          else
          {
            const fs::path cg = cache.dir / "content_gt.ivecs";
            std::vector<int32_t> loaded;
            if (load_fused_groundtruth(cg, ds.nq, DEFAULT_K, ds.nb, loaded))
            {
              gt = std::move(loaded);
              std::cout << "📦 Loaded cached content-space filtered GT.\n";
            }
            else
            {
              std::cout << "⏳ Computing content-space filtered ground truth...\n";
              gt = compute_content_filtered_groundtruth(ds, filter_index, DEFAULT_K);
              write_fused_groundtruth(cg, gt, ds.nq, DEFAULT_K);
            }
          }

          const double recall = compute_recall(ann_results, gt, ds.nq, DEFAULT_K, DEFAULT_K, ds.nb);
          // --- DIAGNOSTIC: recall & coverage bucketed by query tag count (1 vs >=2) ---
          {
            size_t n1 = 0, n2 = 0, cov1 = 0, cov2 = 0;
            double rec1 = 0.0, rec2 = 0.0;
            for (size_t qi = 0; qi < ds.nq; ++qi)
            {
              const size_t tc = static_cast<size_t>(ds.q_attr_indptr[qi + 1] - ds.q_attr_indptr[qi]);
              size_t valid = 0, hit = 0;
              for (size_t k = 0; k < DEFAULT_K; ++k)
              {
                const int32_t gg2 = gt[qi * DEFAULT_K + k];
                if (gg2 < 0) continue;
                ++valid;
                for (size_t j = 0; j < DEFAULT_K; ++j)
                  if (ann_results[qi * DEFAULT_K + j] == gg2) { ++hit; break; }
              }
              const bool has_res = ann_results[qi * DEFAULT_K] >= 0;
              const double r = valid ? static_cast<double>(hit) / static_cast<double>(valid) : 0.0;
              if (tc <= 1) { ++n1; rec1 += r; if (has_res) ++cov1; }
              else { ++n2; rec2 += r; if (has_res) ++cov2; }
            }
            std::cout << std::setprecision(4)
                      << "🔬 single-tag: n=" << n1 << " recall=" << (n1 ? rec1 / n1 : 0.0)
                      << " coverage=" << (n1 ? static_cast<double>(cov1) / static_cast<double>(n1) : 0.0) << "\n"
                      << "🔬 multi-tag : n=" << n2 << " recall=" << (n2 ? rec2 / n2 : 0.0)
                      << " coverage=" << (n2 ? static_cast<double>(cov2) / static_cast<double>(n2) : 0.0) << "\n";
          }
          const double qps = query_seconds > 0.0 ? static_cast<double>(ds.nq) / query_seconds : 0.0;
          std::cout << std::fixed << std::setprecision(4)
                    << "\n📈 FusedANN recall (tag-partitioned, G=" << tag_groups << "): " << recall << "\n"
                    << std::setprecision(2)
                    << "⚡ QPS: " << qps << " queries/s [search+rerank] (search " << search_seconds
                    << " s; rerank " << postproc_seconds << " s; total incl. build/load " << total_seconds << " s)\n"
                    << "🔎 Queries with results: " << queries_with_results << " / " << ds.nq << "\n"
                    << "🎉 Done (tag-partitioned).\n";
          return EXIT_SUCCESS;
        }

        const size_t attr_e_dim = ds.attr_fused_dim ? ds.attr_fused_dim : 1;
        const size_t n_exp = ds.attr_indices.size();    // base nodes = total base tags
        const size_t q_nexp = ds.q_attr_indices.size(); // query nodes = total query tags
        log_stage("Per-attribute expansion: one fused node per (doc, tag)...");
        std::cout << "   • " << ds.nb << " docs -> " << n_exp << " nodes ("
                  << std::fixed << std::setprecision(2)
                  << (ds.nb ? double(n_exp) / double(ds.nb) : 0.0)
                  << " tags/doc), attr_e_dim=" << attr_e_dim << "\n";

        // 1) Expanded fused base: fuse(content_doc, e[tag]) for every (doc, tag).
        std::vector<float> fused_exp(n_exp * ds.dim);
        std::vector<uint32_t> exp_to_doc(n_exp);
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          std::vector<float> emb(attr_e_dim);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
          for (size_t doc = 0; doc < ds.nb; ++doc)
          {
            const float *content = ds.xb.data() + doc * ds.dim;
            for (int64_t p = ds.attr_indptr[doc]; p < ds.attr_indptr[doc + 1]; ++p)
            {
              fusedann::make_tag_embedding(ds.attr_indices[static_cast<size_t>(p)], attr_e_dim, emb.data());
              fuse_single_query(content, emb.data(), ds.dim, attr_e_dim,
                                static_cast<float>(alpha_used), static_cast<float>(beta_used),
                                fused_exp.data() + static_cast<size_t>(p) * ds.dim);
              exp_to_doc[static_cast<size_t>(p)] = static_cast<uint32_t>(doc);
            }
          }
        }

        // 2) Build (or load) the Vamana graph over the expanded nodes. The graph
        //    is cached keyed on build-determining params (nnz, dim, degree, passes,
        //    alpha/beta, attr dim, data hashes) so search-only sweeps (varying beam /
        //    rerank_k) reuse it. Build beam is decoupled from search beam.
        FloatMatrixView exp_view{fused_exp.data(), n_exp, ds.dim};
        PointRange exp_points(exp_view);
        const long build_beam = parse_env_long("FUSEDANN_PARLAY_BUILD_BEAM", beam_width);

        GraphCacheMeta exp_expected{};
        exp_expected.nb = n_exp;
        exp_expected.dim = ds.dim;
        exp_expected.attr_dim = attr_e_dim;
        exp_expected.degree = graph_degree;
        exp_expected.passes = build_passes;
        exp_expected.base_hash = hash_path(fs::absolute(base_path).string());
        exp_expected.attr_hash = hash_path(fs::absolute(base_attr_path).string());
        exp_expected.alpha_real = alpha;
        exp_expected.beta_real = beta;
        exp_expected.alpha_mult = alpha_mult;
        exp_expected.beta_mult = beta_mult;
        const fs::path exp_graph_bin = cache.dir / "vamana_expanded.bin";
        const fs::path exp_graph_meta = cache.dir / "vamana_expanded.meta";

        std::unique_ptr<pann::Graph<uint32_t>> graph_ptr;
        uint32_t start_point = 0;
        GraphCacheMeta exp_cached{};
        if (fs::exists(exp_graph_bin) && fs::exists(exp_graph_meta) &&
            read_graph_metadata(exp_graph_meta, exp_cached) &&
            graph_metadata_matches(exp_cached, exp_expected))
        {
          std::string gp = exp_graph_bin.string();
          graph_ptr = std::make_unique<pann::Graph<uint32_t>>(gp.data());
          start_point = exp_cached.start_id;
          std::cout << "📦 Loaded expanded Vamana graph from " << exp_graph_bin << "\n";
        }
        else
        {
          pann::BuildParams build_params(static_cast<long>(graph_degree), build_beam,
                                         PARLAY_VAMANA_ALPHA, build_passes);
          graph_ptr = std::make_unique<pann::Graph<uint32_t>>(graph_degree, n_exp);
          pann::stats<uint32_t> build_stats(n_exp);
          pann::knn_index<PointRange, PointRange, uint32_t> index(build_params);
          std::cout << "🏗️  Building expanded Vamana graph (" << n_exp << " nodes, degree "
                    << graph_degree << ", passes " << build_passes
                    << ", build_beam " << build_beam << ")...\n";
          index.build_index(*graph_ptr, exp_points, exp_points, build_stats);
          start_point = index.get_start();
          GraphCacheMeta tw = exp_expected;
          tw.start_id = start_point;
          std::string gp = exp_graph_bin.string();
          graph_ptr->save(gp.data());
          write_graph_metadata(exp_graph_meta, tw);
          std::cout << "💾 Stored expanded Vamana graph at " << exp_graph_bin << "\n";
        }

        // 3) Query nodes: fuse(content_q, e[tag]) for every (query, tag).
        std::vector<float> fused_qexp(q_nexp * ds.dim);
        std::vector<uint32_t> qexp_to_query(q_nexp);
        const auto qphase_start = std::chrono::steady_clock::now();
        {
          std::vector<float> emb(attr_e_dim);
          for (size_t qi = 0; qi < ds.nq; ++qi)
          {
            const float *content = ds.xq.data() + qi * ds.dim;
            for (int64_t p = ds.q_attr_indptr[qi]; p < ds.q_attr_indptr[qi + 1]; ++p)
            {
              fusedann::make_tag_embedding(ds.q_attr_indices[static_cast<size_t>(p)], attr_e_dim, emb.data());
              fuse_single_query(content, emb.data(), ds.dim, attr_e_dim,
                                static_cast<float>(alpha_used), static_cast<float>(beta_used),
                                fused_qexp.data() + static_cast<size_t>(p) * ds.dim);
              qexp_to_query[static_cast<size_t>(p)] = static_cast<uint32_t>(qi);
            }
          }
        }

        // 4) One batched ANN search over all query nodes.
        FloatMatrixView qexp_view{fused_qexp.data(), q_nexp, ds.dim};
        PointRange qexp_points(qexp_view, exp_points.params);
        const long e_visit = std::max<long>(beam_width, std::min<long>(visit_limit_env, static_cast<long>(n_exp)));
        const long e_deg = std::max<long>(1, std::min<long>(degree_limit_env, static_cast<long>(graph_ptr->max_degree())));
        pann::QueryParams e_qp(static_cast<long>(final_candidate_cap), beam_width,
                               e_visit, e_deg, rerank_factor, batch_factor);
        pann::stats<uint32_t> e_stats(q_nexp);
        const auto search_start = std::chrono::steady_clock::now();
        auto e_results = pann::searchAll(qexp_points, *graph_ptr, exp_points, e_stats, start_point, e_qp);
        const auto search_end = std::chrono::steady_clock::now();
        const double search_seconds = std::chrono::duration<double>(search_end - search_start).count();

        // 5) Map expanded hits -> parent docs (union over the query's tag nodes),
        //    apply the exact subset filter, rerank by content distance.
        std::vector<int32_t> ann_results(ds.nq * DEFAULT_K, -1);
        std::vector<std::vector<uint32_t>> cand_points(ds.nq);
        for (size_t p = 0; p < q_nexp; ++p)
        {
          const uint32_t qi = qexp_to_query[p];
          const auto &nbrs = e_results[p];
          const size_t topn = std::min(nbrs.size(), final_candidate_cap);
          auto &dst = cand_points[qi];
          for (size_t k = 0; k < topn; ++k) dst.push_back(nbrs[k]);
        }

        size_t queries_with_results = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          std::vector<uint8_t> visited(ds.nb, 0); // thread-local
          std::vector<uint32_t> touched;
          std::vector<Candidate> filtered;
          size_t local_qwr = 0;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
          for (size_t qi = 0; qi < ds.nq; ++qi)
          {
            touched.clear();
            filtered.clear();
            const int64_t tb = ds.q_attr_indptr[qi];
            const int64_t te = ds.q_attr_indptr[qi + 1];
            const size_t tcount = static_cast<size_t>(te - tb);
            const int32_t *qtags = ds.q_attr_indices.data() + tb;
            const float *qvec = ds.xq.data() + qi * ds.dim;
            for (uint32_t ep : cand_points[qi])
            {
              if (ep >= n_exp) continue;
              const uint32_t doc = exp_to_doc[ep];
              if (doc >= ds.nb || visited[doc]) continue;
              visited[doc] = 1;
              touched.push_back(doc);
              if (!doc_matches_sparse(ds, doc, qtags, tcount)) continue;
              Candidate c;
              c.id = static_cast<int32_t>(doc);
              c.content_dist = l2_distance(ds.xb.data() + static_cast<size_t>(doc) * ds.dim, qvec, ds.dim);
              filtered.push_back(c);
            }
            for (uint32_t d : touched) visited[d] = 0;
            if (filtered.size() > DEFAULT_K)
            {
              std::partial_sort(filtered.begin(), filtered.begin() + DEFAULT_K, filtered.end(),
                                [](const Candidate &a, const Candidate &b) { return a.content_dist < b.content_dist; });
              filtered.resize(DEFAULT_K);
            }
            else
            {
              std::sort(filtered.begin(), filtered.end(),
                        [](const Candidate &a, const Candidate &b) { return a.content_dist < b.content_dist; });
            }
            if (!filtered.empty()) ++local_qwr;
            for (size_t k = 0; k < DEFAULT_K; ++k)
              ann_results[qi * DEFAULT_K + k] = (k < filtered.size()) ? filtered[k].id : -1;
          }
#ifdef _OPENMP
#pragma omp atomic
#endif
          queries_with_results += local_qwr;
        }
        const auto qphase_end = std::chrono::steady_clock::now();
        const double query_seconds = std::chrono::duration<double>(qphase_end - qphase_start).count();

        // 6) Ground truth: official (doc-level) if provided, else content-space filtered GT.
        std::vector<int32_t> gt;
        if (!ds.groundtruth.empty() && ds.gt_k >= DEFAULT_K)
        {
          gt.assign(ds.nq * DEFAULT_K, -1);
          for (size_t i = 0; i < ds.nq; ++i)
          {
            size_t f = 0;
            for (size_t j = 0; j < ds.gt_k && f < DEFAULT_K; ++j)
            {
              const int32_t id = ds.groundtruth[i * ds.gt_k + j];
              if (id >= 0 && static_cast<size_t>(id) < ds.nb) gt[i * DEFAULT_K + f++] = id;
            }
          }
          std::cout << "📦 Using provided (official) ground truth for expanded eval.\n";
        }
        else
        {
          const fs::path content_gt_path = cache.dir / "content_gt.ivecs";
          std::vector<int32_t> loaded;
          if (load_fused_groundtruth(content_gt_path, ds.nq, DEFAULT_K, ds.nb, loaded))
          {
            gt = std::move(loaded);
            std::cout << "📦 Loaded cached content-space filtered GT from " << content_gt_path << "\n";
          }
          else
          {
            std::cout << "⏳ Computing content-space filtered ground truth for expanded eval...\n";
            gt = compute_content_filtered_groundtruth(ds, filter_index, DEFAULT_K);
            write_fused_groundtruth(content_gt_path, gt, ds.nq, DEFAULT_K);
            std::cout << "💾 Cached content-space filtered GT at " << content_gt_path << "\n";
          }
        }

        const double recall = compute_recall(ann_results, gt, ds.nq, DEFAULT_K, DEFAULT_K, ds.nb);
        const double qps = query_seconds > 0.0 ? static_cast<double>(ds.nq) / query_seconds : 0.0;
        std::cout << std::fixed << std::setprecision(4)
                  << "\n📈 FusedANN recall (per-attribute expansion): " << recall << "\n"
                  << std::setprecision(2)
                  << "⚡ QPS: " << qps << " queries/s (query " << query_seconds
                  << " s; graph/search " << search_seconds << " s; " << n_exp << " nodes)\n"
                  << "🔎 Queries with results: " << queries_with_results << " / " << ds.nq << "\n"
                  << "🎉 Done (expanded).\n";
        return EXIT_SUCCESS;
      }

      FloatMatrixView base_view{fused_base.data(), ds.nb, ds.dim};
      PointRange base_points(base_view);

      const uint64_t base_hash = hash_path(fs::absolute(base_path).string());
      const uint64_t attr_hash = hash_path(fs::absolute(base_attr_path).string());

      GraphCacheMeta expected_meta;
      expected_meta.nb = ds.nb;
      expected_meta.dim = ds.dim;
      expected_meta.attr_dim = ds.attr_fused_dim;
      expected_meta.degree = graph_degree;
      expected_meta.passes = build_passes;
      expected_meta.base_hash = base_hash;
      expected_meta.attr_hash = attr_hash;
      expected_meta.alpha_real = alpha;
      expected_meta.beta_real = beta;
      expected_meta.alpha_mult = alpha_mult;
      expected_meta.beta_mult = beta_mult;

      GraphCacheMeta cached_meta;
      std::unique_ptr<pann::Graph<uint32_t>> graph_ptr;
      uint32_t start_point = 0;

      log_stage("Preparing sparse Vamana graph (load/build)...");
      if (fs::exists(cache.graph_bin) && fs::exists(cache.graph_meta) &&
          read_graph_metadata(cache.graph_meta, cached_meta) &&
          graph_metadata_matches(cached_meta, expected_meta))
      {
        std::string graph_path = cache.graph_bin.string();
        graph_ptr = std::make_unique<pann::Graph<uint32_t>>(graph_path.data());
        start_point = cached_meta.start_id;
        std::cout << "📦 Loaded Vamana graph from " << cache.graph_bin << "\n";
      }
      else
      {
        pann::BuildParams build_params(static_cast<long>(graph_degree),
                                       beam_width,
                                       PARLAY_VAMANA_ALPHA,
                                       build_passes);

        graph_ptr = std::make_unique<pann::Graph<uint32_t>>(graph_degree, ds.nb);
        pann::Graph<uint32_t> &graph = *graph_ptr;
        pann::stats<uint32_t> build_stats(ds.nb);
        pann::knn_index<PointRange, PointRange, uint32_t> index(build_params);

        std::cout << "🏗️  Building new Vamana graph (degree "
                  << graph_degree
                  << ", passes " << build_passes << ")...\n";
        index.build_index(graph, base_points, base_points, build_stats);
        start_point = index.get_start();

        GraphCacheMeta meta_to_write = expected_meta;
        meta_to_write.start_id = start_point;
        std::string graph_path = cache.graph_bin.string();
        graph.save(graph_path.data());
        write_graph_metadata(cache.graph_meta, meta_to_write);
        std::cout << "💾 Stored Vamana graph at " << cache.graph_bin << "\n";
      }

      pann::Graph<uint32_t> &graph = *graph_ptr;

      const long visit_limit = std::max<long>(beam_width,
                                              std::min<long>(visit_limit_env,
                                                             static_cast<long>(ds.nb)));
      const long degree_limit = std::max<long>(
          1, std::min<long>(degree_limit_env,
                            static_cast<long>(graph.max_degree())));

        pann::QueryParams query_params(static_cast<long>(final_candidate_cap),
                       beam_width,
                       visit_limit,
                       degree_limit,
                       rerank_factor,
                       batch_factor);

        // Allocate outside the timed region to reduce malloc noise.
        std::vector<float> fused_queries(ds.nq * ds.dim);
        std::vector<float> query_attrs_projected(ds.nq * ds.attr_fused_dim);
        std::vector<uint8_t> scratch_bitmap_u8;
        std::vector<float> raw_attr;
        std::vector<float> proj_attr(ds.attr_fused_dim);

        const auto query_phase_start = std::chrono::steady_clock::now();

        // Online query preprocessing: hashing/bitmap build (if sparse), PCA projection, and fusion.
        for (size_t qi = 0; qi < ds.nq; ++qi)
        {
          const uint8_t *bitmap_bytes = nullptr;
          build_query_attr_raw_float_and_bitmap_bytes(ds,
                                                      qi,
                                                      scratch_bitmap_u8,
                                                      raw_attr,
                                                      bitmap_bytes);
          (void)bitmap_bytes; // sparse path uses CSR bytes only for cache keys elsewhere

          project_query_attr_dense_pca(ds,
                                       raw_attr.data(),
                                       ds.attr_dim,
                                       proj_attr.data(),
                                       ds.attr_fused_dim);
          std::memcpy(query_attrs_projected.data() + qi * ds.attr_fused_dim,
                      proj_attr.data(),
                      sizeof(float) * ds.attr_fused_dim);

          const float *content = ds.xq.data() + qi * ds.dim;
          fuse_single_query(content,
                            proj_attr.data(),
                            ds.dim,
                            ds.attr_fused_dim,
                            static_cast<float>(alpha_used),
                            static_cast<float>(beta_used),
                            fused_queries.data() + qi * ds.dim);
        }

        FloatMatrixView query_view{fused_queries.data(), ds.nq, ds.dim};
        PointRange query_points(query_view, base_points.params);

        pann::stats<uint32_t> query_stats(ds.nq);
        const auto search_start = std::chrono::steady_clock::now();
        auto parlay_results = pann::searchAll(query_points,
                          graph,
                          base_points,
                          query_stats,
                          start_point,
                          query_params);
        const auto search_end = std::chrono::steady_clock::now();
        double search_only_seconds =
          std::chrono::duration<double>(search_end - search_start).count();

      const bool attr_approx = parse_env_flag("FUSEDANN_PARLAY_ATTR_APPROX", false);
      const double attr_approx_threshold = parse_env_double(
          "FUSEDANN_PARLAY_ATTR_APPROX_THRESHOLD", 1.0, -1.0);
      const long fallback_limit_env = parse_env_long(
          "FUSEDANN_PARLAY_FALLBACK_LIMIT",
          static_cast<long>(DEFAULT_FILTER_BRUTE_FORCE_LIMIT));
      const size_t fallback_limit = static_cast<size_t>(
          std::max<long>(1, fallback_limit_env));
      attr_score_weight = alpha_used * attr_score_scale;
      content_score_weight = beta_used * content_score_scale;
      const long prefilter_limit_env = parse_env_long(
          "FUSEDANN_PREFILTER_DIRECT_LIMIT",
          4000,
          0);
      const size_t prefilter_direct_limit = static_cast<size_t>(
          std::max<long>(0, prefilter_limit_env));

      std::vector<int32_t> ann_results(ds.nq * DEFAULT_K, -1);
      std::vector<uint8_t> visited(ds.nb, 0);
      std::vector<uint32_t> touched;
      touched.reserve(final_candidate_cap * 2);
      std::vector<Candidate> scored;
      scored.reserve(std::max(rerank_k, DEFAULT_K));
      std::vector<Candidate> filtered;
      filtered.reserve(rerank_k);

      size_t queries_with_results = 0;

      // ========================================================================
      // 2-STAGE RETRIEVAL PIPELINE (SPARSE ATTRIBUTES PATH):
      // Stage 1: Gather top rerank_k candidates by fused-space L2 distance
      // Stage 2: Apply hard attribute filter (query_tags ⊆ doc_tags)
      // Stage 3: Sort remaining by content_dist, select top DEFAULT_K
      // ========================================================================
      for (size_t qi = 0; qi < ds.nq; ++qi)
      {
        scored.clear();
        filtered.clear();
        touched.clear();

        const int64_t tag_begin = ds.q_attr_indptr.empty() ? 0 : ds.q_attr_indptr[qi];
        const int64_t tag_end = ds.q_attr_indptr.empty() ? 0 : ds.q_attr_indptr[qi + 1];
        const size_t tag_count = static_cast<size_t>(tag_end - tag_begin);
        const int32_t *query_tags = ds.q_attr_indices.data() + tag_begin;

        const float *query_vec = ds.xq.data() + qi * ds.dim;
        const float *query_attr = query_attrs_projected.data() + qi * ds.attr_fused_dim;
        const float *fused_query_vec = fused_queries.data() + qi * ds.dim;

        // =====================================================================
        // STAGE 1: Gather candidates from ANN, score by fused-space L2 distance
        // =====================================================================
        const auto &neighbors = parlay_results[qi];
        const size_t candidate_top = std::min(neighbors.size(), final_candidate_cap);

        for (size_t ki = 0; ki < candidate_top; ++ki)
        {
          const uint32_t doc = neighbors[ki];
          if (doc >= ds.nb)
            continue;
          if (visited[doc])
            continue;
          visited[doc] = 1;
          touched.push_back(doc);

          const float *base_vec = ds.xb.data() + static_cast<size_t>(doc) * ds.dim;
          const float *base_attr = ds.bitmaps_float.data() + static_cast<size_t>(doc) * ds.attr_fused_dim;
          const float *fused_base_vec = fused_base.data() + static_cast<size_t>(doc) * ds.dim;

          // Fused-space L2 distance (score for stage 1 ranking)
          const double fused_dist = l2_distance(fused_base_vec, fused_query_vec, ds.dim);
          // Content L2 distance (for stage 3 ranking)
          const double content_dist = l2_distance(base_vec, query_vec, ds.dim);
          // PCA-projected attribute distance (for diagnostics)
          const double attr_dist = l2_distance(base_attr, query_attr, ds.attr_fused_dim);

          // Hard attribute filter check (sparse: query_tags ⊆ doc_tags)
          const bool attr_filter_passed = doc_matches_sparse(ds, doc, query_tags, tag_count);

          Candidate cand;
          cand.score = fused_dist;
          cand.content_dist = content_dist;
          cand.attr_dist = attr_dist;
          cand.id = static_cast<int32_t>(doc);
          cand.attr_filter_passed = attr_filter_passed;
          scored.push_back(cand);
        }

        // Take top rerank_k by fused distance (stage 1)
        if (scored.size() > rerank_k)
        {
          std::partial_sort(scored.begin(), scored.begin() + static_cast<long>(rerank_k), scored.end(),
                            [](const Candidate &lhs, const Candidate &rhs)
                            { return lhs.score < rhs.score; });
          scored.resize(rerank_k);
        }
        else
        {
          std::sort(scored.begin(), scored.end(),
                    [](const Candidate &lhs, const Candidate &rhs)
                    { return lhs.score < rhs.score; });
        }

        // =====================================================================
        // STAGE 2: Apply hard attribute filter
        // =====================================================================
        for (const auto &cand : scored)
        {
          if (cand.attr_filter_passed)
          {
            filtered.push_back(cand);
          }
        }

        // =====================================================================
        // STAGE 3: Sort by content_dist ascending, select top DEFAULT_K
        // =====================================================================
        if (filtered.size() > DEFAULT_K)
        {
          std::partial_sort(filtered.begin(), filtered.begin() + DEFAULT_K, filtered.end(),
                            [](const Candidate &lhs, const Candidate &rhs)
                            { return lhs.content_dist < rhs.content_dist; });
          filtered.resize(DEFAULT_K);
        }
        else
        {
          std::sort(filtered.begin(), filtered.end(),
                    [](const Candidate &lhs, const Candidate &rhs)
                    { return lhs.content_dist < rhs.content_dist; });
        }

        if (!filtered.empty())
        {
          ++queries_with_results;
        }

        if (diagnostics_enabled)
        {
          all_query_candidates[qi] = scored; // store stage-1 candidates for diagnostics
        }
        capture_candidate_order(qi, scored);

        // Fill results (remaining slots are -1 if fewer than DEFAULT_K pass filter)
        for (size_t ki = 0; ki < DEFAULT_K; ++ki)
        {
          const int32_t chosen = (ki < filtered.size()) ? filtered[ki].id : -1;
          ann_results[qi * DEFAULT_K + ki] = chosen;
        }

        for (uint32_t doc : touched)
        {
          visited[doc] = 0;
        }
      }

      const auto query_phase_end = std::chrono::steady_clock::now();
      const double query_seconds = std::chrono::duration<double>(query_phase_end - query_phase_start).count();

      const auto recall_start = std::chrono::steady_clock::now();
      double recall = compute_recall(ann_results, fused_gt, ds.nq, DEFAULT_K, DEFAULT_K, ds.nb);
      const auto recall_end = std::chrono::steady_clock::now();
      const double recall_seconds = std::chrono::duration<double>(recall_end - recall_start).count();

      double qps = (query_seconds > 0.0)
           ? static_cast<double>(ds.nq) / query_seconds
           : 0.0;

      std::cout << std::fixed << std::setprecision(4)
                << "\n📈 FusedANN recall: " << recall << "\n"
                << std::setprecision(2)
                << "⚡ QPS: " << qps << " queries/s (query " << query_seconds
                << " s; recall " << recall_seconds
                << " s; graph/search " << search_only_seconds << " s)\n"
                << "🔎 Queries with results: " << queries_with_results << " / " << ds.nq << "\n"
                << "🎉 Done.\n";
      emit_diagnostics_for_queries(diagnostics,
                                   ds,
                                   ann_results,
                                   fused_gt,
                                   all_query_candidates,
                                   diagnostics_enabled ? &diagnostics_candidate_orders : nullptr,
                                   attr_score_weight,
                                   content_score_weight,
                                   use_fused_distance,
                                   use_fused_distance ? &fused_base : nullptr,
                                   use_fused_distance ? &fused_queries : nullptr,
                                   nullptr,
                                   0,
                                   nullptr,
                                   alpha,
                                   beta,
                                   alpha_mult,
                                   beta_mult,
                                   per_cluster_alpha_beta);
      return EXIT_SUCCESS;
    }

    log_stage("Preparing dense attribute pipeline...");
    double alpha = 0.0;
    double beta = 0.0;
    std::vector<int32_t> fused_gt;
    double search_only_seconds = 0.0;
    double total_query_seconds = 0.0;
    std::chrono::steady_clock::time_point query_phase_start;
    bool query_timer_started = false;
    size_t total_queries_processed = 0;
    std::vector<int32_t> base_partition_id;
    std::vector<int32_t> query_primary_partition_id;
    std::vector<bool> query_exact_lookup_succeeded;  // Track per-query exact bitmap lookup success
    std::vector<float> dense_fused_base_storage;
    std::vector<float> dense_fused_query_storage;
    const std::vector<float> *diag_fused_base_ptr = nullptr;
    const std::vector<float> *diag_fused_query_ptr = nullptr;
        {
      if (partition_k > 0)
      {
        base_partition_id.assign(ds.nb, -1);
        for (size_t i = 0; i < ds.nb; ++i)
        {
          const uint8_t *row = ds.bitmaps.data() + i * ds.attr_dim;
          base_partition_id[i] = bitmap_hash_partition_id(row, ds.attr_dim, partition_k);
        }

        query_primary_partition_id.assign(ds.nq, -1);
        std::vector<uint8_t> scratch_q_bitmap;
        std::vector<float> scratch_q_raw_float;
        const uint8_t *q_bitmap_bytes = nullptr;
        for (size_t qi = 0; qi < ds.nq; ++qi)
        {
          build_query_attr_raw_float_and_bitmap_bytes(ds,
                                                      qi,
                                                      scratch_q_bitmap,
                                                      scratch_q_raw_float,
                                                      q_bitmap_bytes);
          query_primary_partition_id[qi] = bitmap_hash_partition_id(q_bitmap_bytes, ds.attr_dim, partition_k);
        }

        std::cout << "🧩 Using deterministic bitmap-hash partitions (K=" << partition_k
                  << ") for diagnostics; dense search remains per-attribute-key for correctness.\n";
      }

      log_stage("Building dense attribute groups...");
      AttrGroupMap groups = build_attribute_groups(ds.bitmaps, ds.nb, ds.attr_dim);

      log_stage("Loading or estimating α/β heuristics (dense)...");
      if (!load_alpha_beta_or_throw(alpha, beta))
      {
        std::cout << "\n🧪 Estimating α/β heuristics (auto mode)...\n";
        AlphaBetaStats stats = auto_alpha_beta(ds, groups);
        alpha = stats.alpha;
        beta = stats.beta;
        write_alpha_beta(cache.alpha_beta, alpha, beta);
        std::cout << std::fixed << std::setprecision(4)
                  << "\n🧮 Auto α/β stats:\n"
                  << "   μ_max ≈ " << stats.mu_max
                  << " | Á_f ≈ " << stats.cluster_radius
                  << " | â_min ≈ " << stats.min_attr_distance << "\n"
                  << "   → α = " << stats.alpha
                  << ", β = " << stats.beta
                  << "   (cached at " << cache.alpha_beta << ")\n";
      }

      const double alpha_used = alpha * alpha_mult;
      const double beta_used = beta * beta_mult;
      std::cout << std::fixed << std::setprecision(4)
                << "🧮 α/β: real(α=" << alpha << ", β=" << beta
                << ") applied(α=" << alpha_used << ", β=" << beta_used << ")\n"
                << std::defaultfloat;

      log_stage("Fusing dense base vectors...");
      std::cout << "\n⚙️  Fusing base vectors...\n";
      auto &fused_base = dense_fused_base_storage;
      fused_base.resize(ds.nb * ds.dim);
      Transformer::SingleTransform(ds.xb.data(),
                                   ds.bitmaps_float.data(),
                                   ds.nb,
                                   ds.dim,
                                   ds.attr_fused_dim,
                                   static_cast<float>(alpha_used),
                                   static_cast<float>(beta_used),
                                   fused_base.data());

      log_stage("Loading or computing fused-space ground truth (dense)...");
      if (load_fused_groundtruth(cache.fused_gt, ds.nq, DEFAULT_K, ds.nb, fused_gt))
      {
        std::cout << "📦 Loaded fused-space ground truth from " << cache.fused_gt << "\n";
      }
      else
      {
        std::cout << "⏳ Computing attribute-filtered fused-space ground truth (per-attribute brute-force)...\n";
        fused_gt = compute_filtered_groundtruth(ds, groups, DEFAULT_K);
        write_fused_groundtruth(cache.fused_gt, fused_gt, ds.nq, DEFAULT_K);
        std::cout << "💾 Stored fused-space ground truth at " << cache.fused_gt << "\n";
      }

      AttrGroupMap query_groups;
      query_groups.reserve(groups.size());
      for (size_t qi = 0; qi < ds.nq; ++qi)
      {
        std::string key(reinterpret_cast<const char *>(ds.q_bitmaps.data() + qi * ds.attr_dim),
                        static_cast<long>(ds.attr_dim));
        query_groups[key].push_back(qi);
      }

      const long debug_doc_id = parse_env_long_any("FUSEDANN_DEBUG_DOC_ID", -1);
      if (debug_doc_id >= 0)
      {
        bool reported = false;
        for (const auto &entry : query_groups)
        {
          const AttrKey &key = entry.first;
          const auto base_it = groups.find(key);
          if (base_it == groups.end())
          {
            std::cout << "[debug] query attr key missing in base groups (queries="
                      << entry.second.size() << ")\n";
            continue;
          }
          const auto &base_ids = base_it->second;
          bool contains = std::find(base_ids.begin(), base_ids.end(),
                                    static_cast<size_t>(debug_doc_id)) != base_ids.end();
          std::cout << "[debug] attr group size=" << base_ids.size()
                    << " | queries sharing key=" << entry.second.size()
                    << " | contains doc " << debug_doc_id << "? "
                    << (contains ? "yes" : "no") << "\n";
          if (contains)
          {
            reported = true;
            break;
          }
        }
        if (!reported)
        {
          std::cout << "[debug] no query attr key contained doc " << debug_doc_id
                    << " in its base group\n";
        }
      }

      attr_score_weight = alpha_used * attr_score_scale;
      content_score_weight = beta_used * content_score_scale;
      diag_fused_base_ptr = &dense_fused_base_storage;
      diag_fused_query_ptr = &dense_fused_query_storage;

      if (diagnostics_enabled)
      {
        const size_t diag_attr_dim = ds.attr_fused_dim ? ds.attr_fused_dim : ds.attr_dim;
        dense_fused_base_storage.resize(ds.nb * ds.dim);
        dense_fused_query_storage.resize(ds.nq * ds.dim);
        Transformer::SingleTransform(ds.xb.data(),
                                     ds.bitmaps_float.data(),
                                     ds.nb,
                                     ds.dim,
                                     diag_attr_dim,
                                     static_cast<float>(alpha_used),
                                     static_cast<float>(beta_used),
                                     dense_fused_base_storage.data());
        Transformer::SingleTransform(ds.xq.data(),
                                     ds.q_bitmaps_float.data(),
                                     ds.nq,
                                     ds.dim,
                                     diag_attr_dim,
                                     static_cast<float>(alpha_used),
                                     static_cast<float>(beta_used),
                                     dense_fused_query_storage.data());
      }

      log_stage("Running ParlayANN search and scoring sparse candidates...");
      const bool attr_approx = parse_env_flag("FUSEDANN_PARLAY_ATTR_APPROX", false);
      const double attr_approx_threshold = parse_env_double(
          "FUSEDANN_PARLAY_ATTR_APPROX_THRESHOLD", 1.0, -1.0);

      size_t group_counter = 0;
      struct PartitionedIndex
      {
        std::unique_ptr<pann::Graph<uint32_t>> graph;
        std::vector<float> base_data;
        std::vector<uint32_t> id_map;
        uint32_t start_point = 0;
        size_t nb_group = 0;
      };
      std::unordered_map<AttrKey, PartitionedIndex> index_map;

      log_stage("Building partitioned indices per attribute group...");
      std::cout << "\n🏗️  Building partitioned indices for " << groups.size() << " groups...\n";

      for (const auto &entry : groups)
      {
        const AttrKey &key = entry.first;
        const auto &base_ids = entry.second;
        if (base_ids.empty())
          continue;

        const auto query_it = query_groups.find(key);
        if (query_it == query_groups.end())
          continue;

        ++group_counter;
        const size_t nb_group = base_ids.size();

        PartitionedIndex part;
        part.nb_group = nb_group;
        part.base_data.resize(nb_group * ds.dim);
        part.id_map.resize(nb_group);

        for (size_t i = 0; i < nb_group; ++i)
        {
          const size_t global_idx = base_ids[i];
          std::memcpy(part.base_data.data() + i * ds.dim,
                      fused_base.data() + global_idx * ds.dim,
                      sizeof(float) * ds.dim);
          part.id_map[i] = static_cast<uint32_t>(global_idx);
        }

        FloatMatrixView group_base_view{part.base_data.data(), nb_group, ds.dim};
        PointRange base_points(group_base_view);

        const uint32_t group_degree = static_cast<uint32_t>(
            std::min<size_t>(graph_degree, nb_group > 1 ? nb_group - 1 : nb_group));

        pann::BuildParams build_params_dense(static_cast<long>(group_degree),
                                             beam_width,
                                             PARLAY_VAMANA_ALPHA,
                                             build_passes);

        part.graph = std::make_unique<pann::Graph<uint32_t>>(group_degree, nb_group);
        pann::stats<uint32_t> build_stats(nb_group);
        pann::knn_index<PointRange, PointRange, uint32_t> index(build_params_dense);

        std::cout << "   • Group " << group_counter << ": " << nb_group << " vectors\n";
        index.build_index(*part.graph, base_points, base_points, build_stats);
        part.start_point = index.get_start();

        if (debug_doc_id >= 0)
        {
          auto it = std::find(part.id_map.begin(), part.id_map.end(),
                              static_cast<uint32_t>(debug_doc_id));
          if (it != part.id_map.end())
          {
            const size_t local_idx = static_cast<size_t>(std::distance(part.id_map.begin(), it));
            const float *orig = fused_base.data() + static_cast<size_t>(debug_doc_id) * ds.dim;
            const float *copied = part.base_data.data() + local_idx * ds.dim;
            double diff_norm = 0.0;
            for (size_t d = 0; d < ds.dim; ++d)
            {
              const double delta = static_cast<double>(orig[d]) - static_cast<double>(copied[d]);
              diff_norm += delta * delta;
            }
            std::cout << "[debug] copied fused_base for doc " << debug_doc_id
                      << " into group " << group_counter
                      << " | l2_copy_error=" << std::sqrt(diff_norm) << "\n";
          }
        }

        index_map.emplace(key, std::move(part));
      }

      // Query-time starts after indices are built (index/load time excluded).
      query_phase_start = std::chrono::steady_clock::now();
      query_timer_started = true;
      total_queries_processed = ds.nq;

      log_stage("Searching partitioned indices...");
      std::cout << "\n🔎 Searching queries over partitioned indices...\n";

      for (const auto &entry : query_groups)
      {
        const AttrKey &key = entry.first;
        const auto &query_ids = entry.second;
        if (query_ids.empty())
          continue;

        const auto index_it = index_map.find(key);
        if (index_it == index_map.end())
          continue;

        PartitionedIndex &part = index_it->second;
        const size_t nq_group = query_ids.size();

        std::vector<float> content_block(nq_group * ds.dim);
        std::vector<float> attr_projected_block(nq_group * ds.attr_fused_dim);
        std::vector<float> group_queries(nq_group * ds.dim);

        std::vector<uint8_t> scratch_bitmap_u8;
        std::vector<float> raw_attr;
        for (size_t i = 0; i < nq_group; ++i)
        {
          const size_t global_idx = query_ids[i];
          std::memcpy(content_block.data() + i * ds.dim,
                      ds.xq.data() + global_idx * ds.dim,
                      sizeof(float) * ds.dim);

          const uint8_t *bitmap_bytes = nullptr;
          build_query_attr_raw_float_and_bitmap_bytes(ds,
                                                      global_idx,
                                                      scratch_bitmap_u8,
                                                      raw_attr,
                                                      bitmap_bytes);
          (void)bitmap_bytes;
          project_query_attr_dense_pca(ds,
                                       raw_attr.data(),
                                       ds.attr_dim,
                                       attr_projected_block.data() + i * ds.attr_fused_dim,
                                       ds.attr_fused_dim);
        }

        Transformer::SingleTransform(content_block.data(),
                                     attr_projected_block.data(),
                                     nq_group,
                                     ds.dim,
                                     ds.attr_fused_dim,
                                     static_cast<float>(alpha_used),
                                     static_cast<float>(beta_used),
                                     group_queries.data());

        FloatMatrixView group_base_view{part.base_data.data(), part.nb_group, ds.dim};
        FloatMatrixView group_query_view{group_queries.data(), nq_group, ds.dim};

        PointRange base_points(group_base_view);
        PointRange query_points(group_query_view, base_points.params);

        const long visit_limit = std::max<long>(beam_width,
                                                std::min<long>(visit_limit_env,
                                                               static_cast<long>(part.nb_group)));
        const long degree_limit = std::max<long>(
            1, std::min<long>(degree_limit_env,
                              static_cast<long>(part.graph->max_degree())));

        pann::QueryParams query_params(static_cast<long>(final_candidate_cap),
                                       beam_width,
                                       visit_limit,
                                       degree_limit,
                                       rerank_factor,
                                       batch_factor);

        pann::stats<uint32_t> query_stats(nq_group);
        const auto search_start = std::chrono::steady_clock::now();
        auto parlay_results = pann::searchAll(query_points,
                                              *part.graph,
                                              base_points,
                                              query_stats,
                                              part.start_point,
                                              query_params);
        const auto search_end = std::chrono::steady_clock::now();
        const double group_seconds =
            std::chrono::duration<double>(search_end - search_start).count();

        search_only_seconds += group_seconds;

        // =====================================================================
        // 2-STAGE RETRIEVAL PIPELINE (DENSE ATTRIBUTES PATH - PARTITIONED):
        // Note: Groups are pre-filtered by exact bitmap match, so stage 2 filter
        // is implicitly satisfied for all candidates in the group.
        // Stage 1: Score by fused-space L2 distance
        // Stage 2: Hard attribute filter (implicit - same bitmap group)
        // Stage 3: Sort remaining by content_dist, select top DEFAULT_K
        // Parallel over queries: each writes a distinct all_query_candidates[gq].
        // =====================================================================
        const bool want_attr_diag = (diagnostics.mode != DiagnosticsMode::None);
        // Parallelize per-query scoring with parlay (NOT OpenMP — omp threads
        // contend with ParlayANN's spin-waiting pool and end up slower). Each
        // query writes its own all_query_candidates[global_q], so no shared state.
        parlay::parallel_for(0, nq_group, [&](size_t local_q)
        {
          const auto &neighbors = parlay_results[local_q];
          const size_t global_q = query_ids[local_q];
          const size_t candidate_top = std::min<size_t>(neighbors.size(), final_candidate_cap);

          const float *query_vec = ds.xq.data() + global_q * ds.dim;
          const float *query_attr = attr_projected_block.data() + local_q * ds.attr_fused_dim;
          const float *fused_query_vec = group_queries.data() + local_q * ds.dim;

          auto &dst = all_query_candidates[global_q];
          dst.reserve(dst.size() + candidate_top);
          for (size_t ki = 0; ki < candidate_top; ++ki)
          {
            const uint32_t local_id = neighbors[ki];
            if (local_id >= part.id_map.size())
              continue;

            const int32_t global_candidate = static_cast<int32_t>(part.id_map[local_id]);
            const float *base_vec = ds.xb.data() + static_cast<size_t>(global_candidate) * ds.dim;
            const float *fused_base_vec = part.base_data.data() + static_cast<size_t>(local_id) * ds.dim;

            Candidate cand;
            cand.score = l2_distance(fused_base_vec, fused_query_vec, ds.dim);      // stage-1 rank
            cand.content_dist = l2_distance(base_vec, query_vec, ds.dim);           // stage-3 rank
            cand.id = global_candidate;
            cand.attr_filter_passed = true; // implicit: same bitmap group
            if (want_attr_diag) // attribute distance is diagnostics-only — skip in hot path
              cand.attr_dist = l2_distance(ds.bitmaps_float.data() + static_cast<size_t>(global_candidate) * ds.attr_fused_dim,
                                           query_attr, ds.attr_fused_dim);
            dst.push_back(cand);
          }
        });
      }
    }

    log_stage("Reducing candidate lists and computing recall/QPS...");
    // ========================================================================
    // 2-STAGE RETRIEVAL PIPELINE - FINAL REDUCTION (DENSE PATH):
    // Stage 1: Take top rerank_k by fused distance (score)
    // Stage 2: Hard attribute filter (implicit for dense bitmap groups)
    // Stage 3: Sort by content_dist ascending, select top DEFAULT_K
    // ========================================================================
    std::vector<int32_t> ann_results(ds.nq * DEFAULT_K, -1);
    size_t queries_with_results = 0;
    // Per-query reduction is independent (distinct ann_results[qi]) — parallelize
    // with parlay (same pool as searchAll; OpenMP here contends with it and is
    // slower). capture_candidate_order uses a shared diagnostics buffer, so it runs
    // in a separate serial pass below only when diagnostics are enabled.
    parlay::parallel_for(0, ds.nq, [&](size_t qi)
    {
      auto &candidates = all_query_candidates[qi];
      if (candidates.empty())
        return;

      // Stage 1: cap to top rerank_k by fused-space distance (score). The sparse
      // centroid path already caps each probed graph, so it keeps all and filters.
      if (!sparse_centroid_policy && candidates.size() > rerank_k)
      {
        std::partial_sort(candidates.begin(), candidates.begin() + static_cast<long>(rerank_k), candidates.end(),
                          [](const Candidate &lhs, const Candidate &rhs)
                          { return lhs.score < rhs.score; });
        candidates.resize(rerank_k);
      }
      else
      {
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &lhs, const Candidate &rhs)
                  { return lhs.score < rhs.score; });
      }

      // Stage 2: hard attribute filter (implicit-true for dense bitmap groups;
      // real for the sparse centroid path).
      std::vector<Candidate> filtered;
      filtered.reserve(candidates.size());
      for (const auto &cand : candidates)
        if (cand.attr_filter_passed)
          filtered.push_back(cand);

      // Stage 3: content-distance rerank, select top DEFAULT_K.
      const size_t topk = std::min<size_t>(filtered.size(), static_cast<size_t>(DEFAULT_K));
      std::partial_sort(filtered.begin(), filtered.begin() + static_cast<long>(topk), filtered.end(),
                        [](const Candidate &lhs, const Candidate &rhs)
                        { return lhs.content_dist < rhs.content_dist; });

      for (size_t k = 0; k < DEFAULT_K; ++k)
        ann_results[qi * DEFAULT_K + k] = (k < filtered.size()) ? filtered[k].id : -1;
    });

    // Diagnostics candidate-order capture (serial; shared buffer) only when on.
    if (diagnostics_enabled)
      for (size_t qi = 0; qi < ds.nq; ++qi)
        capture_candidate_order(qi, all_query_candidates[qi]);

    for (size_t qi = 0; qi < ds.nq; ++qi)
      if (ann_results[qi * DEFAULT_K] >= 0)
        ++queries_with_results;

    if (query_timer_started)
    {
      const auto query_phase_end = std::chrono::steady_clock::now();
      total_query_seconds = std::chrono::duration<double>(query_phase_end - query_phase_start).count();
    }

    const auto recall_start = std::chrono::steady_clock::now();
    double recall = compute_recall(ann_results, fused_gt, ds.nq, DEFAULT_K, DEFAULT_K, ds.nb);
    const auto recall_end = std::chrono::steady_clock::now();
    const double recall_seconds = std::chrono::duration<double>(recall_end - recall_start).count();

    double qps = (total_query_seconds > 0.0)
                     ? static_cast<double>(ds.nq) / total_query_seconds
                     : 0.0;

    std::cout << std::fixed << std::setprecision(4)
              << "\n📈 FusedANN recall: " << recall << "\n"
              << std::setprecision(2)
              << "⚡ QPS: " << qps << " queries/s (query " << total_query_seconds
              << " s; recall " << recall_seconds
              << " s; graph/search " << search_only_seconds << " s)\n"
              << "🔎 Queries with results: " << queries_with_results << " / " << ds.nq << "\n"
              << "🎉 Done.\n";
    const std::vector<float> empty_floats;
    emit_diagnostics_for_queries(diagnostics,
                                 ds,
                                 ann_results,
                                 fused_gt,
                                 all_query_candidates,
                                 diagnostics_enabled ? &diagnostics_candidate_orders : nullptr,
                                 attr_score_weight,
                                 content_score_weight,
                                 use_fused_distance,
                                 diag_fused_base_ptr,
                                 diag_fused_query_ptr,
                                 (partition_k > 0 && base_partition_id.size() == ds.nb) ? &base_partition_id : nullptr,
                                 static_cast<int>(partition_k),
                                 (!query_primary_partition_id.empty()) ? &query_primary_partition_id : nullptr,
                                 alpha,
                                 beta,
                                 alpha_mult,
                                 beta_mult,
                                 partition_local_alpha_beta,
                                 (!query_exact_lookup_succeeded.empty()) ? &query_exact_lookup_succeeded : nullptr);
    return EXIT_SUCCESS;
  }
  catch (const std::exception &ex)
  {
    std::cerr << "🔥 Error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
