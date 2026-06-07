#include "fusedann_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <iomanip>
#include <iostream>
#include <limits>
#include <immintrin.h>

#ifndef _mm128_loadu_ps
#define _mm128_loadu_ps _mm_loadu_ps
#define _mm128_mul_ps _mm_mul_ps
#define _mm128_add_ps _mm_add_ps
#endif

#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/parameters.h>

#include <memory>

using namespace fusedann;

static constexpr unsigned DEFAULT_BUILD_ITERS = 8;
static constexpr unsigned DEFAULT_L = 200;
static constexpr unsigned DEFAULT_GRAPH_R = 200;
static constexpr unsigned DEFAULT_S = 10;
static constexpr unsigned DEFAULT_POOL_SIZE = 1200;

struct EfannaGraphIndex {
  std::unique_ptr<efanna2e::Index> initializer;
  std::unique_ptr<efanna2e::IndexGraph> graph;
};

static std::vector<int32_t> rerank_groundtruth(const Dataset &ds,
                                               const std::vector<float> &fused_base,
                                               const std::vector<float> &fused_queries,
                                               size_t k) {
  if (ds.groundtruth.empty() || ds.gt_k == 0)
    throw std::runtime_error("Provided ground truth missing; cannot derive fused-space recall.");
  if (ds.gt_k < k)
    throw std::runtime_error("Provided ground truth has fewer entries than requested k.");

  std::vector<int32_t> fused_ids(ds.nq * k, -1);
  std::vector<std::pair<float, int32_t>> scratch(ds.gt_k);

  for (size_t qi = 0; qi < ds.nq; ++qi) {
    const float *query_vec = fused_queries.data() + qi * ds.dim;
    const int32_t *gt_row = ds.groundtruth.data() + qi * ds.gt_k;

    for (size_t gi = 0; gi < ds.gt_k; ++gi) {
      int32_t idx = gt_row[gi];
      if (idx < 0 || static_cast<size_t>(idx) >= ds.nb) {
        scratch[gi] = {std::numeric_limits<float>::infinity(), -1};
        continue;
      }
      const float *base_vec = fused_base.data() + static_cast<size_t>(idx) * ds.dim;
      float dist = l2_distance_sq(query_vec, base_vec, ds.dim);
      scratch[gi] = {dist, idx};
    }

    auto middle = scratch.begin() + static_cast<std::ptrdiff_t>(k);
    std::partial_sort(scratch.begin(), middle, scratch.end(),
                      [](const auto &lhs, const auto &rhs) {
                        return lhs.first < rhs.first;
                      });

    for (size_t kk = 0; kk < k; ++kk) {
      fused_ids[qi * k + kk] = scratch[kk].second;
    }
  }

  return fused_ids;
}

static EfannaGraphIndex build_efanna_graph_index(const float *dataset,
                                                 size_t nb,
                                                 size_t dim,
                                                 unsigned graph_k,
                                                 unsigned graph_l,
                                                 unsigned graph_s,
                                                 unsigned graph_r,
                                                 unsigned graph_iters) {
  EfannaGraphIndex bundle;

  auto initializer = std::make_unique<efanna2e::IndexRandom>(dim, nb);
  efanna2e::Parameters init_params;
  initializer->Build(nb, dataset, init_params);

  auto graph = std::make_unique<efanna2e::IndexGraph>(
      dim, nb, efanna2e::L2, initializer.get());

  efanna2e::Parameters params;
  params.Set<unsigned>("K", graph_k);
  params.Set<unsigned>("L", graph_l);
  params.Set<unsigned>("S", graph_s);
  params.Set<unsigned>("R", graph_r);
  params.Set<unsigned>("iter", graph_iters);

  graph->Build(nb, dataset, params);

  bundle.initializer = std::move(initializer);
  bundle.graph = std::move(graph);
  return bundle;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <sift_base.fvecs> <sift_base_attrs.bvecs>"
                 " <sift_query.fvecs> <sift_query_attrs.bvecs>"
                 " <sift_groundtruth.ivecs>\n";
    return EXIT_FAILURE;
  }

  const std::string base_path = argv[1];
  const std::string base_attr_path = argv[2];
  const std::string query_path = argv[3];
  const std::string query_attr_path = argv[4];
  const std::string gt_path = argv[5];

  try {
    Dataset ds = load_dataset(base_path, base_attr_path,
                              query_path, query_attr_path,
                              gt_path);

    if (ds.attributes_sparse) {
      throw std::runtime_error("Sparse attribute datasets are not supported by the EFANNA backend.");
    }

    AttrGroupMap groups = build_attribute_groups(
        ds.bitmaps, ds.nb, ds.attr_dim);

    double alpha = 0.5;
    double beta = 3690.9809;

    if (AUTO_ALPHA_BETA) {
      std::cout << "\n🧪 Estimating α/β heuristics (auto mode)...\n";
      AlphaBetaStats stats = auto_alpha_beta(ds, groups);
      alpha = stats.alpha;
      beta = stats.beta;
      std::cout << std::fixed << std::setprecision(4)
                << "\n🧮 Auto α/β stats:\n"
                << "   μ_max ≈ " << stats.mu_max
                << " | Á_f ≈ " << stats.cluster_radius
                << " | â_min ≈ " << stats.min_attr_distance << "\n"
                << "   → α = " << stats.alpha
                << ", β = " << stats.beta
                << "   (computed in " << std::setprecision(2)
                << stats.elapsed << " s)\n";
    } else {
      std::cout << "ℹ️  Using manual α=" << alpha << ", β=" << beta << "\n";
    }

    double alpha_mult = 1.0;
    double beta_mult = 1.0;
    if (const char *raw = std::getenv("FUSEDANN_ALPHA_MULT"))
    {
      char *end = nullptr;
      errno = 0;
      const double parsed = std::strtod(raw, &end);
      if (end != raw && errno == 0 && std::isfinite(parsed) && parsed > 0.0)
        alpha_mult = parsed;
      else
        std::cerr << "⚠️  Ignoring invalid FUSEDANN_ALPHA_MULT (" << raw << ")\n";
    }
    if (const char *raw = std::getenv("FUSEDANN_BETA_MULT"))
    {
      char *end = nullptr;
      errno = 0;
      const double parsed = std::strtod(raw, &end);
      if (end != raw && errno == 0 && std::isfinite(parsed) && parsed > 0.0)
        beta_mult = parsed;
      else
        std::cerr << "⚠️  Ignoring invalid FUSEDANN_BETA_MULT (" << raw << ")\n";
    }

    const double alpha_used = alpha * alpha_mult;
    const double beta_used = beta * beta_mult;
    if (alpha_mult != 1.0 || beta_mult != 1.0)
    {
      std::cout << std::fixed << std::setprecision(4)
                << "🧪 Applying α/β multipliers: α×" << alpha_mult
                << ", β×" << beta_mult
                << " | applied(α=" << alpha_used << ", β=" << beta_used << ")\n"
                << std::defaultfloat;
    }

    std::cout << "\n⚙️  Fusing base vectors...\n";
    std::vector<float> fused_base(ds.nb * ds.dim);
    Transformer::SingleTransform(
        ds.xb.data(),
        ds.bitmaps_float.data(),
        ds.nb,
        ds.dim,
      ds.attr_fused_dim,
        static_cast<float>(alpha_used),
        static_cast<float>(beta_used),
        fused_base.data());

    std::cout << "⚙️  Fusing query vectors...\n";
    std::vector<float> fused_queries(ds.nq * ds.dim);
    Transformer::SingleTransform(
        ds.xq.data(),
        ds.q_bitmaps_float.data(),
        ds.nq,
        ds.dim,
      ds.attr_fused_dim,
        static_cast<float>(alpha_used),
        static_cast<float>(beta_used),
        fused_queries.data());

    std::cout << "\n🎯 Building EFANNA2e graph index (CPU)...\n";
    auto efanna_bundle = build_efanna_graph_index(
        fused_base.data(),
        ds.nb,
        ds.dim,
        DEFAULT_K,
        DEFAULT_L,
        DEFAULT_S,
        DEFAULT_GRAPH_R,
        DEFAULT_BUILD_ITERS);

    std::cout << "🔍 Running Scenario 1 (fused-space search)...\n";
    efanna2e::Parameters search_params;
    search_params.Set<unsigned>("L_search", DEFAULT_POOL_SIZE);

    std::vector<int32_t> ann_results(ds.nq * DEFAULT_K, -1);
    std::vector<unsigned> efanna_indices(DEFAULT_K);

    const auto search_start = std::chrono::steady_clock::now();
    for (size_t qi = 0; qi < ds.nq; ++qi) {
      const float *query_ptr = fused_queries.data() + qi * ds.dim;
      efanna_bundle.graph->Search(
          query_ptr,
          fused_base.data(),
          DEFAULT_K,
          search_params,
          efanna_indices.data());

      for (size_t ki = 0; ki < DEFAULT_K; ++ki) {
        ann_results[qi * DEFAULT_K + ki] =
            static_cast<int32_t>(efanna_indices[ki]);
      }
    }
    const auto search_end = std::chrono::steady_clock::now();
    const double search_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(search_end - search_start).count();

    std::vector<int32_t> gt;
    std::cout << "📘 Computing attribute-filtered ground truth (per-attribute brute-force)...\n";
    gt = compute_filtered_groundtruth(ds, groups, DEFAULT_K);

    double recall = compute_recall(ann_results, gt, ds.nq, DEFAULT_K, DEFAULT_K, ds.nb);
    double qps = (search_seconds > 0.0)
                     ? static_cast<double>(ds.nq) / search_seconds
                     : 0.0;
    std::cout << std::fixed << std::setprecision(4)
              << "\n📈 Scenario 1 recall: " << recall << "\n"
              << std::setprecision(2)
              << "⚡ QPS: " << qps << " queries/s (" << search_seconds << " s)\n";
    std::cout << "🎉 Done.\n";
  } catch (const std::exception &ex) {
    std::cerr << "🔥 Error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}