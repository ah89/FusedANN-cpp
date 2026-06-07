#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace fusedann
{

  // -----------------------------------------------------------------------------
  // Shared configuration defaults
  // -----------------------------------------------------------------------------
  inline constexpr size_t DEFAULT_K = 10;
  inline constexpr bool AUTO_ALPHA_BETA = true;
  inline constexpr size_t ALPHA_SAMPLE_SIZE = 100000;
  inline constexpr size_t ALPHA_SAMPLE_PER_ATTR = 10000;
  inline constexpr double ALPHA_PERCENTILE = 95.0;
  inline constexpr uint64_t ALPHA_RANDOM_SEED = 123;
  inline constexpr double MIN_ALPHA = 1.01;
  inline constexpr double MIN_BETA = 1.01;
  inline constexpr size_t PAIRWISE_BLOCK_SIZE = 100000;
  inline constexpr double ALPHA_EPS_FUSED = 0.5;
  inline constexpr double ALPHA_MARGIN = 1.0;
  inline constexpr size_t ALPHA_CROSS_SAMPLES = 15000;
  inline constexpr double ALPHA_ATTR_PERCENTILE = 5.0;
  inline constexpr size_t MAX_ATTR_CENTROID_SAMPLES = 4096;
  inline constexpr double DEFAULT_SPARSE_PCA_SAMPLE_RATIO = 0.25;
  inline constexpr size_t DEFAULT_SPARSE_PCA_SAMPLE_MIN = 4096;
  inline constexpr size_t DEFAULT_SPARSE_PCA_SAMPLE_MAX = 200000;
  static constexpr size_t FUSEDANN_PCA_TARGET_DIM = 500;

  inline double fusedann_env_double(const char *name,
                                    double fallback,
                                    double min_value,
                                    double max_value = std::numeric_limits<double>::infinity())
  {
    if (const char *raw = std::getenv(name))
    {
      char *end = nullptr;
      errno = 0;
      double parsed = std::strtod(raw, &end);
      if (end != raw && errno == 0 && std::isfinite(parsed) &&
          parsed >= min_value && parsed <= max_value)
      {
        return parsed;
      }
      std::cerr << "⚠️  Ignoring invalid value for " << name << " (" << raw << ")\n";
    }
    return fallback;
  }

  inline size_t fusedann_env_size_t(const char *name,
                                    size_t fallback,
                                    size_t min_value = 1)
  {
    if (const char *raw = std::getenv(name))
    {
      char *end = nullptr;
      errno = 0;
      unsigned long long parsed = std::strtoull(raw, &end, 10);
      if (end != raw && errno == 0 && parsed >= min_value)
      {
        return static_cast<size_t>(parsed);
      }
      std::cerr << "⚠️  Ignoring invalid value for " << name << " (" << raw << ")\n";
    }
    return fallback;
  }

  // Deterministic unit-norm embedding for a SINGLE attribute id. Used by the
  // per-attribute expansion path so the same tag always maps to the same vector
  // on both the base (doc,tag) side and the query side. Distinct tags map to
  // (quasi-)distinct directions; identical tags map to distance 0. This is what
  // makes "the attribute vector refer to one and only one attribute".
  inline void make_tag_embedding(int32_t tag, size_t dim, float *out)
  {
    uint64_t s = static_cast<uint64_t>(static_cast<uint32_t>(tag)) * 0x9E3779B97F4A7C15ull
               + 0xD1B54A32D192ED03ull;
    double norm = 0.0;
    for (size_t j = 0; j < dim; ++j)
    {
      // splitmix64 step
      s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ull;
      s ^= s >> 27; s *= 0x94D049BB133111EBull;
      s ^= s >> 31;
      const double v = (static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
      out[j] = static_cast<float>(v);
      norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm < 1e-12) norm = 1.0;
    const float inv = static_cast<float>(1.0 / norm);
    for (size_t j = 0; j < dim; ++j) out[j] *= inv;
  }

  // Group-local "offset one-hot" embedding for a single tag, used by the
  // tag-partitioned expansion: tags are split into `groups` (tag % groups);
  // within a group the tag's local id is (tag / groups), bucketed into `dim`.
  // The result is a (smaller) one-hot vector of dimension `dim`. Two tags in
  // the same group+bucket collide (harmless — the exact subset filter removes
  // non-matches); raising `groups` or `dim` reduces collisions.
  inline int tag_group_of(int32_t tag, int groups)
  {
    return static_cast<int>(static_cast<uint32_t>(tag) % static_cast<uint32_t>(groups));
  }

  inline void make_group_onehot(int32_t tag, int groups, size_t dim, float *out)
  {
    for (size_t j = 0; j < dim; ++j) out[j] = 0.0f;
    const uint32_t local = static_cast<uint32_t>(tag) / static_cast<uint32_t>(groups);
    out[static_cast<size_t>(local % static_cast<uint32_t>(dim))] = 1.0f;
  }

  // -----------------------------------------------------------------------------
  // Dataset containers
  // -----------------------------------------------------------------------------
  struct Dataset
  {
    size_t dim = 0;
    size_t attr_dim = 0;
    size_t attr_fused_dim = 0;
    size_t nb = 0;
    size_t nq = 0;
    size_t gt_k = 0;

    std::vector<float> xb;
    std::vector<float> xq;
    std::vector<uint8_t> bitmaps;
    std::vector<uint8_t> q_bitmaps;
    std::vector<float> bitmaps_float;
    std::vector<float> q_bitmaps_float;
    bool attributes_pca = false;
    bool sparse_geometric = false;

    // When dense attribute PCA is applied, we also store the PCA model so callers
    // can project query attributes online (inside the timed query loop) without
    // changing the underlying model used for base preprocessing.
    size_t attr_pca_input_dim = 0;
    size_t attr_pca_output_dim = 0;
    std::vector<float> attr_pca_mean;       // [attr_pca_input_dim]
    std::vector<float> attr_pca_components; // row-major [attr_pca_output_dim][attr_pca_input_dim]

    std::vector<int32_t> groundtruth;
    bool attributes_sparse = true;
    size_t attr_vocab = 0;
    std::vector<int64_t> attr_indptr;
    std::vector<int32_t> attr_indices;
    std::vector<int64_t> q_attr_indptr;
    std::vector<int32_t> q_attr_indices;
  };

  struct AlphaBetaStats
  {
    double mu_max = 0.0;
    double cluster_radius = 0.0;
    double min_attr_distance = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double elapsed = 0.0;
  };

  // -----------------------------------------------------------------------------
  // Binary file loaders (fvecs / bvecs / ivecs)
  // -----------------------------------------------------------------------------
  template <typename T>
  inline void read_or_throw(std::ifstream &in, T *dst, std::streamsize bytes)
  {
    if (!in.read(reinterpret_cast<char *>(dst), bytes))
    {
      throw std::runtime_error("Failed to read expected amount of data.");
    }
  }

  inline void load_fvecs(const std::string &path,
                         std::vector<float> &out,
                         size_t &rows,
                         size_t &dim)
  {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin)
    {
      throw std::runtime_error("Cannot open fvecs file: " + path);
    }

    const std::streamoff file_bytes = fin.tellg();
    fin.seekg(0, std::ios::beg);

    // Peek first header to learn the dimension.
    int32_t dim32 = 0;
    read_or_throw(fin, &dim32, sizeof(dim32));
    if (dim32 <= 0)
    {
      throw std::runtime_error("Invalid dimension in fvecs file: " + path);
    }
    dim = static_cast<size_t>(dim32);

    const size_t record_bytes = sizeof(int32_t) + sizeof(float) * dim;
    if (record_bytes == 0 || file_bytes % record_bytes != 0)
    {
      throw std::runtime_error("File size is not a multiple of record size for " + path);
    }

    rows = static_cast<size_t>(file_bytes / record_bytes);
    out.resize(rows * dim);

    fin.seekg(0, std::ios::beg); // rewind to start so we can read everything

    for (size_t r = 0; r < rows; ++r)
    {
      int32_t cur_dim = 0;
      read_or_throw(fin, &cur_dim, sizeof(cur_dim));
      if (cur_dim != dim32)
      {
        throw std::runtime_error("Inconsistent dimension in fvecs file: " + path);
      }
      read_or_throw(fin, out.data() + r * dim, sizeof(float) * dim);
    }
  }

  inline void load_bvecs(const std::string &path, std::vector<uint8_t> &out,
                         size_t &rows, size_t &dim)
  {
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
      throw std::runtime_error("Cannot open bvecs file: " + path);

    fin.seekg(0, std::ios::end);
    auto end = fin.tellg();
    fin.seekg(0, std::ios::beg);

    rows = 0;
    dim = 0;
    out.clear();

    while (fin.tellg() < end)
    {
      uint32_t d = 0;
      read_or_throw(fin, &d, sizeof(d));
      if (dim == 0)
        dim = static_cast<size_t>(d);
      else if (dim != static_cast<size_t>(d))
        throw std::runtime_error("Inconsistent dimension in bvecs file.");

      out.resize((rows + 1) * dim);
      read_or_throw(fin, out.data() + rows * dim, sizeof(uint8_t) * dim);
      ++rows;
    }
  }

  inline void load_ivecs(const std::string &path, std::vector<int32_t> &out,
                         size_t &rows, size_t &dim)
  {
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
      throw std::runtime_error("Cannot open ivecs file: " + path);

    fin.seekg(0, std::ios::end);
    auto end = fin.tellg();
    fin.seekg(0, std::ios::beg);

    rows = 0;
    dim = 0;
    out.clear();

    while (fin.tellg() < end)
    {
      int32_t d = 0;
      read_or_throw(fin, &d, sizeof(d));
      if (dim == 0)
        dim = static_cast<size_t>(d);
      else if (dim != static_cast<size_t>(d))
        throw std::runtime_error("Inconsistent dimension in ivecs file.");

      out.resize((rows + 1) * dim);
      read_or_throw(fin, out.data() + rows * dim, sizeof(int32_t) * dim);
      ++rows;
    }
  }

  inline std::string lowercase_extension(const std::string &path)
  {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos)
      return "";
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch)
                   { return static_cast<char>(std::tolower(ch)); });
    return ext;
  }

  inline void load_u8bin(const std::string &path, std::vector<float> &out,
                         size_t &rows, size_t &dim)
  {
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
      throw std::runtime_error("Cannot open u8bin file: " + path);

    uint32_t header[2] = {0, 0};
    read_or_throw(fin, header, sizeof(header));
    rows = static_cast<size_t>(header[0]);
    dim = static_cast<size_t>(header[1]);

    out.resize(rows * dim);
    std::vector<uint8_t> buffer(dim);
    for (size_t i = 0; i < rows; ++i)
    {
      read_or_throw(fin, buffer.data(), static_cast<std::streamsize>(dim));
      float *dst = out.data() + i * dim;
      for (size_t j = 0; j < dim; ++j)
      {
        dst[j] = static_cast<float>(buffer[j]) / 255.0f;
      }
    }
  }

  inline void load_spmat(const std::string &path,
                         std::vector<int64_t> &indptr,
                         std::vector<int32_t> &indices,
                         size_t &rows,
                         size_t &cols)
  {
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
      throw std::runtime_error("Cannot open spmat file: " + path);

    int64_t header[3] = {0, 0, 0};
    read_or_throw(fin, header, sizeof(header));
    rows = static_cast<size_t>(header[0]);
    cols = static_cast<size_t>(header[1]);
    const int64_t nnz = header[2];
    if (rows == 0 && nnz != 0)
      throw std::runtime_error("Invalid spmat header (rows=0 but nnz>0) for: " + path);

    indptr.resize(rows + 1);
    const std::streamsize indptr_bytes =
        static_cast<std::streamsize>(indptr.size()) * sizeof(int64_t);
    read_or_throw(fin, indptr.data(), indptr_bytes);

    indices.resize(static_cast<size_t>(nnz));
    const std::streamsize index_bytes =
        static_cast<std::streamsize>(indices.size()) * sizeof(int32_t);
    read_or_throw(fin, indices.data(), index_bytes);

    const std::streamoff value_bytes =
        static_cast<std::streamoff>(nnz) * static_cast<std::streamoff>(sizeof(float));
    if (!fin.seekg(value_bytes, std::ios::cur))
      throw std::runtime_error("Failed to skip values section in spmat: " + path);
  }

  inline void load_ibin(const std::string &path, std::vector<int32_t> &out,
                        size_t &rows, size_t &dim)
  {
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
      throw std::runtime_error("Cannot open ibin file: " + path);

    uint32_t header[2] = {0, 0};
    read_or_throw(fin, header, sizeof(header));
    rows = static_cast<size_t>(header[0]);
    dim = static_cast<size_t>(header[1]);

    out.resize(rows * dim);
    const std::streamsize index_bytes =
        static_cast<std::streamsize>(out.size()) * sizeof(int32_t);
    read_or_throw(fin, out.data(), index_bytes);

    const std::streamoff dist_bytes =
        static_cast<std::streamoff>(rows) * static_cast<std::streamoff>(dim) *
        static_cast<std::streamoff>(sizeof(float));
    if (!fin.seekg(dist_bytes, std::ios::cur))
      throw std::runtime_error("Failed to skip distance section in ibin: " + path);
  }

  inline void sort_sparse_rows(std::vector<int64_t> &indptr,
                               std::vector<int32_t> &indices)
  {
    if (indptr.empty())
      return;
    const size_t rows = indptr.size() - 1;
    for (size_t row = 0; row < rows; ++row)
    {
      const int64_t begin = indptr[row];
      const int64_t end = indptr[row + 1];
      if (begin > end)
        throw std::runtime_error("Sparse indptr must be non-decreasing");
      std::sort(indices.begin() + begin, indices.begin() + end);
    }
  }

  inline void spmat_to_dense_bitmaps(const std::vector<int64_t> &indptr,
                                     const std::vector<int32_t> &indices,
                                     size_t rows,
                                     size_t vocab,
                                     std::vector<uint8_t> &bitmaps,
                                     std::vector<float> &bitmaps_float)
  {
    if (indptr.size() != rows + 1)
      throw std::runtime_error("Sparse attribute indptr has invalid shape for dense materialization.");
    bitmaps.assign(rows * vocab, 0);
    for (size_t row = 0; row < rows; ++row)
    {
      const int64_t begin = indptr[row];
      const int64_t end = indptr[row + 1];
      if (begin > end)
        throw std::runtime_error("Sparse attribute indptr must be non-decreasing for dense materialization.");
      for (int64_t pos = begin; pos < end; ++pos)
      {
        const int32_t tag = indices[static_cast<size_t>(pos)];
        if (tag < 0 || static_cast<size_t>(tag) >= vocab)
          throw std::runtime_error("Sparse attribute tag out of range during dense materialization.");
        bitmaps[row * vocab + static_cast<size_t>(tag)] = 1;
      }
    }
    bitmaps_float.resize(bitmaps.size());
    for (size_t i = 0; i < bitmaps.size(); ++i)
      bitmaps_float[i] = static_cast<float>(bitmaps[i]);
  }

  inline void spmat_to_hashed_bitmaps(const std::vector<int64_t> &indptr,
                                       const std::vector<int32_t> &indices,
                                       size_t rows,
                                       size_t hash_dim,
                                       size_t vocab,
                                       std::vector<uint8_t> &bitmaps,
                                       std::vector<float> &bitmaps_float)
  {
    if (hash_dim == 0)
      throw std::runtime_error("Hash dimension for bitmap projection must be positive.");
    if (indptr.size() != rows + 1)
      throw std::runtime_error("Sparse attribute indptr has invalid shape for hashed materialization.");

    bitmaps.assign(rows * hash_dim, 0);
    bitmaps_float.assign(rows * hash_dim, 0.0f);

    for (size_t row = 0; row < rows; ++row)
    {
      const int64_t begin = indptr[row];
      const int64_t end = indptr[row + 1];
      if (begin > end)
        throw std::runtime_error("Sparse attribute indptr must be non-decreasing for hashed materialization.");
      for (int64_t pos = begin; pos < end; ++pos)
      {
        const int32_t tag = indices[static_cast<size_t>(pos)];
        if (tag < 0 || static_cast<size_t>(tag) >= vocab)
          throw std::runtime_error("Sparse attribute tag out of range during hashed materialization.");
        const size_t bucket = static_cast<size_t>(static_cast<uint32_t>(tag)) % hash_dim;
        bitmaps[row * hash_dim + bucket] = 1;
        bitmaps_float[row * hash_dim + bucket] = 1.0f;
      }
    }
  }

  inline size_t select_pca_target_dim(size_t attr_dim, size_t content_dim)
  {
    size_t env_target = fusedann_env_size_t(
        "FUSEDANN_PCA_TARGET_DIM",
        FUSEDANN_PCA_TARGET_DIM,
        1);
    if (env_target > 0)
    {
      if (env_target > attr_dim)
      {
        std::cerr << "⚠️  FUSEDANN_PCA_TARGET_DIM (" << env_target
                  << ") > attribute dimension (" << attr_dim
                  << "), clamping to " << attr_dim << "\n";
        return attr_dim;
      }
      return env_target;
    }
    if (content_dim == 0)
      return 0;
    if (attr_dim <= content_dim)
      return attr_dim;
    return content_dim;
  }

#ifdef _OPENMP
#include <omp.h>
#endif

  inline void apply_dense_attribute_pca(Dataset &ds, size_t target_override = 0)
  {
    constexpr size_t MAX_PCA_ITER = 30;
    constexpr double CONVERGENCE_TOL = 1e-6;
    constexpr size_t PCA_SAMPLE_ROWS = 4096;

    const size_t original_dim = ds.attr_dim;
    size_t target_dim = 0;
    if (target_override > 0)
      target_dim = std::min(target_override, original_dim);
    if (target_dim == 0)
      target_dim = select_pca_target_dim(original_dim, ds.dim);
    if (target_dim == 0 || target_dim >= original_dim)
      return;

    const size_t sample_rows = std::min(ds.nb, PCA_SAMPLE_ROWS);
    if (sample_rows == 0)
      return;

    std::vector<float> base_input = std::move(ds.bitmaps_float);
    std::vector<float> query_input = std::move(ds.q_bitmaps_float);

    std::vector<double> mean(original_dim, 0.0);
#ifdef _OPENMP
    {
      const int num_threads = omp_get_max_threads();
      std::vector<double> mean_thread(static_cast<size_t>(num_threads) * original_dim, 0.0);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        double *local = mean_thread.data() + static_cast<size_t>(tid) * original_dim;
#pragma omp for schedule(static)
        for (size_t row = 0; row < ds.nb; ++row)
        {
          const float *src = base_input.data() + row * original_dim;
          for (size_t j = 0; j < original_dim; ++j)
          {
            local[j] += static_cast<double>(src[j]);
          }
        }
      }
      for (int t = 0; t < num_threads; ++t)
      {
        const double *local = mean_thread.data() + static_cast<size_t>(t) * original_dim;
        for (size_t j = 0; j < original_dim; ++j)
        {
          mean[j] += local[j];
        }
      }
    }
#else
    for (size_t row = 0; row < ds.nb; ++row)
    {
      const float *src = base_input.data() + row * original_dim;
      for (size_t j = 0; j < original_dim; ++j)
      {
        mean[j] += static_cast<double>(src[j]);
      }
    }
#endif
    const double inv_rows = ds.nb ? 1.0 / static_cast<double>(ds.nb) : 0.0;
    for (size_t j = 0; j < original_dim; ++j)
    {
      mean[j] *= inv_rows;
    }

    std::vector<double> sample_data(static_cast<size_t>(sample_rows) * original_dim, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t s = 0; s < sample_rows; ++s)
    {
      const size_t idx = static_cast<size_t>((static_cast<unsigned long long>(s) * ds.nb) /
                                             static_cast<unsigned long long>(sample_rows));
      const float *src = base_input.data() + idx * original_dim;
      double *dst = sample_data.data() + s * original_dim;
      for (size_t j = 0; j < original_dim; ++j)
      {
        dst[j] = static_cast<double>(src[j]) - mean[j];
      }
    }

    const size_t effective_components = std::min({target_dim, sample_rows, original_dim});
    if (effective_components == 0)
    {
      ds.bitmaps_float = std::move(base_input);
      ds.q_bitmaps_float = std::move(query_input);
      ds.attr_fused_dim = ds.attr_dim;
      ds.attributes_pca = false;
      return;
    }

    std::vector<std::vector<double>> components;
    components.reserve(effective_components);

    std::mt19937_64 rng(1729);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> vec(original_dim, 0.0);
    std::vector<double> y(original_dim, 0.0);
    std::vector<double> projections(sample_rows, 0.0);

    for (size_t c = 0; c < effective_components; ++c)
    {
      for (size_t j = 0; j < original_dim; ++j)
      {
        vec[j] = dist(rng);
      }
      for (const auto &prev : components)
      {
        double proj = 0.0;
        for (size_t j = 0; j < original_dim; ++j)
        {
          proj += vec[j] * prev[j];
        }
        for (size_t j = 0; j < original_dim; ++j)
        {
          vec[j] -= proj * prev[j];
        }
      }
      double norm = 0.0;
      for (double v : vec)
        norm += v * v;
      norm = std::sqrt(norm);
      if (norm < 1e-12)
        break;
      for (size_t j = 0; j < original_dim; ++j)
      {
        vec[j] /= norm;
      }

      for (size_t iter = 0; iter < MAX_PCA_ITER; ++iter)
      {
        std::fill(projections.begin(), projections.end(), 0.0);
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
          for (size_t row = 0; row < sample_rows; ++row)
          {
            const double *row_ptr = sample_data.data() + row * original_dim;
            double dot = 0.0;
            for (size_t j = 0; j < original_dim; ++j)
            {
              dot += row_ptr[j] * vec[j];
            }
            projections[row] = dot;
          }
        }

        std::fill(y.begin(), y.end(), 0.0);
#ifdef _OPENMP
        {
          const int num_threads = omp_get_max_threads();
          std::vector<double> y_thread(static_cast<size_t>(num_threads) * original_dim, 0.0);
#pragma omp parallel
          {
            const int tid = omp_get_thread_num();
            double *local = y_thread.data() + static_cast<size_t>(tid) * original_dim;
#pragma omp for schedule(static)
            for (size_t row = 0; row < sample_rows; ++row)
            {
              const double *row_ptr = sample_data.data() + row * original_dim;
              const double coeff = projections[row];
              if (coeff == 0.0)
                continue;
              for (size_t j = 0; j < original_dim; ++j)
              {
                local[j] += row_ptr[j] * coeff;
              }
            }
          }
          for (int t = 0; t < num_threads; ++t)
          {
            const double *local = y_thread.data() + static_cast<size_t>(t) * original_dim;
            for (size_t j = 0; j < original_dim; ++j)
            {
              y[j] += local[j];
            }
          }
        }
#else
        for (size_t row = 0; row < sample_rows; ++row)
        {
          const double *row_ptr = sample_data.data() + row * original_dim;
          const double coeff = projections[row];
          for (size_t j = 0; j < original_dim; ++j)
          {
            y[j] += row_ptr[j] * coeff;
          }
        }
#endif

        for (const auto &prev : components)
        {
          double proj = 0.0;
          for (size_t j = 0; j < original_dim; ++j)
          {
            proj += y[j] * prev[j];
          }
          for (size_t j = 0; j < original_dim; ++j)
          {
            y[j] -= proj * prev[j];
          }
        }

        double y_norm = 0.0;
        for (double v : y)
          y_norm += v * v;
        y_norm = std::sqrt(y_norm);
        if (y_norm < 1e-12)
          break;

        double diff = 0.0;
        for (size_t j = 0; j < original_dim; ++j)
        {
          const double updated = y[j] / y_norm;
          diff += std::fabs(updated - vec[j]);
          vec[j] = updated;
        }
        if (diff < CONVERGENCE_TOL * original_dim)
          break;
      }

      components.push_back(vec);
    }

    size_t aligned_dim = components.size();
    while (aligned_dim > 0 && (ds.dim % aligned_dim) != 0)
    {
      components.pop_back();
      --aligned_dim;
    }

    const size_t final_dim = aligned_dim;
    if (final_dim == 0)
    {
      ds.bitmaps_float = std::move(base_input);
      ds.q_bitmaps_float = std::move(query_input);
      ds.attr_fused_dim = ds.attr_dim;
      ds.attributes_pca = false;

      ds.attr_pca_input_dim = 0;
      ds.attr_pca_output_dim = 0;
      ds.attr_pca_mean.clear();
      ds.attr_pca_components.clear();
      return;
    }

    // Persist PCA model for online query projection.
    ds.attr_pca_input_dim = original_dim;
    ds.attr_pca_output_dim = final_dim;
    ds.attr_pca_mean.resize(original_dim);
    for (size_t j = 0; j < original_dim; ++j)
    {
      ds.attr_pca_mean[j] = static_cast<float>(mean[j]);
    }
    ds.attr_pca_components.resize(final_dim * original_dim);
    for (size_t c = 0; c < final_dim; ++c)
    {
      const auto &comp = components[c];
      float *dst = ds.attr_pca_components.data() + c * original_dim;
      for (size_t j = 0; j < original_dim; ++j)
      {
        dst[j] = static_cast<float>(comp[j]);
      }
    }

    std::vector<float> transformed_base(ds.nb * final_dim, 0.0f);
    std::vector<float> transformed_query(ds.nq * final_dim, 0.0f);

#ifdef _OPENMP
#pragma omp parallel
    {
      std::vector<double> centered(original_dim, 0.0);
#pragma omp for schedule(static)
      for (size_t row = 0; row < ds.nb; ++row)
      {
        const float *src = base_input.data() + row * original_dim;
        for (size_t j = 0; j < original_dim; ++j)
        {
          centered[j] = static_cast<double>(src[j]) - mean[j];
        }
        for (size_t c = 0; c < final_dim; ++c)
        {
          const auto &comp = components[c];
          double dot = 0.0;
          for (size_t j = 0; j < original_dim; ++j)
          {
            dot += centered[j] * comp[j];
          }
          transformed_base[row * final_dim + c] = static_cast<float>(dot);
        }
      }

#pragma omp for schedule(static)
      for (size_t row = 0; row < ds.nq; ++row)
      {
        const float *src = query_input.data() + row * original_dim;
        for (size_t j = 0; j < original_dim; ++j)
        {
          centered[j] = static_cast<double>(src[j]) - mean[j];
        }
        for (size_t c = 0; c < final_dim; ++c)
        {
          const auto &comp = components[c];
          double dot = 0.0;
          for (size_t j = 0; j < original_dim; ++j)
          {
            dot += centered[j] * comp[j];
          }
          transformed_query[row * final_dim + c] = static_cast<float>(dot);
        }
      }
    }
#else
    {
      std::vector<double> centered(original_dim, 0.0);

      for (size_t row = 0; row < ds.nb; ++row)
      {
        const float *src = base_input.data() + row * original_dim;
        for (size_t j = 0; j < original_dim; ++j)
        {
          centered[j] = static_cast<double>(src[j]) - mean[j];
        }
        for (size_t c = 0; c < final_dim; ++c)
        {
          const auto &comp = components[c];
          double dot = 0.0;
          for (size_t j = 0; j < original_dim; ++j)
          {
            dot += centered[j] * comp[j];
          }
          transformed_base[row * final_dim + c] = static_cast<float>(dot);
        }
      }

      for (size_t row = 0; row < ds.nq; ++row)
      {
        const float *src = query_input.data() + row * original_dim;
        for (size_t j = 0; j < original_dim; ++j)
        {
          centered[j] = static_cast<double>(src[j]) - mean[j];
        }
        for (size_t c = 0; c < final_dim; ++c)
        {
          const auto &comp = components[c];
          double dot = 0.0;
          for (size_t j = 0; j < original_dim; ++j)
          {
            dot += centered[j] * comp[j];
          }
          transformed_query[row * final_dim + c] = static_cast<float>(dot);
        }
      }
    }
#endif

    ds.bitmaps_float = std::move(transformed_base);
    ds.q_bitmaps_float = std::move(transformed_query);
    ds.attr_fused_dim = final_dim;
    ds.attributes_pca = true;

    double total_sq_error = 0.0;
    double total_sq_norm = 0.0;
    for (size_t s = 0; s < sample_rows; ++s)
    {
      const double *vec = sample_data.data() + s * original_dim;
      double vec_norm_sq = 0.0;
      for (size_t j = 0; j < original_dim; ++j)
      {
        vec_norm_sq += vec[j] * vec[j];
      }
      double proj_norm_sq = 0.0;
      for (size_t c = 0; c < final_dim; ++c)
      {
        double dot = 0.0;
        for (size_t j = 0; j < original_dim; ++j)
        {
          dot += vec[j] * components[c][j];
        }
        proj_norm_sq += dot * dot;
      }
      total_sq_error += (vec_norm_sq - proj_norm_sq);
      total_sq_norm += vec_norm_sq;
    }
    double mse = sample_rows > 0 ? total_sq_error / static_cast<double>(sample_rows) : 0.0;
    double loss_pct = (total_sq_norm > 1e-9) ? (total_sq_error / total_sq_norm) * 100.0 : 0.0;

    std::cout << "   • Dense attributes: PCA reduction "
              << original_dim << " → " << final_dim
              << " using " << sample_rows << " samples\n"
              << "   • PCA Reconstruction MSE: " << mse
              << " (" << std::fixed << std::setprecision(2) << loss_pct << "%)\n";
  }
  void log_parallel_backend()
  {
  #ifdef _OPENMP
      std::cout << "OpenMP enabled (threads = " << omp_get_max_threads() << ")\n";
  #else
      std::cout << "OpenMP not available; running single-threaded\n";
  #endif
  }

  inline void apply_sparse_attribute_pca(Dataset &ds, size_t target_override = 0)
  {
    if (!ds.attributes_sparse)
      return;

    constexpr size_t MAX_PCA_ITER = 30;
    constexpr double CONVERGENCE_TOL = 1e-6;

    const size_t original_dim = ds.attr_vocab;
    if (original_dim == 0 || ds.nb == 0)
      return;

    size_t target_dim = target_override > 0
                            ? std::min(target_override, original_dim)
                            : select_pca_target_dim(original_dim, ds.dim);
    if (target_dim == 0)
      return;

    const double ratio = fusedann_env_double(
        "FUSEDANN_SPARSE_PCA_SAMPLE_RATIO",
        DEFAULT_SPARSE_PCA_SAMPLE_RATIO,
        0.0,
        1.0);
    const size_t min_rows = fusedann_env_size_t(
        "FUSEDANN_SPARSE_PCA_SAMPLE_MIN",
        DEFAULT_SPARSE_PCA_SAMPLE_MIN,
        1);
    const size_t max_rows = fusedann_env_size_t(
        "FUSEDANN_SPARSE_PCA_SAMPLE_MAX",
        DEFAULT_SPARSE_PCA_SAMPLE_MAX,
        1);

    size_t desired = static_cast<size_t>(std::ceil(static_cast<double>(ds.nb) * ratio));
    if (desired == 0 && ds.nb > 0)
      desired = 1;
    size_t sample_rows = std::max(desired, min_rows);
    sample_rows = std::min(sample_rows, ds.nb);
    sample_rows = std::min(sample_rows, max_rows);
    if (sample_rows == 0)
      return;

    std::vector<double> mean(original_dim, 0.0);
    if (ds.attr_indptr.size() != ds.nb + 1)
      throw std::runtime_error("Sparse attribute indptr has invalid shape.");

#ifdef _OPENMP
#pragma omp parallel
    {
      std::vector<double> local_mean(original_dim, 0.0);
#pragma omp for schedule(static)
      for (size_t doc = 0; doc < ds.nb; ++doc)
      {
        const int64_t begin = ds.attr_indptr[doc];
        const int64_t end = ds.attr_indptr[doc + 1];
        for (int64_t pos = begin; pos < end; ++pos)
        {
          const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
          if (tag < 0 || static_cast<size_t>(tag) >= original_dim)
            throw std::runtime_error("Sparse attribute tag out of range (base dataset).");
          local_mean[static_cast<size_t>(tag)] += 1.0;
        }
      }
#pragma omp critical
      {
        for (size_t j = 0; j < original_dim; ++j)
          mean[j] += local_mean[j];
      }
    }
#else
    for (size_t doc = 0; doc < ds.nb; ++doc)
    {
      const int64_t begin = ds.attr_indptr[doc];
      const int64_t end = ds.attr_indptr[doc + 1];
      for (int64_t pos = begin; pos < end; ++pos)
      {
        const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
        if (tag < 0 || static_cast<size_t>(tag) >= original_dim)
          throw std::runtime_error("Sparse attribute tag out of range (base dataset).");
        mean[static_cast<size_t>(tag)] += 1.0;
      }
    }
#endif

    const double inv_nb = 1.0 / static_cast<double>(ds.nb);
    for (double &m : mean)
      m *= inv_nb;

    std::vector<size_t> sample_ids;
    sample_ids.reserve(sample_rows);
    if (sample_rows == ds.nb)
    {
      sample_ids.resize(ds.nb);
      std::iota(sample_ids.begin(), sample_ids.end(), 0);
    }
    else
    {
      const double stride = static_cast<double>(ds.nb) /
                            static_cast<double>(sample_rows);
      double cursor = 0.0;
      while (sample_ids.size() < sample_rows)
      {
        size_t idx = static_cast<size_t>(cursor);
        if (idx >= ds.nb)
          idx = ds.nb - 1;
        if (sample_ids.empty() || sample_ids.back() != idx)
          sample_ids.push_back(idx);
        cursor += stride;
      }
    }
    sample_rows = sample_ids.size();
    if (sample_rows == 0)
      return;

    std::vector<std::vector<double>> components;
    components.reserve(target_dim);

    std::vector<double> direction(original_dim, 0.0);
    std::vector<double> cov_vec(original_dim, 0.0);
    std::vector<double> y(sample_rows, 0.0);

    std::mt19937_64 rng(2025);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (size_t comp_idx = 0; comp_idx < target_dim; ++comp_idx)
    {
      bool initialized = false;
      for (size_t attempt = 0; attempt < 5 && !initialized; ++attempt)
      {
        for (size_t j = 0; j < original_dim; ++j)
          direction[j] = dist(rng);
        for (const auto &prev : components)
        {
          double proj = 0.0;
          for (size_t j = 0; j < original_dim; ++j)
            proj += direction[j] * prev[j];
          for (size_t j = 0; j < original_dim; ++j)
            direction[j] -= proj * prev[j];
        }
        double norm = 0.0;
        for (double v : direction)
          norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-10)
        {
          for (double &v : direction)
            v /= norm;
          initialized = true;
        }
      }
      if (!initialized)
        break;

      for (size_t iter = 0; iter < MAX_PCA_ITER; ++iter)
      {
        double baseline = 0.0;
        for (size_t j = 0; j < original_dim; ++j)
          baseline -= mean[j] * direction[j];

        double total_y = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : total_y)
#endif
        for (size_t si = 0; si < sample_rows; ++si)
        {
          const size_t doc = sample_ids[si];
          const int64_t begin = ds.attr_indptr[doc];
          const int64_t end = ds.attr_indptr[doc + 1];
          double sum = baseline;
          for (int64_t pos = begin; pos < end; ++pos)
          {
            const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
            sum += direction[static_cast<size_t>(tag)];
          }
          y[si] = sum;
          total_y += sum;
        }

#ifdef _OPENMP
#pragma omp parallel
        {
          std::vector<double> local_cov(original_dim, 0.0);
#pragma omp for schedule(static)
          for (size_t si = 0; si < sample_rows; ++si)
          {
            const double yi = y[si];
            if (yi == 0.0)
              continue;
            const size_t doc = sample_ids[si];
            const int64_t begin = ds.attr_indptr[doc];
            const int64_t end = ds.attr_indptr[doc + 1];
            for (int64_t pos = begin; pos < end; ++pos)
            {
              const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
              local_cov[static_cast<size_t>(tag)] += yi;
            }
          }
#pragma omp critical
          {
            for (size_t j = 0; j < original_dim; ++j)
              cov_vec[j] += local_cov[j];
          }
        }
#else
        std::fill(cov_vec.begin(), cov_vec.end(), 0.0);
        for (size_t si = 0; si < sample_rows; ++si)
        {
          const double yi = y[si];
          if (yi == 0.0)
            continue;
          const size_t doc = sample_ids[si];
          const int64_t begin = ds.attr_indptr[doc];
          const int64_t end = ds.attr_indptr[doc + 1];
          for (int64_t pos = begin; pos < end; ++pos)
          {
            const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
            cov_vec[static_cast<size_t>(tag)] += yi;
          }
        }
#endif

        for (size_t j = 0; j < original_dim; ++j)
          cov_vec[j] -= mean[j] * total_y;

        for (const auto &prev : components)
        {
          double proj = 0.0;
          for (size_t j = 0; j < original_dim; ++j)
            proj += cov_vec[j] * prev[j];
          for (size_t j = 0; j < original_dim; ++j)
            cov_vec[j] -= proj * prev[j];
        }

        double norm = 0.0;
        for (double v : cov_vec)
          norm += v * v;
        norm = std::sqrt(norm);
        if (norm < 1e-12)
          break;

        double diff = 0.0;
        for (size_t j = 0; j < original_dim; ++j)
        {
          const double updated = cov_vec[j] / norm;
          diff += std::fabs(updated - direction[j]);
          direction[j] = updated;
        }
        if (diff < CONVERGENCE_TOL * original_dim)
          break;

#ifdef _OPENMP
#pragma omp single
        std::fill(cov_vec.begin(), cov_vec.end(), 0.0);
#else
        std::fill(cov_vec.begin(), cov_vec.end(), 0.0);
#endif
      }

      double final_norm = 0.0;
      for (double v : direction)
        final_norm += v * v;
      final_norm = std::sqrt(final_norm);
      if (final_norm < 1e-10)
        break;

      components.emplace_back(direction);
    }

    size_t aligned_dim = components.size();
    while (aligned_dim > 0 && (ds.dim % aligned_dim) != 0)
    {
      components.pop_back();
      --aligned_dim;
    }

    const size_t final_dim = aligned_dim;
    if (final_dim == 0)
    {
      ds.attr_fused_dim = 0;
      ds.attributes_pca = false;
      ds.attr_pca_input_dim = 0;
      ds.attr_pca_output_dim = 0;
      ds.attr_pca_mean.clear();
      ds.attr_pca_components.clear();
      return;
    }

    ds.attr_pca_input_dim = original_dim;
    ds.attr_pca_output_dim = final_dim;
    ds.attr_pca_mean.resize(original_dim);
    for (size_t j = 0; j < original_dim; ++j)
      ds.attr_pca_mean[j] = static_cast<float>(mean[j]);
    ds.attr_pca_components.resize(final_dim * original_dim);
    for (size_t c = 0; c < final_dim; ++c)
    {
      float *dst = ds.attr_pca_components.data() + c * original_dim;
      for (size_t j = 0; j < original_dim; ++j)
        dst[j] = static_cast<float>(components[c][j]);
    }

    std::vector<double> baseline(final_dim, 0.0);
    for (size_t c = 0; c < final_dim; ++c)
    {
      double sum = 0.0;
      const auto &comp = components[c];
      for (size_t j = 0; j < original_dim; ++j)
        sum += mean[j] * comp[j];
      baseline[c] = -sum;
    }

    ds.bitmaps_float.assign(ds.nb * final_dim, 0.0f);
  #ifdef _OPENMP
  #pragma omp parallel
      {
        std::vector<double> accum(final_dim, 0.0);
  #pragma omp for schedule(static)
        for (size_t doc = 0; doc < ds.nb; ++doc)
        {
          std::copy(baseline.begin(), baseline.end(), accum.begin());
          const int64_t begin = ds.attr_indptr[doc];
          const int64_t end = ds.attr_indptr[doc + 1];
          for (int64_t pos = begin; pos < end; ++pos)
          {
            const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
            for (size_t c = 0; c < final_dim; ++c)
              accum[c] += components[c][static_cast<size_t>(tag)];
          }
          float *dst = ds.bitmaps_float.data() + doc * final_dim;
          for (size_t c = 0; c < final_dim; ++c)
            dst[c] = static_cast<float>(accum[c]);
        }
      }
  #else
      {
        std::vector<double> accum(final_dim, 0.0);
        for (size_t doc = 0; doc < ds.nb; ++doc)
        {
          std::copy(baseline.begin(), baseline.end(), accum.begin());
          const int64_t begin = ds.attr_indptr[doc];
          const int64_t end = ds.attr_indptr[doc + 1];
          for (int64_t pos = begin; pos < end; ++pos)
          {
            const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
            for (size_t c = 0; c < final_dim; ++c)
              accum[c] += components[c][static_cast<size_t>(tag)];
          }
          float *dst = ds.bitmaps_float.data() + doc * final_dim;
          for (size_t c = 0; c < final_dim; ++c)
            dst[c] = static_cast<float>(accum[c]);
        }
      }
  #endif

      ds.q_bitmaps_float.assign(ds.nq * final_dim, 0.0f);
  #ifdef _OPENMP
  #pragma omp parallel
      {
        std::vector<double> accum(final_dim, 0.0);
  #pragma omp for schedule(static)
        for (size_t qi = 0; qi < ds.nq; ++qi)
        {
          std::copy(baseline.begin(), baseline.end(), accum.begin());
          const int64_t begin = ds.q_attr_indptr[qi];
          const int64_t end = ds.q_attr_indptr[qi + 1];
          for (int64_t pos = begin; pos < end; ++pos)
          {
            const int32_t tag = ds.q_attr_indices[static_cast<size_t>(pos)];
            if (tag < 0 || static_cast<size_t>(tag) >= original_dim)
              throw std::runtime_error("Sparse attribute tag out of range (query dataset).");
            for (size_t c = 0; c < final_dim; ++c)
              accum[c] += components[c][static_cast<size_t>(tag)];
          }
          float *dst = ds.q_bitmaps_float.data() + qi * final_dim;
          for (size_t c = 0; c < final_dim; ++c)
            dst[c] = static_cast<float>(accum[c]);
        }
      }
  #else
      {
        std::vector<double> accum(final_dim, 0.0);
        for (size_t qi = 0; qi < ds.nq; ++qi)
        {
          std::copy(baseline.begin(), baseline.end(), accum.begin());
          const int64_t begin = ds.q_attr_indptr[qi];
          const int64_t end = ds.q_attr_indptr[qi + 1];
          for (int64_t pos = begin; pos < end; ++pos)
          {
            const int32_t tag = ds.q_attr_indices[static_cast<size_t>(pos)];
            if (tag < 0 || static_cast<size_t>(tag) >= original_dim)
              throw std::runtime_error("Sparse attribute tag out of range (query dataset).");
            for (size_t c = 0; c < final_dim; ++c)
              accum[c] += components[c][static_cast<size_t>(tag)];
          }
          float *dst = ds.q_bitmaps_float.data() + qi * final_dim;
          for (size_t c = 0; c < final_dim; ++c)
            dst[c] = static_cast<float>(accum[c]);
        }
      }
  #endif

    ds.attr_fused_dim = final_dim;
    ds.attributes_pca = true;

    double total_sq_error = 0.0;
    double total_sq_norm = 0.0;
    double mean_norm_sq = 0.0;
    for (double m : mean)
      mean_norm_sq += m * m;

    std::vector<double> mean_dot_comp(final_dim, 0.0);
    for (size_t c = 0; c < final_dim; ++c)
    {
      for (size_t j = 0; j < original_dim; ++j)
        mean_dot_comp[c] += mean[j] * components[c][j];
    }

    for (size_t s = 0; s < sample_rows; ++s)
    {
      const size_t doc = sample_ids[s];
      const int64_t begin = ds.attr_indptr[doc];
      const int64_t end = ds.attr_indptr[doc + 1];

      double x_sparse_norm_sq = static_cast<double>(end - begin);
      double x_sparse_dot_mean = 0.0;
      for (int64_t pos = begin; pos < end; ++pos)
      {
        const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
        x_sparse_dot_mean += mean[static_cast<size_t>(tag)];
      }
      double vec_norm_sq = x_sparse_norm_sq + mean_norm_sq - 2.0 * x_sparse_dot_mean;

      double proj_norm_sq = 0.0;
      for (size_t c = 0; c < final_dim; ++c)
      {
        double x_sparse_dot_comp = 0.0;
        for (int64_t pos = begin; pos < end; ++pos)
        {
          const int32_t tag = ds.attr_indices[static_cast<size_t>(pos)];
          x_sparse_dot_comp += components[c][static_cast<size_t>(tag)];
        }
        double w = x_sparse_dot_comp - mean_dot_comp[c];
        proj_norm_sq += w * w;
      }
      total_sq_error += (vec_norm_sq - proj_norm_sq);
      total_sq_norm += vec_norm_sq;
    }
    double mse = sample_rows > 0 ? total_sq_error / static_cast<double>(sample_rows) : 0.0;
    double loss_pct = (total_sq_norm > 1e-9) ? (total_sq_error / total_sq_norm) * 100.0 : 0.0;

    const double sample_percent = (ds.nb == 0)
                                      ? 0.0
                                      : (100.0 * static_cast<double>(sample_rows) /
                                         static_cast<double>(ds.nb));
    std::ostringstream pct_stream;
    pct_stream << std::fixed << std::setprecision(2) << sample_percent;
    std::cout << "   • Sparse attributes: PCA reduction "
              << original_dim << " → " << final_dim
              << " using " << sample_rows
              << " samples (~" << pct_stream.str() << "% of base)\n"
              << "   • PCA Reconstruction MSE: " << mse
              << " (" << std::fixed << std::setprecision(2) << loss_pct << "%)\n";
  }
  // -----------------------------------------------------------------------------
  // Dataset loader wrapper
  // -----------------------------------------------------------------------------
  inline Dataset load_dataset_old(const std::string &base_fvecs,
                                  const std::string &base_attrs_bvecs,
                                  const std::string &query_fvecs,
                                  const std::string &query_attrs_bvecs,
                                  const std::string &gt_ivecs_path)
  {
    Dataset ds;
    size_t rows, dim;

    const std::string base_ext = lowercase_extension(base_fvecs);
    if (base_ext == ".fvecs")
    {
      load_fvecs(base_fvecs, ds.xb, rows, dim);
    }
    else if (base_ext == ".u8bin")
    {
      load_u8bin(base_fvecs, ds.xb, rows, dim);
    }
    else
    {
      throw std::runtime_error("Unsupported base vector format: " + base_fvecs);
    }
    ds.nb = rows;
    ds.dim = dim;

    const std::string attr_ext = lowercase_extension(base_attrs_bvecs);
    if (attr_ext == ".bvecs")
    {
      size_t attr_rows = 0;
      size_t attr_dim = 0;
      load_bvecs(base_attrs_bvecs, ds.bitmaps, attr_rows, attr_dim);
      if (attr_rows != ds.nb)
        throw std::runtime_error("Base attribute rows do not match base vectors.");
      ds.attributes_sparse = false;
      ds.attr_dim = attr_dim;
      ds.attr_fused_dim = attr_dim;
      ds.attributes_pca = false;
      ds.bitmaps_float.resize(ds.bitmaps.size());
      for (size_t i = 0; i < ds.bitmaps.size(); ++i)
      {
        ds.bitmaps_float[i] = static_cast<float>(ds.bitmaps[i]);
      }
    }
    else if (attr_ext == ".spmat")
    {
      size_t attr_rows = 0;
      size_t attr_vocab = 0;
      load_spmat(base_attrs_bvecs, ds.attr_indptr, ds.attr_indices, attr_rows, attr_vocab);
      if (attr_rows != ds.nb)
        throw std::runtime_error("Base sparse attribute rows do not match base vectors.");

      std::cout << "    ↳ sorting sparse rows …\n";
      sort_sparse_rows(ds.attr_indptr, ds.attr_indices);

      const size_t default_hash_dim = std::min<size_t>(attr_vocab, static_cast<size_t>(256));
      const size_t hash_dim = fusedann_env_size_t(
          "FUSEDANN_BITMAP_HASH_DIM",
          default_hash_dim,
          1);

      const size_t dense_dim = std::min(hash_dim, attr_vocab);
      std::cout << "    ↳ hashing sparse attrs into dense bitmaps (dim="
                << dense_dim << ", vocab=" << attr_vocab << ")…\n";

      ds.attributes_sparse = true; // preserve sparse indices for filtering
      ds.attr_vocab = attr_vocab;
      ds.attr_dim = dense_dim;
      ds.attr_fused_dim = dense_dim;
      ds.attributes_pca = false;

      spmat_to_hashed_bitmaps(ds.attr_indptr,
                              ds.attr_indices,
                              ds.nb,
                              dense_dim,
                              ds.attr_vocab,
                              ds.bitmaps,
                              ds.bitmaps_float);
    }
    else
    {
      throw std::runtime_error("Unsupported base attribute format: " + base_attrs_bvecs);
    }

    const std::string query_ext = lowercase_extension(query_fvecs);
    if (query_ext == ".fvecs")
    {
      load_fvecs(query_fvecs, ds.xq, rows, dim);
    }
    else if (query_ext == ".u8bin")
    {
      load_u8bin(query_fvecs, ds.xq, rows, dim);
    }
    else
    {
      throw std::runtime_error("Unsupported query vector format: " + query_fvecs);
    }
    ds.nq = rows;
    if (ds.dim != dim)
      throw std::runtime_error("Query dimension mismatch with base vectors.");

    const std::string query_attr_ext = lowercase_extension(query_attrs_bvecs);
    if (ds.attributes_sparse)
    {
      // No longer expected; dense materialization above sets attributes_sparse=false.
      throw std::runtime_error("Sparse attributes should have been materialized to dense bitmaps.");
    }
    else
    {
      if (query_attr_ext == ".bvecs")
      {
        size_t attr_rows = 0;
        size_t attr_dim = 0;
        load_bvecs(query_attrs_bvecs, ds.q_bitmaps, attr_rows, attr_dim);
        if (attr_rows != ds.nq || attr_dim != ds.attr_dim)
          throw std::runtime_error(
              "Query attribute data mismatch with query vectors or base attributes.");
        ds.q_bitmaps_float.resize(ds.q_bitmaps.size());
        for (size_t i = 0; i < ds.q_bitmaps.size(); ++i)
        {
          ds.q_bitmaps_float[i] = static_cast<float>(ds.q_bitmaps[i]);
        }
      }
      else if (query_attr_ext == ".spmat")
      {
        size_t attr_rows = 0;
        size_t attr_vocab = 0;
        load_spmat(query_attrs_bvecs, ds.q_attr_indptr, ds.q_attr_indices, attr_rows, attr_vocab);
        if (attr_rows != ds.nq)
          throw std::runtime_error("Query sparse attribute rows do not match queries.");
        if (attr_vocab != ds.attr_vocab)
          throw std::runtime_error("Query sparse attribute vocab mismatch with base attributes.");
        std::cout << "    ↳ materializing dense query bitmaps …\n";
        spmat_to_dense_bitmaps(ds.q_attr_indptr,
                               ds.q_attr_indices,
                               ds.nq,
                               ds.attr_vocab,
                               ds.q_bitmaps,
                               ds.q_bitmaps_float);
        ds.q_attr_indptr.clear();
        ds.q_attr_indices.clear();
      }
      else
      {
        throw std::runtime_error("Unsupported query attribute format: " + query_attrs_bvecs);
      }

      // Always run PCA on dense bitmaps to a target dimension (default 10, overridable).
      const size_t bitmap_pca_dim = fusedann_env_size_t(
          "FUSEDANN_BITMAP_PCA_DIM",
          std::min<size_t>(ds.attr_dim, static_cast<size_t>(10)),
          1);
      apply_dense_attribute_pca(ds, bitmap_pca_dim);
    }

    if (!gt_ivecs_path.empty())
    {
      std::ifstream gt_probe(gt_ivecs_path, std::ios::binary);
      if (!gt_probe)
      {
        std::cout << "📭 Ground truth file missing at " << gt_ivecs_path
                  << " (will proceed without cached GT)\n";
      }
      else
      {
        gt_probe.close();
        const std::string gt_ext = lowercase_extension(gt_ivecs_path);
        size_t gt_rows = 0;
        size_t k = 0;
        if (gt_ext == ".ivecs")
        {
          load_ivecs(gt_ivecs_path, ds.groundtruth, gt_rows, k);
        }
        else if (gt_ext == ".ibin")
        {
          load_ibin(gt_ivecs_path, ds.groundtruth, gt_rows, k);
        }
        else
        {
          throw std::runtime_error("Unsupported ground truth format: " + gt_ivecs_path);
        }
        if (gt_rows != ds.nq)
          throw std::runtime_error("Ground truth row count does not match queries.");
        ds.gt_k = k;
        std::cout << "📦 Loaded dataset ground truth from " << gt_ivecs_path
                  << " (k=" << ds.gt_k << ")\n";
      }
    }

    std::cout << "✅ Loaded dataset:\n"
              << "   • Base vectors:  " << ds.nb << " × " << ds.dim << "\n"
              << "   • Query vectors: " << ds.nq << " × " << ds.dim << "\n";
    if (ds.attributes_sparse)
    {
      std::cout << "   • Attribute vocab: " << ds.attr_vocab << " (sparse CSR)\n";
      if (ds.attr_fused_dim > 0)
      {
        std::cout << "   • Attribute PCA dim: " << ds.attr_fused_dim << "\n";
      }
    }
    else
    {
      std::cout << "   • Attribute dim: " << ds.attr_dim;
      if (ds.attributes_pca)
      {
        std::cout << " (PCA → " << ds.attr_fused_dim << ")";
      }
      std::cout << "\n";
    }
    return ds;
  }

  inline Dataset load_dataset(const std::string &base_fvecs,
                              const std::string &base_attrs_bvecs,
                              const std::string &query_fvecs,
                              const std::string &query_attrs_bvecs,
                              const std::string &gt_ivecs_path)
  {
    Dataset ds;
    size_t rows = 0, dim = 0;
    const auto env_flag_enabled = [](const char *name) {
      const char *raw = std::getenv(name);
      if (raw == nullptr)
        return false;
      const std::string value(raw);
      return value == "1" || value == "true" || value == "TRUE" ||
             value == "yes" || value == "YES" || value == "on" || value == "ON";
    };
    const auto positive_env_long = [](const char *name) {
      const char *raw = std::getenv(name);
      if (raw == nullptr)
        return false;
      char *end = nullptr;
      errno = 0;
      const long value = std::strtol(raw, &end, 10);
      return end != raw && errno == 0 && value > 0;
    };
    const bool sparse_geometric_requested =
        env_flag_enabled("FUSEDANN_SPARSE_GEOMETRIC") ||
        positive_env_long("FUSEDANN_PARTITION_K");

    std::cout << "[1/9] Loading base vectors from " << base_fvecs << " …\n";
    const std::string base_ext = lowercase_extension(base_fvecs);
    if (base_ext == ".fvecs")
    {
      load_fvecs(base_fvecs, ds.xb, rows, dim);
    }
    else if (base_ext == ".u8bin")
    {
      load_u8bin(base_fvecs, ds.xb, rows, dim);
    }
    else
    {
      throw std::runtime_error("Unsupported base vector format: " + base_fvecs);
    }
    ds.nb = rows;
    ds.dim = dim;
    std::cout << "    ✔ base vectors: " << ds.nb << " × " << ds.dim << "\n";

    std::cout << "[2/9] Loading base attributes from " << base_attrs_bvecs << " …\n";
    const std::string attr_ext = lowercase_extension(base_attrs_bvecs);
    if (attr_ext == ".bvecs")
    {
      size_t attr_rows = 0;
      size_t attr_dim = 0;
      load_bvecs(base_attrs_bvecs, ds.bitmaps, attr_rows, attr_dim);
      if (attr_rows != ds.nb)
        throw std::runtime_error("Base attribute rows do not match base vectors.");
      ds.attributes_sparse = false;
      ds.attr_vocab = attr_dim;
      ds.attr_dim = attr_dim;
      ds.attr_fused_dim = attr_dim;
      ds.attributes_pca = false;
      ds.bitmaps_float.resize(ds.bitmaps.size());
      std::cout << "    ↳ converting dense attrs uint8 → float …\n";
      for (size_t i = 0; i < ds.bitmaps.size(); ++i)
      {
        ds.bitmaps_float[i] = static_cast<float>(ds.bitmaps[i]);
      }
    }
    else if (attr_ext == ".spmat")
    {
      size_t attr_rows = 0;
      size_t attr_vocab = 0;
      load_spmat(base_attrs_bvecs, ds.attr_indptr, ds.attr_indices, attr_rows, attr_vocab);
      if (attr_rows != ds.nb)
        throw std::runtime_error("Base sparse attribute rows do not match base vectors.");

      std::cout << "    ↳ sorting sparse rows …\n";
      sort_sparse_rows(ds.attr_indptr, ds.attr_indices);

      ds.attributes_sparse = true; // keep CSR for filtering paths
      ds.attr_vocab = attr_vocab;
      ds.attributes_pca = false;
      ds.sparse_geometric = sparse_geometric_requested;

      if (ds.sparse_geometric)
      {
        ds.attr_dim = attr_vocab;
        ds.attr_fused_dim = attr_vocab;
        std::cout << "    ↳ preserving sparse one-hot geometry (vocab="
                  << attr_vocab << "); no modulo hashing\n";
      }
      else
      {
        const size_t default_hash_dim = std::min<size_t>(attr_vocab, static_cast<size_t>(256));
        const size_t hash_dim = fusedann_env_size_t(
            "FUSEDANN_BITMAP_HASH_DIM",
            default_hash_dim,
            1);
        const size_t dense_dim = std::min(hash_dim, attr_vocab);
        ds.attr_dim = dense_dim;
        ds.attr_fused_dim = dense_dim;
        std::cout << "    ↳ hashing sparse attrs into dense bitmaps (dim="
                  << dense_dim << ", vocab=" << attr_vocab << ")…\n";
        spmat_to_hashed_bitmaps(ds.attr_indptr,
                                ds.attr_indices,
                                ds.nb,
                                dense_dim,
                                ds.attr_vocab,
                                ds.bitmaps,
                                ds.bitmaps_float);
      }
    }
    else
    {
      throw std::runtime_error("Unsupported base attribute format: " + base_attrs_bvecs);
    }
    std::cout << "    ✔ base attributes loaded\n";

    std::cout << "[3/9] Loading query vectors from " << query_fvecs << " …\n";
    const std::string query_ext = lowercase_extension(query_fvecs);
    if (query_ext == ".fvecs")
    {
      load_fvecs(query_fvecs, ds.xq, rows, dim);
    }
    else if (query_ext == ".u8bin")
    {
      load_u8bin(query_fvecs, ds.xq, rows, dim);
    }
    else
    {
      throw std::runtime_error("Unsupported query vector format: " + query_fvecs);
    }
    ds.nq = rows;
    if (ds.dim != dim)
      throw std::runtime_error("Query dimension mismatch with base vectors.");
    std::cout << "    ✔ query vectors: " << ds.nq << " × " << ds.dim << "\n";

    std::cout << "[4/9] Loading query attributes from " << query_attrs_bvecs << " …\n";
    const std::string query_attr_ext = lowercase_extension(query_attrs_bvecs);
    if (query_attr_ext == ".bvecs")
    {
      size_t attr_rows = 0;
      size_t attr_dim = 0;
      load_bvecs(query_attrs_bvecs, ds.q_bitmaps, attr_rows, attr_dim);
      if (attr_rows != ds.nq || attr_dim != ds.attr_dim)
        throw std::runtime_error(
            "Query attribute data mismatch with query vectors or base attributes.");
      ds.q_bitmaps_float.resize(ds.q_bitmaps.size());
      std::cout << "    ↳ converting query attrs uint8 → float …\n";
      for (size_t i = 0; i < ds.q_bitmaps.size(); ++i)
      {
        ds.q_bitmaps_float[i] = static_cast<float>(ds.q_bitmaps[i]);
      }
    }
    else if (query_attr_ext == ".spmat")
    {
      size_t attr_rows = 0;
      size_t attr_vocab = 0;
      load_spmat(query_attrs_bvecs, ds.q_attr_indptr, ds.q_attr_indices, attr_rows, attr_vocab);
      if (attr_rows != ds.nq)
        throw std::runtime_error("Query sparse attribute rows do not match queries.");
      if (attr_vocab != ds.attr_vocab)
        throw std::runtime_error("Query sparse attribute vocab mismatch with base attributes.");
      std::cout << "    ↳ sorting query sparse rows …\n";
      sort_sparse_rows(ds.q_attr_indptr, ds.q_attr_indices);
      if (!ds.sparse_geometric)
      {
        std::cout << "    ↳ hashing query sparse attrs into dense bitmaps …\n";
        spmat_to_hashed_bitmaps(ds.q_attr_indptr,
                                ds.q_attr_indices,
                                ds.nq,
                                ds.attr_dim,
                                ds.attr_vocab,
                                ds.q_bitmaps,
                                ds.q_bitmaps_float);
      }
    }
    else
    {
      throw std::runtime_error("Unsupported query attribute format: " + query_attrs_bvecs);
    }
    std::cout << "    ✔ query attributes loaded\n";

    std::cout << "[5/9] Applying attribute PCA …\n";
    log_parallel_backend();
    const size_t bitmap_pca_dim = fusedann_env_size_t(
      "FUSEDANN_BITMAP_PCA_DIM",
      std::min<size_t>(ds.attr_dim, static_cast<size_t>(10)),
      1);
    if (ds.sparse_geometric)
      apply_sparse_attribute_pca(ds, bitmap_pca_dim);
    else
      apply_dense_attribute_pca(ds, bitmap_pca_dim);

    if (!gt_ivecs_path.empty())
    {
      std::cout << "[6/9] Loading ground truth from " << gt_ivecs_path << " …\n";
      std::ifstream gt_probe(gt_ivecs_path, std::ios::binary);
      if (!gt_probe)
      {
        std::cout << "📭 Ground truth file missing at " << gt_ivecs_path
                  << " (will proceed without cached GT)\n";
      }
      else
      {
        gt_probe.close();
        const std::string gt_ext = lowercase_extension(gt_ivecs_path);
        size_t gt_rows = 0;
        size_t k = 0;
        if (gt_ext == ".ivecs")
        {
          load_ivecs(gt_ivecs_path, ds.groundtruth, gt_rows, k);
        }
        else if (gt_ext == ".ibin")
        {
          load_ibin(gt_ivecs_path, ds.groundtruth, gt_rows, k);
        }
        else
        {
          throw std::runtime_error("Unsupported ground truth format: " + gt_ivecs_path);
        }
        if (gt_rows != ds.nq)
          throw std::runtime_error("Ground truth row count does not match queries.");
        ds.gt_k = k;
        std::cout << "    ✔ ground truth loaded (k=" << ds.gt_k << ")\n";
      }
    }
    else
    {
      std::cout << "[6/9] Skipping ground truth load (path empty)\n";
    }

    std::cout << "[7/9] Final dataset summary …\n";
    std::cout << "✅ Loaded dataset:\n"
              << "   • Base vectors:  " << ds.nb << " × " << ds.dim << "\n"
              << "   • Query vectors: " << ds.nq << " × " << ds.dim << "\n";
    std::cout << "   • Attribute dim: " << ds.attr_dim;
    if (ds.attributes_pca)
    {
      std::cout << " (PCA → " << ds.attr_fused_dim << ")";
    }
    std::cout << "\n";

    std::cout << "[8/9] Dataset ready for use.\n";
    std::cout << "[9/9] load_dataset() complete.\n";

    return ds;
  }
  // -----------------------------------------------------------------------------
  // Attribute grouping (keyed by raw bitmap bytes)
  // -----------------------------------------------------------------------------
  using AttrKey = std::string;
  using AttrGroupMap = std::unordered_map<AttrKey, std::vector<size_t>>;

  inline AttrGroupMap build_attribute_groups(const std::vector<uint8_t> &attrs,
                                             size_t rows, size_t attr_dim)
  {
    AttrGroupMap groups;
    groups.reserve(rows);
    const uint8_t *ptr = attrs.data();

    for (size_t i = 0; i < rows; ++i)
    {
      AttrKey key(reinterpret_cast<const char *>(ptr + i * attr_dim),
                  static_cast<long>(attr_dim));
      groups[key].push_back(i);
    }
    std::cout << "🧩 Attribute groups built: " << groups.size()
              << " unique combinations\n";
    return groups;
  }

  inline AttrGroupMap build_sparse_attribute_groups(const Dataset &ds)
  {
    if (!ds.attributes_sparse)
      throw std::runtime_error("build_sparse_attribute_groups called on dense dataset");
    if (ds.attr_indptr.size() != ds.nb + 1)
      throw std::runtime_error("Sparse attribute indptr has invalid shape.");
    AttrGroupMap groups;
    groups.reserve(ds.nb);
    std::string key;
    for (size_t i = 0; i < ds.nb; ++i)
    {
      const int64_t begin = ds.attr_indptr[i];
      const int64_t end = ds.attr_indptr[i + 1];
      if (begin > end)
        throw std::runtime_error("Sparse attribute indptr must be non-decreasing.");
      const size_t count = static_cast<size_t>(end - begin);
      key.assign(reinterpret_cast<const char *>(ds.attr_indices.data() + begin),
                 static_cast<long>(count * sizeof(int32_t)));
      groups[key].push_back(i);
    }
    std::cout << "🧩 Attribute groups built (sparse): " << groups.size()
              << " unique combinations\n";
    return groups;
  }

  // -----------------------------------------------------------------------------
  // Distance helpers
  // -----------------------------------------------------------------------------
  inline float l2_distance_sq(const float *a, const float *b, size_t dim)
  {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i)
    {
      float diff = a[i] - b[i];
      dist += diff * diff;
    }
    return dist;
  }

  inline std::vector<int32_t> compute_filtered_groundtruth(const Dataset &ds,
                                                           const AttrGroupMap &groups,
                                                           size_t k)
  {
    std::vector<int32_t> filtered(ds.nq * k, -1);
    if (k == 0 || ds.attr_dim == 0 || ds.dim == 0)
      return filtered;

    std::unordered_map<AttrKey, std::vector<size_t>> attr_to_queries;
    attr_to_queries.reserve(groups.size());

    for (size_t qi = 0; qi < ds.nq; ++qi)
    {
      AttrKey key(reinterpret_cast<const char *>(ds.q_bitmaps.data() + qi * ds.attr_dim),
                  static_cast<long>(ds.attr_dim));
      attr_to_queries[key].push_back(qi);
    }

    for (auto &entry : attr_to_queries)
    {
      const AttrKey &key = entry.first;
      const auto base_it = groups.find(key);
      if (base_it == groups.end())
        continue;

      const std::vector<size_t> &base_ids = base_it->second;
      if (base_ids.empty())
        continue;

      const size_t k_eff = std::min(k, base_ids.size());
      std::vector<std::pair<float, int32_t>> candidates(base_ids.size());

      for (size_t qi : entry.second)
      {
        const float *query_vec = ds.xq.data() + qi * ds.dim;

        for (size_t bi = 0; bi < base_ids.size(); ++bi)
        {
          size_t base_idx = base_ids[bi];
          const float *base_vec = ds.xb.data() + base_idx * ds.dim;
          float dist = l2_distance_sq(query_vec, base_vec, ds.dim);
          candidates[bi] = {dist, static_cast<int32_t>(base_idx)};
        }

        auto middle = candidates.begin() + static_cast<std::ptrdiff_t>(k_eff);
        std::partial_sort(candidates.begin(), middle, candidates.end(),
                          [](const auto &lhs, const auto &rhs)
                          {
                            return lhs.first < rhs.first;
                          });

        for (size_t kk = 0; kk < k_eff; ++kk)
        {
          filtered[qi * k + kk] = candidates[kk].second;
        }
      }
    }

    return filtered;
  }

  inline float max_pairwise_distance(const float *data, size_t rows, size_t dim,
                                     size_t block = PAIRWISE_BLOCK_SIZE)
  {
    if (rows < 2)
      return 0.0f;

    const size_t blk = std::max<size_t>(1, std::min(block, rows));
    float max_dist_sq = 0.0f;

    for (size_t i0 = 0; i0 < rows; i0 += blk)
    {
      const size_t i1 = std::min(rows, i0 + blk);
      for (size_t j0 = i0; j0 < rows; j0 += blk)
      {
        const size_t j1 = std::min(rows, j0 + blk);
        for (size_t i = i0; i < i1; ++i)
        {
          const float *vi = data + i * dim;
          for (size_t j = (j0 == i0) ? i + 1 : j0; j < j1; ++j)
          {
            if (i == j)
              continue;
            const float *vj = data + j * dim;
            const float dist_sq = l2_distance_sq(vi, vj, dim);
            if (dist_sq > max_dist_sq)
              max_dist_sq = dist_sq;
          }
        }
      }
    }
    return std::sqrt(max_dist_sq);
  }

  // -----------------------------------------------------------------------------
  // α / β estimation (mirrors Python logic)
  // -----------------------------------------------------------------------------
  inline double l2_distance(const float *a, const float *b, size_t dim)
  {
    double dist = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
      const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      dist += diff * diff;
    }
    return std::sqrt(dist);
  }

  template <typename T>
  inline T percentile_from_sorted(const std::vector<T> &values, double pct01)
  {
    if (values.empty())
      return static_cast<T>(0);
    const double clamped = std::clamp(pct01, 0.0, 1.0);
    const double pos = clamped * static_cast<double>(values.size() - 1);
    const size_t idx = static_cast<size_t>(std::floor(pos));
    const size_t idx2 = std::min(idx + 1, values.size() - 1);
    const double frac = pos - static_cast<double>(idx);
    return static_cast<T>((1.0 - frac) * values[idx] + frac * values[idx2]);
  }

  template <typename T>
  inline T percentile(std::vector<T> values, double pct01)
  {
    if (values.empty())
      return static_cast<T>(0);
    std::sort(values.begin(), values.end());
    return percentile_from_sorted(values, pct01);
  }

  inline double estimate_intra_diameter_quantile(const Dataset &ds,
                                                 const AttrGroupMap &groups,
                                                 double percentile_pct,
                                                 size_t sample_per_group,
                                                 std::mt19937_64 &rng)
  {
    std::vector<double> diameters;
    diameters.reserve(groups.size());

    for (const auto &[_, ids] : groups)
    {
      if (ids.size() < 2)
        continue;

      size_t sample_sz = ids.size();
      if (sample_per_group > 0 && ids.size() > sample_per_group)
      {
        sample_sz = sample_per_group;
      }

      std::vector<size_t> sub_ids = ids;
      if (sample_sz < ids.size())
      {
        std::shuffle(sub_ids.begin(), sub_ids.end(), rng);
        sub_ids.resize(sample_sz);
      }

      std::vector<float> sample(sample_sz * ds.dim);
      for (size_t i = 0; i < sample_sz; ++i)
      {
        std::memcpy(sample.data() + i * ds.dim,
                    ds.xb.data() + sub_ids[i] * ds.dim,
                    sizeof(float) * ds.dim);
      }
      diameters.push_back(max_pairwise_distance(sample.data(), sample_sz, ds.dim));
    }

    if (diameters.empty())
      return 1.0;

    std::sort(diameters.begin(), diameters.end());
    const double pct01 = std::clamp(percentile_pct / 100.0, 0.0, 1.0);
    return percentile_from_sorted(diameters, pct01);
  }

  inline double estimate_cross_content_quantile(const Dataset &ds,
                                                const AttrGroupMap &groups,
                                                size_t num_pairs,
                                                double percentile_pct,
                                                std::mt19937_64 &rng)
  {
    if (groups.size() < 2 || ds.nb < 2)
      return 1.0;

    std::vector<const std::vector<size_t> *> buckets;
    buckets.reserve(groups.size());
    for (const auto &kv : groups)
    {
      if (!kv.second.empty())
        buckets.push_back(&kv.second);
    }
    if (buckets.size() < 2)
      return 1.0;

    std::uniform_int_distribution<size_t> bucket_dist(0, buckets.size() - 1);
    std::vector<double> distances;
    distances.reserve(num_pairs);

    size_t generated = 0;
    size_t attempts = 0;
    const size_t max_attempts = num_pairs * 10 + 1;

    while (generated < num_pairs && attempts < max_attempts)
    {
      const size_t bi = bucket_dist(rng);
      size_t bj = bucket_dist(rng);
      ++attempts;

      if (bi == bj)
        continue;

      const auto &A = *buckets[bi];
      const auto &B = *buckets[bj];
      if (A.empty() || B.empty())
        continue;

      std::uniform_int_distribution<size_t> a_dist(0, A.size() - 1);
      std::uniform_int_distribution<size_t> b_dist(0, B.size() - 1);

      const size_t idx_a = A[a_dist(rng)];
      const size_t idx_b = B[b_dist(rng)];
      const double dist = l2_distance(ds.xb.data() + idx_a * ds.dim,
                                      ds.xb.data() + idx_b * ds.dim,
                                      ds.dim);
      distances.push_back(dist);
      ++generated;
    }

    if (distances.empty())
      return 1.0;

    const double pct01 = std::clamp(percentile_pct / 100.0, 0.0, 1.0);
    return percentile(distances, pct01);
  }

  inline double attribute_centroid_distance_quantile(const Dataset &ds,
                                                     const AttrGroupMap &groups,
                                                     double percentile_pct,
                                                     std::mt19937_64 &rng)
  {
    if (groups.size() < 2 || ds.attr_fused_dim == 0)
      return 1.0;

    std::vector<const AttrGroupMap::value_type *> sampled_groups;
    sampled_groups.reserve(std::min(groups.size(), MAX_ATTR_CENTROID_SAMPLES));
    size_t seen = 0;
    for (const auto &entry : groups)
    {
      if (entry.second.empty())
        continue;
      ++seen;
      if (sampled_groups.size() < MAX_ATTR_CENTROID_SAMPLES)
      {
        sampled_groups.push_back(&entry);
      }
      else
      {
        std::uniform_int_distribution<size_t> dist(0, seen - 1);
        const size_t idx = dist(rng);
        if (idx < sampled_groups.size())
          sampled_groups[idx] = &entry;
      }
    }

    if (sampled_groups.size() < 2)
      return 1.0;

    const size_t centroid_count = sampled_groups.size();
    std::vector<double> centroids(centroid_count * ds.attr_fused_dim, 0.0);

    for (size_t gi = 0; gi < centroid_count; ++gi)
    {
      const auto &ids = sampled_groups[gi]->second;
      double *mean = centroids.data() + gi * ds.attr_fused_dim;
      for (size_t id : ids)
      {
        const float *attr = ds.bitmaps_float.data() + id * ds.attr_fused_dim;
        for (size_t d = 0; d < ds.attr_fused_dim; ++d)
        {
          mean[d] += static_cast<double>(attr[d]);
        }
      }
      const double inv = 1.0 / static_cast<double>(ids.size());
      for (size_t d = 0; d < ds.attr_fused_dim; ++d)
      {
        mean[d] *= inv;
      }
    }

    std::vector<double> distances;
    distances.reserve(centroid_count * (centroid_count - 1) / 2);
    for (size_t i = 0; i < centroid_count; ++i)
    {
      const double *ci = centroids.data() + i * ds.attr_fused_dim;
      for (size_t j = i + 1; j < centroid_count; ++j)
      {
        const double *cj = centroids.data() + j * ds.attr_fused_dim;
        double acc = 0.0;
        for (size_t d = 0; d < ds.attr_fused_dim; ++d)
        {
          const double diff = ci[d] - cj[d];
          acc += diff * diff;
        }
        const double dist = std::sqrt(acc);
        if (dist > 0.0)
          distances.push_back(dist);
      }
    }

    if (distances.empty())
      return 1.0;

    const double pct01 = std::clamp(percentile_pct / 100.0, 0.0, 1.0);
    return percentile(distances, pct01);
  }

  inline AlphaBetaStats auto_alpha_beta(const Dataset &ds,
                                        const AttrGroupMap &groups,
                                        double intra_percentile = ALPHA_PERCENTILE,
                                        double inter_percentile = ALPHA_PERCENTILE,
                                        double attr_percentile = ALPHA_ATTR_PERCENTILE)
  {
    std::mt19937_64 rng(ALPHA_RANDOM_SEED);
    const auto start = std::chrono::high_resolution_clock::now();

    AlphaBetaStats stats;

    const auto stage1_begin = std::chrono::high_resolution_clock::now();
    std::cout << "   • Stage 1/3: estimating within-attribute diameter quantile ("
              << intra_percentile << "%, up to " << ALPHA_SAMPLE_PER_ATTR
              << " samples each)..." << std::flush;
    const double q_intra = estimate_intra_diameter_quantile(ds, groups,
                                                            intra_percentile,
                                                            ALPHA_SAMPLE_PER_ATTR,
                                                            rng);
    stats.cluster_radius = q_intra;
    const auto stage1_end = std::chrono::high_resolution_clock::now();
    const double stage1_elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(stage1_end - stage1_begin)
            .count();
    std::cout << std::fixed << std::setprecision(2)
              << " done (" << stage1_elapsed << " s, q_intra ≈ " << q_intra << ")\n"
              << std::defaultfloat;

    const auto stage2_begin = std::chrono::high_resolution_clock::now();
    std::cout << "   • Stage 2/3: estimating cross-attribute content quantile ("
              << inter_percentile << "%, " << ALPHA_CROSS_SAMPLES
              << " sampled pairs)..." << std::flush;
    const double q_inter = estimate_cross_content_quantile(ds, groups,
                                                           ALPHA_CROSS_SAMPLES,
                                                           inter_percentile,
                                                           rng);
    stats.mu_max = q_inter;
    const auto stage2_end = std::chrono::high_resolution_clock::now();
    const double stage2_elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(stage2_end - stage2_begin)
            .count();
    std::cout << std::fixed << std::setprecision(2)
              << " done (" << stage2_elapsed << " s, q_inter ≈ " << q_inter << ")\n"
              << std::defaultfloat;

    const auto stage3_begin = std::chrono::high_resolution_clock::now();
    std::cout << "   • Stage 3/3: estimating attribute distance quantile ("
              << attr_percentile << "%)..." << std::flush;
    double sigma_min = attribute_centroid_distance_quantile(ds, groups,
                                                            attr_percentile,
                                                            rng);
    const auto stage3_end = std::chrono::high_resolution_clock::now();
    const double stage3_elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(stage3_end - stage3_begin)
            .count();
    std::cout << std::fixed << std::setprecision(2)
              << " done (" << stage3_elapsed << " s, σ_min ≈ " << sigma_min << ")\n"
              << std::defaultfloat;

    const double eps = 1e-6;
    const double delta_intra = std::max(q_intra, eps);
    const double delta_inter = std::max(q_inter, eps);
    sigma_min = std::max(sigma_min, eps);

    double beta = delta_intra / std::max(ALPHA_EPS_FUSED, eps);
    beta = std::max(beta, static_cast<double>(MIN_BETA));

    const double attr_dim = static_cast<double>(std::max<size_t>(ds.attr_fused_dim, 1));
    const double content_dim = static_cast<double>(std::max<size_t>(ds.dim, 1));
    const double K = std::max(1.0, content_dim / attr_dim);

    double alpha = (delta_inter + ALPHA_MARGIN * delta_intra) /
                   (std::sqrt(K) * sigma_min);
    alpha = std::max(alpha, static_cast<double>(MIN_ALPHA));

    stats.beta = beta;
    stats.alpha = alpha;
    stats.min_attr_distance = sigma_min;

    const auto end = std::chrono::high_resolution_clock::now();
    stats.elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    std::cout << std::fixed << std::setprecision(4)
              << "   → α ≈ " << stats.alpha
              << ", β ≈ " << stats.beta
              << std::setprecision(2)
              << "   (K=" << K
              << ", ε_f=" << ALPHA_EPS_FUSED
              << ", μ=" << ALPHA_MARGIN << ")\n"
              << std::defaultfloat;

    return stats;
  }

  // -----------------------------------------------------------------------------
  // Transformer (faithful C++ port of fused_space/utils/transform.py)
  // -----------------------------------------------------------------------------
  class Transformer
  {
  public:
    static void SingleTransform(const float *vectors,
                                const float *attributes,
                                size_t batch_size,
                                size_t content_dim,
                                size_t attr_dim,
                                float alpha,
                                float beta,
                                float *out)
    {
      if (content_dim == 0)
        throw std::invalid_argument("content_dim must be positive.");
      const float alpha_f = alpha;
      const float beta_f = beta;
      if (beta_f == 0.0f)
      {
        throw std::invalid_argument("beta must be non-zero.");
      }
      const float inv_beta = 1.0f / beta_f;
      const float alpha_over_beta = alpha_f * inv_beta;

      if (attr_dim == 0)
      {
        for (size_t row = 0; row < batch_size; ++row)
        {
          const float *vec_row = vectors + row * content_dim;
          float *out_row = out + row * content_dim;
          for (size_t j = 0; j < content_dim; ++j)
          {
            out_row[j] = vec_row[j] * inv_beta;
          }
        }
        return;
      }

      if (attributes == nullptr)
        throw std::invalid_argument("attributes pointer must be non-null when attr_dim > 0");

      std::vector<float> expanded_buffer;
      if (attr_dim != content_dim)
      {
        expanded_buffer.resize(content_dim);
      }
#if defined(__AVX2__)
      const bool use_avx2 = content_dim >= 8;
#else
      const bool use_avx2 = false;
#endif

      for (size_t row = 0; row < batch_size; ++row)
      {
        const float *vec_row = vectors + row * content_dim;
        const float *attr_row = attributes + row * attr_dim;
        float *out_row = out + row * content_dim;

        const float *expanded_ptr = nullptr;
        if (attr_dim == content_dim)
        {
          expanded_ptr = attr_row;
        }
        else
        {
          ExpandRow(attr_row, attr_dim, content_dim, expanded_buffer.data());
          expanded_ptr = expanded_buffer.data();
        }

#if defined(__AVX2__)
        if (use_avx2)
        {
          const __m256 inv_beta_vec = _mm256_set1_ps(inv_beta);
          const __m256 alpha_over_beta_vec = _mm256_set1_ps(alpha_over_beta);
          size_t j = 0;
          for (; j + 8 <= content_dim; j += 8)
          {
            const __m256 vec_vals = _mm256_loadu_ps(vec_row + j);
            const __m256 attr_vals = _mm256_loadu_ps(expanded_ptr + j);
            const __m256 scaled_vec = _mm256_mul_ps(inv_beta_vec, vec_vals);
            const __m256 scaled_attr = _mm256_mul_ps(alpha_over_beta_vec, attr_vals);
            const __m256 fused = _mm256_sub_ps(scaled_vec, scaled_attr);
            _mm256_storeu_ps(out_row + j, fused);
          }
          for (; j < content_dim; ++j)
          {
            out_row[j] = vec_row[j] * inv_beta - expanded_ptr[j] * alpha_over_beta;
          }
        }
        else
#endif
        {
          for (size_t j = 0; j < content_dim; ++j)
          {
            out_row[j] = vec_row[j] * inv_beta - expanded_ptr[j] * alpha_over_beta;
          }
        }
      }
    }

  private:
    using Pair = std::pair<size_t, size_t>;

    struct PairHash
    {
      size_t operator()(const Pair &p) const noexcept
      {
        return std::hash<size_t>()(p.first) ^ (std::hash<size_t>()(p.second) << 1);
      }
    };

    static const std::vector<int> &RepeatIndices(size_t source_dim, size_t target_dim)
    {
      if (source_dim == 0)
        throw std::invalid_argument("Attribute vectors must have positive dimensionality.");

      std::lock_guard<std::mutex> guard(cache_mutex_);
      const Pair key{source_dim, target_dim};
      auto it = repeat_index_cache_.find(key);
      if (it != repeat_index_cache_.end())
      {
        return it->second;
      }

      auto [insert_it, inserted] =
          repeat_index_cache_.emplace(key, std::vector<int>{});
      std::vector<int> &indices = insert_it->second;
      indices.resize(target_dim);
      for (size_t j = 0; j < target_dim; ++j)
      {
        indices[j] = static_cast<int>(j % source_dim);
      }
      return indices;
    }

    static void ExpandRow(const float *attrs,
                          size_t source_dim,
                          size_t target_dim,
                          float *out)
    {
      if (source_dim == target_dim)
      {
        std::memcpy(out, attrs, sizeof(float) * target_dim);
        return;
      }
      const auto &repeats = RepeatIndices(source_dim, target_dim);
      for (size_t j = 0; j < target_dim; ++j)
      {
        out[j] = attrs[repeats[j]];
      }
    }

    inline static std::unordered_map<Pair, std::vector<int>, PairHash> repeat_index_cache_;
    inline static std::mutex cache_mutex_;
  };

  // -----------------------------------------------------------------------------
  // Recall computation (k is number of neighbors per query)
  // -----------------------------------------------------------------------------
  inline double compute_recall(const std::vector<int32_t> &pred,
                               const std::vector<int32_t> &gt,
                               size_t nq, size_t k, size_t gt_k, size_t nb)
  {
    size_t correct = 0;
    size_t total_valid_gt = 0;

    for (size_t i = 0; i < nq; ++i)
    {
      const int32_t *p = pred.data() + i * k;
      const int32_t *g = gt.data() + i * gt_k;

      size_t valid_gt_count = 0;
      for (size_t t = 0; t < k; ++t)
      {
        if (g[t] != -1 && static_cast<size_t>(g[t]) < nb)
          valid_gt_count++;
      }

      if (valid_gt_count == 0)
        continue;

      total_valid_gt += valid_gt_count;

      for (size_t j = 0; j < k; ++j)
      {
        if (p[j] == -1)
          continue;
        for (size_t t = 0; t < k; ++t)
        {
          if (g[t] != -1 && static_cast<size_t>(g[t]) < nb && p[j] == g[t])
          {
            ++correct;
            break;
          }
        }
      }
    }
    return total_valid_gt == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total_valid_gt);
  }

  // -----------------------------------------------------------------------------
  // Brute-force exact kNN (fallback if ground truth unavailable)
  // -----------------------------------------------------------------------------
  inline std::vector<int32_t> brute_force_knn(const std::vector<float> &xb,
                                              const std::vector<float> &xq,
                                              size_t nb, size_t nq,
                                              size_t dim, size_t k)
  {
    std::vector<int32_t> gt(nq * k, -1);
    std::vector<float> dist(nq * k, std::numeric_limits<float>::max());

    for (size_t qi = 0; qi < nq; ++qi)
    {
      const float *q = xq.data() + qi * dim;
      for (size_t bi = 0; bi < nb; ++bi)
      {
        const float *b = xb.data() + bi * dim;
        float d = l2_distance_sq(q, b, dim);
        for (size_t slot = 0; slot < k; ++slot)
        {
          if (d < dist[qi * k + slot])
          {
            for (size_t shift = k - 1; shift > slot; --shift)
            {
              dist[qi * k + shift] = dist[qi * k + shift - 1];
              gt[qi * k + shift] = gt[qi * k + shift - 1];
            }
            dist[qi * k + slot] = d;
            gt[qi * k + slot] = static_cast<int32_t>(bi);
            break;
          }
        }
      }
    }
    return gt;
  }

} // namespace fusedann
