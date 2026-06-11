#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>

// Result of a single query search.
struct SearchResult {
    std::vector<uint32_t> ids;  // nearest neighbor IDs (sorted by distance)
    uint32_t dist_cmps;         // number of distance computations
    double latency_us;          // search latency in microseconds
};

class VamanaIndex {
  public:
    VamanaIndex() = default;
    ~VamanaIndex();

    // ---- Build ----
    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, float gamma);

    // ---- Search ----
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // ---- Persistence ----
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_; }

  private:
    // A candidate = (distance, node_id)
    using Candidate = std::pair<float, uint32_t>;

    // ---- Core algorithms ----
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L) const;

    void robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                      float alpha, uint32_t R);

    // ================= DATA =================
    float*   data_    = nullptr;
    uint32_t npts_    = 0;
    uint32_t dim_     = 0;
    bool     owns_data_ = false;

    // ================= GRAPH =================
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    // ================= CONCURRENCY =================
    mutable std::vector<std::mutex> locks_;

    // ================= HELPERS =================
    const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }

    // =====================================================
    // 🔥 SQ8 (SCALAR QUANTIZATION)
    // =====================================================

    // Quantized storage (int8 instead of float)
    std::vector<int8_t> quantized_data_;

    // Per-dimension scaling
    std::vector<float> min_vals_;
    std::vector<float> max_vals_;

    // Toggle SQ8
    bool use_sq_ = true;

    // Quantization function
    void quantize_data();

    // Distance with SQ8 (dequantize on-the-fly)
    float compute_l2sq_sq8(const float* query, uint32_t idx) const;

    // =====================================================
    // (OPTIONAL OLD OPTIMIZATION — NOT USED NOW)
    // =====================================================
    mutable std::vector<int> visited_buffer_;
    mutable uint32_t current_iter_ = 0;
};