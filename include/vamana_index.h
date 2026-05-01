#pragma once
#include <vector>
#include <string>
#include <set>
#include <mutex>
#include <memory>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

struct SearchResult {
    std::vector<uint32_t> ids;
    uint32_t dist_cmps;
    double latency_us;
};

struct ATVB {
    std::vector<uint32_t> visit_id;
    std::vector<float> best_dist;
    uint32_t current_iter;
    ATVB() : current_iter(0) {}
    void init(size_t N) {
        if (visit_id.size() != N) {
            visit_id.assign(N, 0);
            best_dist.assign(N, std::numeric_limits<float>::infinity());
            current_iter = 0;
        }
    }
    void reset() {
        current_iter++;
        if (current_iter == 0) { 
            std::fill(visit_id.begin(), visit_id.end(), 0); 
            current_iter = 1; 
        }
    }
    inline bool should_visit(uint32_t node, float dist) {
        if (visit_id[node] != current_iter || dist < best_dist[node]) {
            visit_id[node] = current_iter;
            best_dist[node] = dist;
            return true;
        }
        return false;
    }
};

class VamanaIndex {
public:
    VamanaIndex();
    ~VamanaIndex();

    void build(const std::string& data_path, uint32_t R, uint32_t L, float alpha, float gamma);
    void save(const std::string& path) const;
    
    // Updated to accept the base data path for the Re-ranker
    void load_pq(const std::string& index_path, const std::string& codebook_path, const std::string& pq_data_path, const std::string& base_data_path);
    
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    using Candidate = std::pair<float, uint32_t>;

private:
    float* data_;
    uint8_t* u8_data_; // Re-added for the Re-ranker
    bool owns_data_;
    uint32_t npts_;
    uint32_t dim_;
    uint32_t start_node_;
    
    std::vector<float> pq_codebook_;
    std::vector<uint8_t> pq_data_;
    uint32_t pq_M_;

    std::vector<std::vector<uint32_t>> graph_;
    std::unique_ptr<std::mutex[]> locks_;
    mutable std::vector<ATVB> thread_buffers_;

    uint32_t calculate_medoid();
    void quantize_data(); // Re-added to create SQ8 data
    void robust_prune(uint32_t p, std::vector<Candidate>& candidates, float alpha, uint32_t R);
    std::pair<std::vector<Candidate>, uint32_t> greedy_search(const float* query, uint32_t L) const;

    inline const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};