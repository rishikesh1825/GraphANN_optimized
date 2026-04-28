#ifndef VAMANA_INDEX_H
#define VAMANA_INDEX_H

#include "io_utils.h"
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <set>

struct SearchResult {
    std::vector<uint32_t> ids;
    double latency_us;
    uint32_t dist_cmps;
};

class VamanaIndex {
public:
    using Candidate = std::pair<float, uint32_t>;

    VamanaIndex();
    ~VamanaIndex();

    void build(const std::string& data_path, uint32_t R, uint32_t L, float alpha, float gamma);
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

private:
    uint32_t calculate_medoid();
    void quantize_data();
    
    // Search engines
    std::pair<std::vector<Candidate>, uint32_t> greedy_search_u8(const uint8_t* query_u8, uint32_t L) const;
    std::pair<std::vector<Candidate>, uint32_t> greedy_search(const float* query, uint32_t L) const;
    
    void robust_prune(uint32_t p, std::vector<Candidate>& candidates, float alpha, uint32_t R);

    const float* get_vector(uint32_t i) const { return data_ + (size_t)i * dim_; }
    size_t get_total_edges() const {
        size_t total = 0;
        for (const auto& neighbors : graph_) total += neighbors.size();
        return total;
    }

    float* data_;
    uint8_t* u8_data_; 
    bool owns_data_;
    uint32_t npts_;
    uint32_t dim_;
    uint32_t start_node_;
    
    std::vector<std::vector<uint32_t>> graph_;
    mutable std::vector<std::mutex> locks_;
};

#endif