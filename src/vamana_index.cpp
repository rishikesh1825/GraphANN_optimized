#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <stdexcept>
#include <cstdlib>

// ===================== SQ8 QUANTIZATION =====================

void VamanaIndex::quantize_data() {
    min_vals_.assign(dim_, std::numeric_limits<float>::max());
    max_vals_.assign(dim_, std::numeric_limits<float>::lowest());

    for (uint32_t i = 0; i < npts_; i++) {
        const float* vec = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++) {
            min_vals_[d] = std::min(min_vals_[d], vec[d]);
            max_vals_[d] = std::max(max_vals_[d], vec[d]);
        }
    }

    quantized_data_.resize(npts_ * dim_);

    for (uint32_t i = 0; i < npts_; i++) {
        const float* vec = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++) {
            float minv = min_vals_[d];
            float maxv = max_vals_[d];

            float norm = (vec[d] - minv) / (maxv - minv + 1e-8f);
            int8_t q = static_cast<int8_t>(norm * 255.0f - 128);
            quantized_data_[i * dim_ + d] = q;
        }
    }

    std::cout << "SQ8 Quantization complete\n";
}

float VamanaIndex::compute_l2sq_sq8(const float* query, uint32_t idx) const {
    float sum = 0.0f;
    for (uint32_t d = 0; d < dim_; d++) {
        int8_t q = quantized_data_[idx * dim_ + d];
        float minv = min_vals_[d];
        float maxv = max_vals_[d];
        float deq = ((q + 128.0f) / 255.0f) * (maxv - minv) + minv;
        float diff = query[d] - deq;
        sum += diff * diff;
    }
    return sum;
}

// ===================== DESTRUCTOR =====================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ===================== GREEDY SEARCH =====================

struct SearchNode {
    float dist;
    uint32_t id;
    bool expanded;
    bool operator<(const SearchNode& other) const {
        if (dist == other.dist) return id < other.id;
        return dist < other.dist;
    }
};

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {

    std::vector<uint8_t> visited(npts_, 0);
    std::vector<SearchNode> candidates;
    candidates.reserve(L + 1);

    uint32_t dist_cmps = 0;
    float start_dist = use_sq_
        ? compute_l2sq_sq8(query, start_node_)
        : compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;

    candidates.push_back({start_dist, start_node_, false});
    visited[start_node_] = 1;

    while (true) {
        // 1. Find the best UNEXPANDED node
        int best_idx = -1;
        for (size_t i = 0; i < candidates.size(); i++) {
            if (!candidates[i].expanded) {
                best_idx = i;
                break;
            }
        }

        // If everything is expanded, we are done
        if (best_idx == -1) break;

        // 2. Mark as expanded
        candidates[best_idx].expanded = true;
        uint32_t best_node = candidates[best_idx].id;

        std::vector<uint32_t> neighbors;
        {
            if (!use_sq_) std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        // 3. Explore neighbors (Early stopping explicitly removed to handle SQ8 noise)
        for (uint32_t nbr : neighbors) {
            if (visited[nbr]) continue;
            visited[nbr] = 1;

            float d = use_sq_
                ? compute_l2sq_sq8(query, nbr)
                : compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            // Insert into sorted position
            SearchNode new_node = {d, nbr, false};
            auto it = std::lower_bound(candidates.begin(), candidates.end(), new_node);

            if (it != candidates.end() || candidates.size() < L) {
                candidates.insert(it, new_node);
                if (candidates.size() > L) {
                    candidates.pop_back(); // strictly cap at L
                }
            }
        }
    }

    std::vector<Candidate> results;
    for (const auto& c : candidates) {
        results.push_back({c.dist, c.id});
    }
    return {results, dist_cmps};
}

// ===================== ROBUST PRUNE =====================

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {

    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Merge existing graph neighbors (Necessary for graph degree control loop)
    for (uint32_t exist_nbr : graph_[node]) {
        float d = compute_l2sq(get_vector(node), get_vector(exist_nbr), dim_);
        candidates.push_back({d, exist_nbr});
    }
    
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.second == b.second; }), 
        candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (auto& [dist, id] : candidates) {
        if (new_neighbors.size() >= R) break;

        bool keep = true;
        for (uint32_t prev : new_neighbors) {
            float d = compute_l2sq(get_vector(id), get_vector(prev), dim_);
            if (dist > alpha * d) {
                keep = false;
                break;
            }
        }
        if (keep) new_neighbors.push_back(id);
    }

    graph_[node] = new_neighbors;
}

// ===================== SINGLE-PASS BUILD =====================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {

    if (data_ == nullptr) {
        FloatMatrix mat = load_fbin(data_path);
        npts_ = mat.npts;
        dim_ = mat.dims;
        data_ = mat.data.release();
        owns_data_ = true;
    }

    // 1. CRITICAL: Initialize structures BEFORE anything else
    graph_.assign(npts_, std::vector<uint32_t>());
    locks_ = std::vector<std::mutex>(npts_);

    bool old_use_sq = use_sq_;
    use_sq_ = false;

    // 2. MEDOID START NODE ALGORITHM
    std::vector<float> centroid(dim_, 0.0f);
    for (uint32_t i = 0; i < npts_; i++) {
        const float* vec = get_vector(i);
        for (uint32_t j = 0; j < dim_; j++) {
            centroid[j] += vec[j];
        }
    }
    for (uint32_t j = 0; j < dim_; j++) {
        centroid[j] /= (float)npts_;
    }

    uint32_t medoid = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < npts_; i++) {
        float dist = compute_l2sq(get_vector(i), centroid.data(), dim_);
        if (dist < best_dist) {
            best_dist = dist;
            medoid = i;
        }
    }
    start_node_ = medoid;
    std::cout << "Medoid start node set to: " << start_node_ << std::endl;

    std::vector<uint32_t> order(npts_);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(order.begin(), order.end(), rng);

    std::cout << "Building Index (Parallel Single-Pass)..." << std::endl;
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);

    // 3. SINGLE-PASS BUILD
    #pragma omp parallel for schedule(dynamic, 64)
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t node = order[i];
        
        // Use a local copy of query vector to prevent concurrent access issues
        const float* query_vec = get_vector(node);
        
        auto [candidates, _] = greedy_search(query_vec, L);
        robust_prune(node, candidates, alpha, R);
        
        for (const auto& cand : candidates) {
            uint32_t nbr = cand.second;
            if (nbr == node) continue;

            std::lock_guard<std::mutex> lock(locks_[nbr]);
            graph_[nbr].push_back(node);
            
            if (graph_[nbr].size() > gamma_R) {
                std::vector<Candidate> nbr_cands;
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_cands.push_back({d, nn});
                }
                robust_prune(nbr, nbr_cands, alpha, R);
            }
        }

        if ((i + 1) % 100000 == 0) {
            #pragma omp critical
            std::cout << "Inserted " << (i + 1) << " / " << npts_ << std::endl;
        }
    }

    use_sq_ = old_use_sq;
    if (use_sq_) {
        quantize_data();
    }
}

// ===================== SEARCH =====================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;
    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L);

    SearchResult res;
    res.dist_cmps = dist_cmps;
    res.latency_us = t.elapsed_us();

    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        res.ids.push_back(candidates[i].second);
    }
    return res;
}

// ===================== SAVE / LOAD =====================

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    out.write((char*)&npts_, 4);
    out.write((char*)&dim_, 4);
    out.write((char*)&start_node_, 4);

    for (auto& nbrs : graph_) {
        uint32_t sz = nbrs.size();
        out.write((char*)&sz, 4);
        out.write((char*)nbrs.data(), sz * sizeof(uint32_t));
    }
}

void VamanaIndex::load(const std::string& index_path, const std::string& data_path) {
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_ = mat.dims;
    data_ = mat.data.release();

    std::ifstream in(index_path, std::ios::binary);
    in.read((char*)&npts_, 4);
    in.read((char*)&dim_, 4);
    in.read((char*)&start_node_, 4);

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t sz;
        in.read((char*)&sz, 4);
        graph_[i].resize(sz);
        in.read((char*)graph_[i].data(), sz * sizeof(uint32_t));
    }

    if (use_sq_) {
        quantize_data();
    }
}