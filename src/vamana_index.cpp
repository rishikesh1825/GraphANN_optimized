#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <malloc.h> // Required for _aligned_malloc on Windows

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// ============================================================================
// Thread-local visited buffer
// ============================================================================
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
thread_local ATVB tl_vis_buf;

// ============================================================================
// Lifecycle
// ============================================================================
VamanaIndex::VamanaIndex() : data_(nullptr), u8_data_(nullptr), owns_data_(false), npts_(0), dim_(0), start_node_(0) {}

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) delete[] data_;
    if (u8_data_) _aligned_free(u8_data_); // Clean up aligned SQ8 buffer
}

// ============================================================================
// Helpers
// ============================================================================
uint32_t VamanaIndex::calculate_medoid() {
    std::vector<float> mean(dim_, 0.0f);
    for (uint32_t i = 0; i < npts_; i++) {
        const float* v = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++) mean[d] += v[d];
    }
    for (float& val : mean) val /= (float)npts_;
    
    uint32_t medoid = 0;
    float min_dist = 1e30f;
    uint32_t sample_size = std::min(npts_, 10000u);
    for (uint32_t i = 0; i < sample_size; i++) {
        float d = compute_l2sq(mean.data(), get_vector(i), dim_);
        if (d < min_dist) { min_dist = d; medoid = i; }
    }
    return medoid;
}

void VamanaIndex::quantize_data() {
    // 1. Calculate size carefully
    size_t total_elements = (size_t)npts_ * dim_;
    std::cout << "  Allocating " << total_elements / (1024*1024) << " MB for SQ8 buffer..." << std::endl;
    
    if (u8_data_) _aligned_free(u8_data_);
    u8_data_ = (uint8_t*)_aligned_malloc(total_elements, 64);
    
    if (u8_data_ == nullptr) {
        throw std::runtime_error("OS refused to allocate SQ8 buffer - out of memory!");
    }

    // 2. USE SERIAL LOOP (No OpenMP) for stability during peak RAM
    for (size_t i = 0; i < total_elements; i++) {
        u8_data_[i] = (uint8_t)std::clamp((int)data_[i], 0, 255);
    }
    std::cout << "  Quantization complete." << std::endl;
}

// ============================================================================
// Search Engines
// ============================================================================
std::pair<std::vector<VamanaIndex::Candidate>, uint32_t> 
VamanaIndex::greedy_search_u8(const uint8_t* query_u8, uint32_t L) const {
    std::set<Candidate> candidate_set;
    tl_vis_buf.init(npts_); tl_vis_buf.reset();
    uint32_t dist_cmps = 0;

    float start_dist = (float)compute_l2sq_u8(query_u8, u8_data_ + (size_t)start_node_ * dim_, dim_);
    candidate_set.insert({start_dist, start_node_});
    tl_vis_buf.should_visit(start_node_, start_dist);

    std::set<uint32_t> expanded;
    while (true) {
        uint32_t best_node = UINT32_MAX;
        for (const auto& cand : candidate_set) {
            if (expanded.find(cand.second) == expanded.end()) { 
                best_node = cand.second; 
                break; 
            }
        }
        if (best_node == UINT32_MAX) break;
        expanded.insert(best_node);

        for (uint32_t nbr : graph_[best_node]) {
            __builtin_prefetch(u8_data_ + (size_t)nbr * dim_, 0, 3);
            float d = (float)compute_l2sq_u8(query_u8, u8_data_ + (size_t)nbr * dim_, dim_);
            dist_cmps++;

            if (!tl_vis_buf.should_visit(nbr, d)) continue;
            candidate_set.insert({d, nbr});
            if (candidate_set.size() > L) candidate_set.erase(std::prev(candidate_set.end()));
        }
    }
    return {std::vector<Candidate>(candidate_set.begin(), candidate_set.end()), dist_cmps};
}

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t> 
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    std::set<Candidate> candidate_set;
    tl_vis_buf.init(npts_); tl_vis_buf.reset();
    uint32_t dist_cmps = 0;
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    candidate_set.insert({start_dist, start_node_});
    tl_vis_buf.should_visit(start_node_, start_dist);
    std::set<uint32_t> expanded;
    while (true) {
        uint32_t best_node = UINT32_MAX;
        for (const auto& cand : candidate_set) {
            if (expanded.find(cand.second) == expanded.end()) { best_node = cand.second; break; }
        }
        if (best_node == UINT32_MAX) break;
        expanded.insert(best_node);
        for (uint32_t nbr : graph_[best_node]) {
            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;
            if (!tl_vis_buf.should_visit(nbr, d)) continue;
            candidate_set.insert({d, nbr});
            if (candidate_set.size() > L) candidate_set.erase(std::prev(candidate_set.end()));
        }
    }
    return {std::vector<Candidate>(candidate_set.begin(), candidate_set.end()), dist_cmps};
}

// ============================================================================
// Core Vamana Logic
// ============================================================================
void VamanaIndex::robust_prune(uint32_t p, std::vector<Candidate>& candidates, float alpha, uint32_t R) {
    std::vector<Candidate> current_candidates = candidates;
    for (uint32_t nbr : graph_[p]) {
        current_candidates.push_back({compute_l2sq(get_vector(p), get_vector(nbr), dim_), nbr});
    }
    std::sort(current_candidates.begin(), current_candidates.end());
    std::vector<uint32_t> new_neighbors;
    for (const auto& cand_star : current_candidates) {
        if (new_neighbors.size() >= R) break;
        bool ok = true;
        for (uint32_t nbr_prime : new_neighbors) {
            if (alpha * compute_l2sq(get_vector(cand_star.second), get_vector(nbr_prime), dim_) <= cand_star.first) {
                ok = false; break;
            }
        }
        if (ok) new_neighbors.push_back(cand_star.second);
    }
    graph_[p] = new_neighbors;
}

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L, float alpha, float gamma) {
    // 1. Load Data
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; 
    dim_ = mat.dims; 
    
    // Release from unique_ptr to the class member
    data_ = mat.data.release(); 
    owns_data_ = true;

    Timer construction_timer;

    // 2. Setup optimizations
    start_node_ = calculate_medoid();
    quantize_data(); // Creates u8_data_ for saving, but keeps data_ valid

    // 3. Initialize Graph
    graph_.assign(npts_, std::vector<uint32_t>());
    locks_ = std::vector<std::mutex>(npts_);
    
    std::mt19937 rng(42);
    for (uint32_t i = 0; i < npts_; i++) {
        for (uint32_t j = 0; j < R; j++) {
            uint32_t neighbor = rng() % npts_;
            if (neighbor != i) graph_[i].push_back(neighbor);
        }
    }

    // 4. Refinement (CRITICAL: Uses greedy_search which requires float data_)
    std::cout << "  Refining graph edges..." << std::endl;
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);

    #pragma omp parallel for schedule(dynamic, 64)
    for (uint32_t i = 0; i < npts_; i++) {
        auto [candidates, cmps] = greedy_search(get_vector(i), L);
        robust_prune(i, candidates, alpha, R);
        
        for (const auto& cand : candidates) {
            uint32_t v = cand.second;
            std::lock_guard<std::mutex> lock(locks_[v]);
            if (std::find(graph_[v].begin(), graph_[v].end(), i) == graph_[v].end()) {
                graph_[v].push_back(i);
                if (graph_[v].size() > gamma_R) {
                    std::vector<Candidate> nbr_cands;
                    for (uint32_t n : graph_[v]) 
                        nbr_cands.push_back({compute_l2sq(get_vector(v), get_vector(n), dim_), n});
                    robust_prune(v, nbr_cands, alpha, R);
                }
            }
        }
    }
    std::cout << "  Construction Time: " << construction_timer.elapsed_seconds() << "s" << std::endl;
}

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    Timer t;
    std::vector<uint8_t> q_u8(dim_);
    for(uint32_t i=0; i<dim_; i++) q_u8[i] = (uint8_t)std::clamp((int)query[i], 0, 255);
    
    auto [candidates, dist_cmps] = greedy_search_u8(q_u8.data(), L);
    
    SearchResult result;
    result.dist_cmps = dist_cmps; result.latency_us = t.elapsed_us();
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) result.ids.push_back(candidates[i].second);
    return result;
}

// ============================================================================
// IO Operations
// ============================================================================
void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    out.write((char*)&npts_, 4); out.write((char*)&dim_, 4); out.write((char*)&start_node_, 4);
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t size = (uint32_t)graph_[i].size();
        out.write((char*)&size, 4); 
        out.write((char*)graph_[i].data(), size * 4);
    }
}

void VamanaIndex::load(const std::string& index_path, const std::string& data_path) {
    try {
        std::cout << "Loading base data for quantization..." << std::endl;
        
        // 1. Load float data
        FloatMatrix mat = load_fbin(data_path);
        npts_ = mat.npts; 
        dim_ = mat.dims;
        
        // Use 'auto' to capture the specific unique_ptr type and deleter
        auto temp_data = std::move(mat.data);
        data_ = temp_data.get(); 

        // 2. Quantize
        quantize_data();

        // 3. THE FIX: Drop the float data immediately
        // Resetting the smart pointer triggers the correct custom deleter
        temp_data.reset(); 
        data_ = nullptr;
        owns_data_ = false;
        
        std::cout << "  Float RAM freed. Loading index file..." << std::endl;

        // 4. Load Index File
        std::ifstream in(index_path, std::ios::binary);
        if (!in) throw std::runtime_error("Could not open index file!");

        uint32_t f_npts, f_dim;
        in.read((char*)&f_npts, 4);
        in.read((char*)&f_dim, 4);
        in.read((char*)&start_node_, 4);

        graph_.clear();
        graph_.resize(npts_); 

        for (uint32_t i = 0; i < npts_; i++) {
            uint32_t degree;
            if (!in.read((char*)&degree, 4)) break;
            graph_[i].resize(degree);
            in.read((char*)graph_[i].data(), degree * sizeof(uint32_t));
        }
        std::cout << "  Index load successful!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nSTABILITY ERROR: " << e.what() << std::endl;
        std::exit(1);
    }
}