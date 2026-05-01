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
// Lifecycle
// ============================================================================
VamanaIndex::VamanaIndex() : data_(nullptr), u8_data_(nullptr), owns_data_(false), npts_(0), dim_(0), start_node_(0), pq_M_(0), locks_(nullptr) {}

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) delete[] data_;
    if (u8_data_) _aligned_free(u8_data_); // Clean up SQ8 data
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
    size_t total_elements = (size_t)npts_ * dim_;
    std::cout << "  Allocating " << total_elements / (1024*1024) << " MB for SQ8 Re-ranker..." << std::endl;
    
    if (u8_data_) _aligned_free(u8_data_);
    u8_data_ = (uint8_t*)_aligned_malloc(total_elements, 64);
    
    if (u8_data_ == nullptr) throw std::runtime_error("Failed to allocate SQ8 buffer!");

    for (size_t i = 0; i < total_elements; i++) {
        u8_data_[i] = (uint8_t)std::clamp((int)data_[i], 0, 255);
    }
}

// ============================================================================
// Search Engines
// ============================================================================

// Internal search used ONLY during graph construction (uses high-precision floats)
std::pair<std::vector<VamanaIndex::Candidate>, uint32_t> 
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    ATVB& vis_buf = thread_buffers_[tid];
    
    std::set<Candidate> candidate_set;
    vis_buf.init(npts_); vis_buf.reset();
    uint32_t dist_cmps = 0;
    
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    candidate_set.insert({start_dist, start_node_});
    vis_buf.should_visit(start_node_, start_dist);
    
    std::set<uint32_t> expanded;
    while (true) {
        uint32_t best_node = UINT32_MAX;
        for (const auto& cand : candidate_set) {
            if (expanded.find(cand.second) == expanded.end()) { best_node = cand.second; break; }
        }
        if (best_node == UINT32_MAX) break;
        expanded.insert(best_node);
        
        std::vector<uint32_t> nbrs;
        if (locks_) {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            nbrs = graph_[best_node];
        } else {
            nbrs = graph_[best_node]; 
        }

        for (uint32_t nbr : nbrs) {
            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;
            if (!vis_buf.should_visit(nbr, d)) continue;
            candidate_set.insert({d, nbr});
            if (candidate_set.size() > L) candidate_set.erase(std::prev(candidate_set.end()));
        }
    }
    return {std::vector<Candidate>(candidate_set.begin(), candidate_set.end()), dist_cmps};
}

// The Two-Stage Product Quantization + SQ8 Re-ranker
SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    Timer t;
    
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    ATVB& vis_buf = thread_buffers_[tid];
    
    // STAGE 1 PREP: Precompute PQ LUT and quantize query to SQ8
    std::vector<float> lut(pq_M_ * 256);
    compute_pq_lut(query, pq_codebook_.data(), lut.data(), pq_M_, dim_);
    
    std::vector<uint8_t> q_u8(dim_);
    for(uint32_t i=0; i<dim_; i++) q_u8[i] = (uint8_t)std::clamp((int)query[i], 0, 255);
    
    std::set<Candidate> candidate_set;
    vis_buf.init(npts_); vis_buf.reset();
    uint32_t dist_cmps = 0;

    float start_dist = compute_adc_distance(lut.data(), pq_data_.data() + start_node_ * pq_M_, pq_M_);
    candidate_set.insert({start_dist, start_node_});
    vis_buf.should_visit(start_node_, start_dist);

    std::set<uint32_t> expanded;
    
    // STAGE 1: Fast Graph Traversal using PQ
    while (true) {
        uint32_t best_node = UINT32_MAX;
        for (const auto& cand : candidate_set) {
            if (expanded.find(cand.second) == expanded.end()) { best_node = cand.second; break; }
        }
        if (best_node == UINT32_MAX) break;
        expanded.insert(best_node);

        std::vector<uint32_t> nbrs;
        if (locks_) {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            nbrs = graph_[best_node];
        } else {
            nbrs = graph_[best_node]; 
        }

        for (uint32_t nbr : nbrs) {
            __builtin_prefetch(pq_data_.data() + nbr * pq_M_, 0, 3);
            float d = compute_adc_distance(lut.data(), pq_data_.data() + nbr * pq_M_, pq_M_);
            dist_cmps++;

            if (!vis_buf.should_visit(nbr, d)) continue;
            candidate_set.insert({d, nbr});
            if (candidate_set.size() > L) candidate_set.erase(std::prev(candidate_set.end()));
        }
    }
    
    // STAGE 2: Precision Re-ranking using SQ8
    std::vector<Candidate> reranked_candidates;
    reranked_candidates.reserve(candidate_set.size());
    
    for (const auto& cand : candidate_set) {
        uint32_t node = cand.second;
        // Compute the true SQ8 distance using AVX2 SIMD
        float true_dist = (float)compute_l2sq_u8(q_u8.data(), u8_data_ + (size_t)node * dim_, dim_);
        reranked_candidates.push_back({true_dist, node});
    }
    
    // Re-sort the final L candidates based on their mathematically more precise distances
    std::sort(reranked_candidates.begin(), reranked_candidates.end());
    
    SearchResult result;
    result.dist_cmps = dist_cmps; 
    result.latency_us = t.elapsed_us();
    for (uint32_t i = 0; i < K && i < reranked_candidates.size(); i++) {
        result.ids.push_back(reranked_candidates[i].second);
    }
    return result;
}

// ============================================================================
// Core Vamana Logic (Build Phase)
// ============================================================================
void VamanaIndex::robust_prune(uint32_t p, std::vector<Candidate>& candidates, float alpha, uint32_t R) {
    std::vector<Candidate> current_candidates = candidates;
    for (uint32_t nbr : graph_[p]) {
        current_candidates.push_back({compute_l2sq(get_vector(p), get_vector(nbr), dim_), nbr});
    }
    std::sort(current_candidates.begin(), current_candidates.end());
    
    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R); 
    
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
    
    graph_[p].clear();
    graph_[p].insert(graph_[p].end(), new_neighbors.begin(), new_neighbors.end());
}

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L, float alpha, float gamma) {
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; 
    dim_ = mat.dims; 
    
    auto safe_data_ptr = std::move(mat.data);
    data_ = safe_data_ptr.get();
    owns_data_ = false; 
    
    Timer construction_timer;
    start_node_ = calculate_medoid();
    
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    
    graph_.clear();
    graph_.resize(npts_);
    std::mt19937 rng(42);
    
    for (uint32_t i = 0; i < npts_; i++) {
        graph_[i].reserve(gamma_R + 2); 
        for (uint32_t j = 0; j < R; j++) {
            uint32_t neighbor = rng() % npts_;
            if (neighbor != i) graph_[i].push_back(neighbor);
        }
    }

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    thread_buffers_.resize(max_threads);
    locks_.reset(new std::mutex[npts_]);
    
    std::cout << "  Refining graph edges..." << std::endl;
    
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

// ============================================================================
// IO Operations
// ============================================================================
void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    out.write((char*)&npts_, 4); 
    out.write((char*)&dim_, 4); 
    out.write((char*)&start_node_, 4);
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t size = (uint32_t)graph_[i].size();
        out.write((char*)&size, 4); 
        out.write((char*)graph_[i].data(), size * 4);
    }
}

void VamanaIndex::load_pq(const std::string& index_path, const std::string& codebook_path, const std::string& pq_data_path, const std::string& base_data_path) {
    std::cout << "Loading PQ Codebook and Compressed Data..." << std::endl;
    
    std::ifstream in_cb(codebook_path, std::ios::binary);
    if (!in_cb) throw std::runtime_error("Could not open codebook file!");
    uint32_t f_dim;
    in_cb.read((char*)&pq_M_, 4);
    in_cb.read((char*)&f_dim, 4);
    dim_ = f_dim;
    pq_codebook_.resize(pq_M_ * 256 * (dim_ / pq_M_));
    in_cb.read((char*)pq_codebook_.data(), pq_codebook_.size() * sizeof(float));

    std::ifstream in_pq(pq_data_path, std::ios::binary);
    if (!in_pq) throw std::runtime_error("Could not open pq compressed data!");
    uint32_t f_npts, temp_m;
    in_pq.read((char*)&f_npts, 4);
    in_pq.read((char*)&temp_m, 4);
    npts_ = f_npts;
    pq_data_.resize(npts_ * pq_M_);
    in_pq.read((char*)pq_data_.data(), pq_data_.size());

    // NEW: Load the base data for the Stage-2 Re-ranker
    std::cout << "Loading base data for SQ8 Re-ranker..." << std::endl;
    FloatMatrix mat = load_fbin(base_data_path);
    auto temp_data = std::move(mat.data);
    data_ = temp_data.get(); 
    quantize_data();
    temp_data.reset(); // Drop the floats!
    data_ = nullptr;
    owns_data_ = false;

    std::cout << "Loading Vamana Graph..." << std::endl;
    std::ifstream in_idx(index_path, std::ios::binary);
    if (!in_idx) throw std::runtime_error("Could not open graph index file!");
    
    uint32_t dummy;
    in_idx.read((char*)&dummy, 4); 
    in_idx.read((char*)&dummy, 4); 
    in_idx.read((char*)&start_node_, 4);

    graph_.clear();
    graph_.resize(npts_); 
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t degree;
        if (!in_idx.read((char*)&degree, 4)) break;
        graph_[i].reserve(degree);
        graph_[i].resize(degree);
        in_idx.read((char*)graph_[i].data(), degree * sizeof(uint32_t));
    }
    
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    thread_buffers_.resize(max_threads);
    locks_ = nullptr; // Ensure locks are disabled for lock-free searching
    
    std::cout << "Two-Stage PQ Engine loaded successfully!" << std::endl;
}