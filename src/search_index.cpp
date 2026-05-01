#include "vamana_index.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>

// Windows-specific headers for memory tracking
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

/**
 * Returns the current resident set size (RAM usage) of the process in MB.
 */
static double get_memory_usage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return (double)pmc.PrivateUsage / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    // Fallback for non-Windows environments
    return 0.0; 
#endif
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index <index_path>"
              << " --data <fbin_path>"
              << " --queries <query_fbin_path>"
              << " --gt <ground_truth_ibin_path>"
              << " --K <num_neighbors>"
              << " --L <comma_separated_L_values>"
              << std::endl;
}

static std::vector<uint32_t> parse_L_values(const std::string& s) {
    std::vector<uint32_t> values;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back(std::atoi(token.c_str()));
    }
    std::sort(values.begin(), values.end());
    return values;
}

static double compute_recall(const std::vector<uint32_t>& result,
                             const uint32_t* gt, uint32_t K) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < K && i < result.size(); i++) {
        for (uint32_t j = 0; j < K; j++) {
            if (result[i] == gt[j]) {
                found++;
                break;
            }
        }
    }
    return (double)found / K;
}

int main(int argc, char** argv) {
    std::string index_path, data_path, query_path, gt_path, L_str;
    uint32_t K = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc)    index_path = argv[++i];
        else if (arg == "--data" && i + 1 < argc) data_path = argv[++i];
        else if (arg == "--queries" && i + 1 < argc) query_path = argv[++i];
        else if (arg == "--gt" && i + 1 < argc)   gt_path = argv[++i];
        else if (arg == "--K" && i + 1 < argc)    K = std::atoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc)    L_str = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (index_path.empty() || data_path.empty() || query_path.empty() ||
        gt_path.empty() || L_str.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<uint32_t> L_values = parse_L_values(L_str);
    
    // --- Load index ---
    std::cout << "Loading index..." << std::endl;
    VamanaIndex index;
    index.load_pq(index_path, "pq_codebook.bin", "pq_compressed.bin", data_path);

    // --- Load queries ---
    FloatMatrix queries = load_fbin(query_path);
    
    // --- Load ground truth ---
    IntMatrix gt = load_ibin(gt_path);

    uint32_t nq = queries.npts;

    // --- Run search for each L value ---
    std::cout << "\n=== Vamana Performance Benchmark (K=" << K << ") ===" << std::endl;
    std::cout << std::setw(6)  << "L"
              << std::setw(12) << "Recall@K"
              << std::setw(14) << "Avg Lat(us)"
              << std::setw(14) << "P99 Lat(us)"
              << std::setw(16) << "Batch Time(s)"
              << std::setw(12) << "RAM (MB)"
              << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    for (uint32_t L : L_values) {
        std::vector<double> recalls(nq);
        std::vector<double> latencies(nq);

        Timer batch_timer; // Track total time for this batch

        // Parallel search using OpenMP
        #pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t q = 0; q < nq; q++) {
            SearchResult res = index.search(queries.row(q), K, L);

            recalls[q] = compute_recall(res.ids, gt.row(q), K);
            latencies[q] = res.latency_us;
        }

        double total_batch_time = batch_timer.elapsed_seconds();
        double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / nq;
        double avg_lat = std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;

        // Calculate P99 Latency
        std::sort(latencies.begin(), latencies.end());
        double p99_lat = latencies[(size_t)(0.99 * nq)];

        std::cout << std::setw(6)  << L
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_lat
                  << std::setw(14) << std::fixed << std::setprecision(1) << p99_lat
                  << std::setw(16) << std::fixed << std::setprecision(3) << total_batch_time
                  << std::setw(12) << std::fixed << std::setprecision(1) << get_memory_usage()
                  << std::endl;
    }

    std::cout << "\nBenchmark complete." << std::endl;
    return 0;
}