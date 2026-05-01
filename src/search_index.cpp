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

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index <index_path>"
              << " --data <fvecs_path>"
              << " --queries <query_fvecs_path>"
              << " --gt <ground_truth_ivecs_path>"
              << " --K <num_neighbors>"
              << " --L <comma_separated_L_values>"
              << std::endl;
}

static std::vector<uint32_t> parse_L_values(const std::string& s) {
    std::vector<uint32_t> values;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back((uint32_t)std::atoi(token.c_str()));
    }
    std::sort(values.begin(), values.end());
    return values;
}

static double compute_recall(const std::vector<uint32_t>& result,
                             const std::vector<uint32_t>& gt, uint32_t K) {
    uint32_t found = 0;
    // We check how many of our top-K results appear in the top-K ground truth[cite: 1].
    for (uint32_t i = 0; i < result.size(); i++) {
        for (uint32_t j = 0; j < K && j < gt.size(); j++) {
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
        else if (arg == "--K" && i + 1 < argc)    K = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc)    L_str = argv[++i];
    }

    if (index_path.empty() || data_path.empty() || query_path.empty() ||
        gt_path.empty() || L_str.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<uint32_t> L_values = parse_L_values(L_str);

    std::cout << "Loading Vamana Index..." << std::endl;
    VamanaIndex index;
    index.load(index_path, data_path);

    std::cout << "Loading queries..." << std::endl;
    FloatMatrix queries = load_fvecs(query_path);

    std::cout << "Loading ground truth..." << std::endl;
    std::vector<std::vector<uint32_t>> gt = load_ivecs(gt_path);

    // Measure memory footprint after the full index and base vectors are in RAM[cite: 1].
    double memory_mb = get_memory_usage_mb();

    uint32_t nq = queries.npts;
    std::cout << "\n=== SIFT1M Baseline Results (K=" << K << ") ===" << std::endl;
    std::cout << "Global RAM Usage: " << std::fixed << std::setprecision(2) << memory_mb << " MB\n" << std::endl;

    std::cout << std::left << std::setw(8) << "L" 
              << std::setw(14) << "Recall@10" 
              << std::setw(18) << "Avg Latency (us)" 
              << std::setw(15) << "Memory (MB)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    for (uint32_t L : L_values) {
        std::vector<double> recalls(nq);
        std::vector<double> latencies(nq);

        #pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t q = 0; q < nq; q++) {
            SearchResult res = index.search(queries.get_row(q), K, L);
            recalls[q] = compute_recall(res.ids, gt[q], K);
            latencies[q] = res.latency_us;
        }

        double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / nq;
        double avg_lat = std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;

        std::cout << std::left << std::setw(8) << L
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(18) << std::fixed << std::setprecision(1) << avg_lat
                  << std::setw(15) << std::fixed << std::setprecision(1) << memory_mb
                  << std::endl;
    }

    return 0;
}