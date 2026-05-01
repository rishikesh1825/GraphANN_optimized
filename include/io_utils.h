#pragma once
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

/**
 * Global RAM tracking for baseline verification
 */
double get_memory_usage_mb();

struct FloatMatrix {
    uint32_t npts;
    uint32_t dims;
    // Managed pointer with custom deleter for aligned memory[cite: 4]
    std::unique_ptr<float[], void(*)(void*)> data{nullptr, [](void*){}};

    inline float* get_row(uint32_t i) const { 
        return data.get() + (size_t)i * dims; 
    }
};

struct IntMatrix {
    uint32_t npts;
    uint32_t dims;
    std::unique_ptr<uint32_t[], void(*)(void*)> data{nullptr, [](void*){}};

    inline uint32_t* get_row(uint32_t i) const { 
        return data.get() + (size_t)i * dims; 
    }
};

// Data Loaders for SIFTIM Dataset[cite: 1, 4]
FloatMatrix load_fvecs(const std::string& path);
FloatMatrix load_fbin(const std::string& path);
IntMatrix load_ibin(const std::string& path);
std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path);

// Memory Utilities
void* aligned_alloc_wrapper(size_t size);
void aligned_free_wrapper(void* ptr);