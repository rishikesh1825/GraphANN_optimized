#include "io_utils.h"
#include <fstream>
#include <stdexcept>
#include <cstdlib>

// Windows requires special headers for aligned allocation
#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_FREE _aligned_free
#else
#define ALIGNED_FREE std::free
#endif

/**
 * Allocates 64-byte-aligned memory.
 * This is CRITICAL for SIMD (AVX2) performance; standard malloc 
 * does not guarantee the 32-byte or 64-byte boundaries needed 
 * for _mm256_load_ps.
 */
static void* aligned_alloc_wrapper(size_t size) {
    size_t alignment = 64;
    // Round up size to multiple of alignment
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    void* ptr = nullptr;

#ifdef _WIN32
    ptr = _aligned_malloc(aligned_size, alignment);
#else
    ptr = std::aligned_alloc(alignment, aligned_size);
#endif

    if (!ptr)
        throw std::runtime_error("Failed to allocate " + std::to_string(size) + " bytes");
    return ptr;
}

/**
 * Loads .fvecs format files.
 * Format: [int dim][float*dim][int dim][float*dim]...
 */
FloatMatrix load_fbin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    // Read first 4 bytes to get dimensionality
    uint32_t dims;
    in.read(reinterpret_cast<char*>(&dims), 4);
    if (!in.good()) throw std::runtime_error("Failed to read header: " + path);

    // Calculate number of points based on file size
    // Each record is 4 bytes (dim) + (dims * 4 bytes for floats)
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    uint32_t npts = static_cast<uint32_t>(file_size / ((dims + 1) * 4));
    in.seekg(0, std::ios::beg);

    FloatMatrix mat;
    mat.npts = npts;
    mat.dims = dims;
    
    // Use custom deleter to ensure _aligned_free is called on Windows
    mat.data = std::unique_ptr<float[], void(*)(void*)>(
        static_cast<float*>(aligned_alloc_wrapper((size_t)npts * dims * sizeof(float))), 
        ALIGNED_FREE
    );

    for (uint32_t i = 0; i < npts; i++) {
        uint32_t record_dim;
        in.read(reinterpret_cast<char*>(&record_dim), 4); // Skip the dim prefix
        in.read(reinterpret_cast<char*>(mat.data.get() + (size_t)i * dims), dims * sizeof(float));
    }

    return mat;
}

/**
 * Loads .ivecs format files (Ground Truth).
 * Format: Same as .fvecs but with uint32_t instead of float.
 */
IntMatrix load_ibin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    uint32_t dims;
    in.read(reinterpret_cast<char*>(&dims), 4);
    if (!in.good()) throw std::runtime_error("Failed to read header: " + path);

    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    uint32_t npts = static_cast<uint32_t>(file_size / ((dims + 1) * 4));
    in.seekg(0, std::ios::beg);

    IntMatrix mat;
    mat.npts = npts;
    mat.dims = dims;
    
    mat.data = std::unique_ptr<uint32_t[], void(*)(void*)>(
        static_cast<uint32_t*>(aligned_alloc_wrapper((size_t)npts * dims * sizeof(uint32_t))), 
        ALIGNED_FREE
    );

    for (uint32_t i = 0; i < npts; i++) {
        uint32_t record_dim;
        in.read(reinterpret_cast<char*>(&record_dim), 4); // Skip the dim prefix
        in.read(reinterpret_cast<char*>(mat.data.get() + (size_t)i * dims), dims * sizeof(uint32_t));
    }

    return mat;
}