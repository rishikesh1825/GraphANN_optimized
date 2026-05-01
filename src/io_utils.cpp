#include "io_utils.h"
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <malloc.h> 

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

double get_memory_usage_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS memCounters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounters, sizeof(memCounters))) {
        return (double)memCounters.WorkingSetSize / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

void* aligned_alloc_wrapper(size_t size) {
    void* ptr = _aligned_malloc(size, 64);
    if (!ptr) throw std::runtime_error("Failed to allocate 64-byte aligned memory");
    return ptr;
}

void aligned_free_wrapper(void* ptr) {
    if (ptr) _aligned_free(ptr);
}

FloatMatrix load_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file: " + path);
    int dim;
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t npts = file_size / (4 + dim * 4);
    FloatMatrix mat;
    mat.npts = (uint32_t)npts;
    mat.dims = (uint32_t)dim;
    mat.data = std::unique_ptr<float[], void(*)(void*)>(
        static_cast<float*>(aligned_alloc_wrapper((size_t)npts * dim * sizeof(float))), 
        aligned_free_wrapper);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < npts; i++) {
        in.seekg(4, std::ios::cur); 
        in.read((char*)(mat.data.get() + i * dim), dim * sizeof(float));
    }
    return mat;
}

FloatMatrix load_fbin(const std::string& path) {
    if (path.find(".fvecs") != std::string::npos) return load_fvecs(path);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + path);
    uint32_t npts, dims;
    in.read(reinterpret_cast<char*>(&npts), 4);
    in.read(reinterpret_cast<char*>(&dims), 4);
    FloatMatrix mat;
    mat.npts = npts; mat.dims = dims;
    mat.data = std::unique_ptr<float[], void(*)(void*)>(
        static_cast<float*>(aligned_alloc_wrapper((size_t)npts * dims * sizeof(float))), 
        aligned_free_wrapper);
    in.read(reinterpret_cast<char*>(mat.data.get()), (size_t)npts * dims * sizeof(float));
    return mat;
}

IntMatrix load_ibin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open ibin file: " + path);
    uint32_t npts, dims;
    in.read(reinterpret_cast<char*>(&npts), 4);
    in.read(reinterpret_cast<char*>(&dims), 4);
    IntMatrix mat;
    mat.npts = npts; mat.dims = dims;
    mat.data = std::unique_ptr<uint32_t[], void(*)(void*)>(
        static_cast<uint32_t*>(aligned_alloc_wrapper((size_t)npts * dims * sizeof(uint32_t))), 
        aligned_free_wrapper);
    in.read(reinterpret_cast<char*>(mat.data.get()), (size_t)npts * dims * sizeof(uint32_t));
    return mat;
}

std::vector<std::vector<uint32_t>> load_ivecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open ivecs file: " + path);
    std::vector<std::vector<uint32_t>> data;
    while (in) {
        uint32_t dim;
        if (!in.read((char*)&dim, 4)) break; 
        std::vector<uint32_t> vec(dim);
        in.read((char*)vec.data(), dim * sizeof(uint32_t));
        data.push_back(vec);
    }
    return data;
}