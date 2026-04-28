#include "distance.h"
#include <immintrin.h> // REQUIRED FOR SIMD AVX2

float compute_l2sq(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    uint32_t i = 0;
    
    // Initialize a 256-bit register to hold 8 zeroes
    __m256 sum256 = _mm256_setzero_ps();
    
    // Process 8 dimensions at a time in a single clock cycle
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }
    
    // Unpack the 256-bit register back into a standard float array
    float tmp[8];
    _mm256_storeu_ps(tmp, sum256);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    // Handle any leftover dimensions (e.g., if dims was 130 instead of 128)
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sum;
}