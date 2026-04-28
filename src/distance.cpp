#include "distance.h"
#include <immintrin.h>

// Floating point distance (used during build and for the first query quantization)
float compute_l2sq(const float* a, const float* b, uint32_t dim) {
    __m256 sum256 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sum256);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; i++) { float d = a[i] - b[i]; sum += d * d; }
    return sum;
}

// SQ8 Integer Distance: Processes 32 dimensions in one cycle!
uint32_t compute_l2sq_u8(const uint8_t* a, const uint8_t* b, uint32_t dim) {
    __m256i sum_vec = _mm256_setzero_si256();
    uint32_t i = 0;
    
    for (; i + 31 < dim; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));

        // Unpack bytes to 16-bit integers
        __m256i al = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
        __m256i ah = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
        __m256i bl = _mm256_unpacklo_epi8(vb, _mm256_setzero_si256());
        __m256i bh = _mm256_unpackhi_epi8(vb, _mm256_setzero_si256());

        // Subtract and Square
        __m256i diff_l = _mm256_sub_epi16(al, bl);
        __m256i diff_h = _mm256_sub_epi16(ah, bh);
        
        sum_vec = _mm256_add_epi32(sum_vec, _mm256_madd_epi16(diff_l, diff_l));
        sum_vec = _mm256_add_epi32(sum_vec, _mm256_madd_epi16(diff_h, diff_h));
    }

    int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, sum_vec);
    uint32_t total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    
    // Tail handling
    for (; i < dim; i++) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        total += d * d;
    }
    return total;
}