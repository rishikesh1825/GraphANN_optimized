#include "distance.h"
#include <immintrin.h>
#include <cstdint> // Added this!

// ============================================================================
// High-Precision Float Math (Used during Build phase)
// ============================================================================
float compute_l2sq(const float* a, const float* b, uint32_t dim) {
    float dist = 0.0f;
    uint32_t i = 0;

#ifdef __AVX2__
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256); // Fused multiply-add
    }
    // Horizontal addition
    float buffer[8];
    _mm256_storeu_ps(buffer, sum256);
    for (int j = 0; j < 8; ++j) dist += buffer[j];
#endif

    // Remainder loop
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// ============================================================================
// Asymmetric Distance Computation (ADC) Math (Used during PQ Search phase)
// ============================================================================

// 1. Precompute the Lookup Table (LUT) for a single query
void compute_pq_lut(const float* query, const float* codebook, float* lut, uint32_t M, uint32_t dim) {
    uint32_t chunk_dim = dim / M;
    
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t k = 0; k < 256; k++) {
            // Calculate distance from the query's sub-vector to the codebook's centroid
            lut[m * 256 + k] = compute_l2sq(
                query + (m * chunk_dim), 
                codebook + (m * 256 * chunk_dim) + (k * chunk_dim), 
                chunk_dim
            );
        }
    }
}

// 2. The ultra-fast ADC distance sum
float compute_adc_distance(const float* lut, const uint8_t* pq_vec, uint32_t M) {
    float dist = 0.0f;
    for (uint32_t m = 0; m < M; m++) {
        dist += lut[m * 256 + pq_vec[m]]; // Array lookup using the centroid ID
    }
    return dist;
}

// ============================================================================
// Fast SQ8 Integer Math (Used during Stage-2 Re-ranking)
// ============================================================================
uint32_t compute_l2sq_u8(const uint8_t* a, const uint8_t* b, uint32_t dim) {
    uint32_t dist = 0;
    uint32_t i = 0;

#ifdef __AVX2__
    __m256i sum256 = _mm256_setzero_si256();
    // Process 32 bytes (dimensions) at a time
    for (; i + 31 < dim; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        
        // Unpack to 16-bit to prevent overflow during squaring
        __m256i va_lo = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
        __m256i va_hi = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
        __m256i vb_lo = _mm256_unpacklo_epi8(vb, _mm256_setzero_si256());
        __m256i vb_hi = _mm256_unpackhi_epi8(vb, _mm256_setzero_si256());

        __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
        __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);

        // Square and add adjacent pairs
        __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

        sum256 = _mm256_add_epi32(sum256, sq_lo);
        sum256 = _mm256_add_epi32(sum256, sq_hi);
    }
    
    uint32_t buffer[8];
    _mm256_storeu_si256((__m256i*)buffer, sum256);
    for (int j = 0; j < 8; ++j) dist += buffer[j];
#endif

    // Remainder loop
    for (; i < dim; ++i) {
        int diff = (int)a[i] - (int)b[i];
        dist += (uint32_t)(diff * diff);
    }
    
    return dist;
}