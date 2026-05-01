#pragma once
#include <cstdint>

// ============================================================================
// High-Precision Float Math (Used during Build phase)
// ============================================================================
float compute_l2sq(const float* a, const float* b, uint32_t dim);

// ============================================================================
// Fast SQ8 Integer Math (Legacy - Kept for compatibility if needed)
// ============================================================================
uint32_t compute_l2sq_u8(const uint8_t* a, const uint8_t* b, uint32_t dim);

// ============================================================================
// Asymmetric Distance Computation (ADC) Math (Used during PQ Search phase)
// ============================================================================
void compute_pq_lut(const float* query, const float* codebook, float* lut, uint32_t M, uint32_t dim);
float compute_adc_distance(const float* lut, const uint8_t* pq_vec, uint32_t M);