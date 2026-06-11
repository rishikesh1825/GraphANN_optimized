#include "distance.h"

float compute_l2sq(const float* __restrict a, const float* __restrict b, uint32_t dim) {
    float sum = 0.0f;

    #pragma omp simd reduction(+:sum)
    for (uint32_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}