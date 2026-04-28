#ifndef DISTANCE_H
#define DISTANCE_H

#include <cstdint>

float compute_l2sq(const float* a, const float* b, uint32_t dim);
uint32_t compute_l2sq_u8(const uint8_t* a, const uint8_t* b, uint32_t dim);

#endif