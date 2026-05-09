/**
 * kernels.cuh — CUDA kernel declarations for constraint checking
 */

#ifndef CONSTRAINT_ENGINE_KERNELS_CUH
#define CONSTRAINT_ENGINE_KERNELS_CUH

#include <cstdint>

/* === Bounds-check kernels (one per precision) === */

void launch_bounds_check_i8(const int8_t* values, const int8_t* lo, const int8_t* hi,
                            uint64_t* mask, int count, cudaStream_t stream);

void launch_bounds_check_i16(const int16_t* values, const int16_t* lo, const int16_t* hi,
                             uint64_t* mask, int count, cudaStream_t stream);

void launch_bounds_check_i32(const int32_t* values, const int32_t* lo, const int32_t* hi,
                             uint64_t* mask, int count, cudaStream_t stream);

void launch_bounds_check_f32(const float* values, const float* lo, const float* hi,
                             uint64_t* mask, int count, cudaStream_t stream);

void launch_bounds_check_f64(const double* values, const double* lo, const double* hi,
                             uint64_t* mask, int count, cudaStream_t stream);

/* === Eisenstein kernel === */

void launch_eisenstein_check(const int32_t* a, const int32_t* b,
                             int32_t radius_squared, uint64_t* mask,
                             int count, cudaStream_t stream);

/* === Violation count reduction kernel === */

void launch_violation_count(const uint64_t* mask, int mask_words, int* count_out,
                            cudaStream_t stream);

/* === Histogram kernel (optional, for stats) === */

void launch_histogram(const uint64_t* mask, int mask_words, int* histogram_out,
                      int num_buckets, cudaStream_t stream);

#endif /* CONSTRAINT_ENGINE_KERNELS_CUH */
