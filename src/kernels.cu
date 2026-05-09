/**
 * kernels.cu — CUDA kernels for constraint checking
 *
 * Multi-precision bounds checking + Eisenstein norm kernel.
 * All kernels are launchable from host code via the launch_* functions.
 */

#include "kernels.cuh"
#include <cstdio>

/* ========================================================================
 * Bounds-check kernels — templated inner kernel
 * ======================================================================== */

template<typename T>
__global__ void bounds_check_kernel(const T* __restrict__ values,
                                     const T* __restrict__ lo,
                                     const T* __restrict__ hi,
                                     uint64_t* __restrict__ mask,
                                     int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread handles one value, packs results into 64-bit mask words
    if (idx < count) {
        T v = values[idx];
        bool violated = (v < lo[idx]) || (v > hi[idx]);
        int word = idx / 64;
        int bit  = idx % 64;
        if (violated) {
            atomicOr(reinterpret_cast<unsigned long long*>(mask) + word, (1ULL << bit));
        }
    }
}

/* Eisenstein norm check: a² + ab + b² ≤ r² */
__global__ void eisenstein_check_kernel(const int32_t* __restrict__ a,
                                         const int32_t* __restrict__ b,
                                         int32_t radius_squared,
                                         uint64_t* __restrict__ mask,
                                         int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int32_t va = a[idx];
        int32_t vb = b[idx];
        int64_t norm = (int64_t)va * va + (int64_t)va * vb + (int64_t)vb * vb;
        bool violated = norm > (int64_t)radius_squared;
        int word = idx / 64;
        int bit  = idx % 64;
        if (violated) {
            atomicOr(reinterpret_cast<unsigned long long*>(mask) + word, (1ULL << bit));
        }
    }
}

/* Violation count: popcount reduction */
__global__ void violation_count_kernel(const uint64_t* __restrict__ mask,
                                        int mask_words,
                                        int* __restrict__ count_out) {
    extern __shared__ int s_sum[];
    int tid = threadIdx.x;
    s_sum[tid] = 0;

    for (int i = tid; i < mask_words; i += blockDim.x) {
        s_sum[tid] += __popcll(mask[i]);
    }
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0) *count_out = s_sum[0];
}

/* ========================================================================
 * Launch wrappers — called from host code
 * ======================================================================== */

static constexpr int BLOCK_SIZE = 256;

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

void launch_bounds_check_i8(const int8_t* values, const int8_t* lo, const int8_t* hi,
                            uint64_t* mask, int count, cudaStream_t stream) {
    int blocks = div_up(count, BLOCK_SIZE);
    bounds_check_kernel<int8_t><<<blocks, BLOCK_SIZE, 0, stream>>>(values, lo, hi, mask, count);
}

void launch_bounds_check_i16(const int16_t* values, const int16_t* lo, const int16_t* hi,
                             uint64_t* mask, int count, cudaStream_t stream) {
    int blocks = div_up(count, BLOCK_SIZE);
    bounds_check_kernel<int16_t><<<blocks, BLOCK_SIZE, 0, stream>>>(values, lo, hi, mask, count);
}

void launch_bounds_check_i32(const int32_t* values, const int32_t* lo, const int32_t* hi,
                             uint64_t* mask, int count, cudaStream_t stream) {
    int blocks = div_up(count, BLOCK_SIZE);
    bounds_check_kernel<int32_t><<<blocks, BLOCK_SIZE, 0, stream>>>(values, lo, hi, mask, count);
}

void launch_bounds_check_f32(const float* values, const float* lo, const float* hi,
                             uint64_t* mask, int count, cudaStream_t stream) {
    int blocks = div_up(count, BLOCK_SIZE);
    bounds_check_kernel<float><<<blocks, BLOCK_SIZE, 0, stream>>>(values, lo, hi, mask, count);
}

void launch_bounds_check_f64(const double* values, const double* lo, const double* hi,
                             uint64_t* mask, int count, cudaStream_t stream) {
    int blocks = div_up(count, BLOCK_SIZE);
    bounds_check_kernel<double><<<blocks, BLOCK_SIZE, 0, stream>>>(values, lo, hi, mask, count);
}

void launch_eisenstein_check(const int32_t* a, const int32_t* b,
                             int32_t radius_squared, uint64_t* mask,
                             int count, cudaStream_t stream) {
    int blocks = div_up(count, BLOCK_SIZE);
    eisenstein_check_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, radius_squared, mask, count);
}

void launch_violation_count(const uint64_t* mask, int mask_words, int* count_out,
                            cudaStream_t stream) {
    violation_count_kernel<<<1, 256, 256 * sizeof(int), stream>>>(mask, mask_words, count_out);
}

void launch_histogram(const uint64_t* mask, int mask_words, int* histogram_out,
                      int num_buckets, cudaStream_t stream) {
    /* Placeholder — full histogram would bucket by violation pattern */
    (void)mask; (void)mask_words; (void)histogram_out; (void)num_buckets; (void)stream;
}
