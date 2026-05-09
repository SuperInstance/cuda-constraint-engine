/**
 * memory_pool.cu — GPU memory pool manager
 *
 * Pre-allocates GPU buffers on engine creation so the hot path
 * never calls cudaMalloc. Supports multi-precision sizing.
 */

#include "constraint_engine.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

/* Internal memory pool structure */
struct CEMemoryPool {
    void* d_values;        /* GPU buffer for values */
    void* d_lo;            /* GPU buffer for lower bounds */
    void* d_hi;            /* GPU buffer for upper bounds */
    void* d_a;             /* GPU buffer for Eisenstein a components */
    void* d_b;             /* GPU buffer for Eisenstein b components */
    uint64_t* d_mask;      /* GPU buffer for violation mask */
    int* d_count;          /* GPU buffer for violation count */

    int capacity;          /* Max constraints this pool can hold */
    int elem_size;         /* Size of one element in bytes */
    int precision;         /* CE_INT8, CE_INT32, etc. */

    size_t total_allocated;/* Total GPU bytes allocated */
};

/* Get element size for a precision mode */
static int precision_elem_size(int precision) {
    switch (precision) {
        case CE_INT8:  return 1;
        case CE_INT16: return 2;
        case CE_INT32: return 4;
        case CE_FP32:  return 4;
        case CE_FP64:  return 8;
        default:       return 4;
    }
}

extern "C" {

/* Create a memory pool for the given config */
CEMemoryPool* ce_mem_pool_create(int max_constraints, int precision, cudaError_t* err) {
    CEMemoryPool* pool = new CEMemoryPool();
    if (!pool) { *err = cudaErrorMemoryAllocation; return nullptr; }
    memset(pool, 0, sizeof(*pool));

    pool->capacity = max_constraints;
    pool->precision = precision;
    pool->elem_size = precision_elem_size(precision);

    int mask_words = (max_constraints + 63) / 64;
    size_t val_bytes = (size_t)max_constraints * pool->elem_size;
    size_t mask_bytes = (size_t)mask_words * sizeof(uint64_t);

    cudaError_t e = cudaSuccess;

    e = cudaMalloc(&pool->d_values, val_bytes);
    if (e != cudaSuccess) goto fail;

    e = cudaMalloc(&pool->d_lo, val_bytes);
    if (e != cudaSuccess) goto fail;

    e = cudaMalloc(&pool->d_hi, val_bytes);
    if (e != cudaSuccess) goto fail;

    /* Eisenstein uses int32 always */
    e = cudaMalloc(&pool->d_a, (size_t)max_constraints * sizeof(int32_t));
    if (e != cudaSuccess) goto fail;

    e = cudaMalloc(&pool->d_b, (size_t)max_constraints * sizeof(int32_t));
    if (e != cudaSuccess) goto fail;

    e = cudaMalloc(&pool->d_mask, mask_bytes);
    if (e != cudaSuccess) goto fail;

    e = cudaMalloc(&pool->d_count, sizeof(int));
    if (e != cudaSuccess) goto fail;

    /* Zero the mask */
    e = cudaMemset(pool->d_mask, 0, mask_bytes);
    if (e != cudaSuccess) goto fail;

    pool->total_allocated = val_bytes * 3 + (size_t)max_constraints * sizeof(int32_t) * 2
                          + mask_bytes + sizeof(int);

    *err = cudaSuccess;
    return pool;

fail:
    fprintf(stderr, "[CE] Memory pool allocation failed: %s\n", cudaGetErrorString(e));
    if (pool->d_values) cudaFree(pool->d_values);
    if (pool->d_lo)     cudaFree(pool->d_lo);
    if (pool->d_hi)     cudaFree(pool->d_hi);
    if (pool->d_a)      cudaFree(pool->d_a);
    if (pool->d_b)      cudaFree(pool->d_b);
    if (pool->d_mask)   cudaFree(pool->d_mask);
    if (pool->d_count)  cudaFree(pool->d_count);
    delete pool;
    *err = e;
    return nullptr;
}

void ce_mem_pool_destroy(CEMemoryPool* pool) {
    if (!pool) return;
    if (pool->d_values) cudaFree(pool->d_values);
    if (pool->d_lo)     cudaFree(pool->d_lo);
    if (pool->d_hi)     cudaFree(pool->d_hi);
    if (pool->d_a)      cudaFree(pool->d_a);
    if (pool->d_b)      cudaFree(pool->d_b);
    if (pool->d_mask)   cudaFree(pool->d_mask);
    if (pool->d_count)  cudaFree(pool->d_count);
    delete pool;
}

/* Resize pool to new capacity. Returns true on success. */
bool ce_mem_pool_resize(CEMemoryPool** pool, int new_capacity, int precision, cudaError_t* err) {
    CEMemoryPool* old = *pool;
    CEMemoryPool* np = ce_mem_pool_create(new_capacity, precision, err);
    if (!np) return false;

    /* Copy old bounds to new pool if they exist */
    if (old) {
        int copy_count = (old->capacity < new_capacity) ? old->capacity : new_capacity;
        size_t copy_bytes = (size_t)copy_count * old->elem_size;
        cudaMemcpy(np->d_lo, old->d_lo, copy_bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(np->d_hi, old->d_hi, copy_bytes, cudaMemcpyDeviceToDevice);
        ce_mem_pool_destroy(old);
    }
    *pool = np;
    return true;
}

/* Getters */
void* ce_mem_pool_values(CEMemoryPool* p) { return p->d_values; }
void* ce_mem_pool_lo(CEMemoryPool* p) { return p->d_lo; }
void* ce_mem_pool_hi(CEMemoryPool* p) { return p->d_hi; }
int32_t* ce_mem_pool_a(CEMemoryPool* p) { return (int32_t*)p->d_a; }
int32_t* ce_mem_pool_b(CEMemoryPool* p) { return (int32_t*)p->d_b; }
uint64_t* ce_mem_pool_mask(CEMemoryPool* p) { return p->d_mask; }
int* ce_mem_pool_count(CEMemoryPool* p) { return p->d_count; }
int ce_mem_pool_capacity(CEMemoryPool* p) { return p->capacity; }
int ce_mem_pool_elem_size(CEMemoryPool* p) { return p->elem_size; }
size_t ce_mem_pool_total(CEMemoryPool* p) { return p->total_allocated; }

} /* extern "C" */
