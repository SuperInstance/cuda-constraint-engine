/**
 * constraint_engine.cu — Core engine implementation
 *
 * Orchestrates memory pool, kernels, streams, and CUDA graphs.
 */

#include "constraint_engine.h"
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <chrono>
#include <vector>

/* === Internal types === */

/* Memory pool internal struct (defined in memory_pool.cu) */
struct CEMemoryPool;

/* Opaque handles for C linkage */
struct CEResult {
    int violation_count;
    int check_count;
    uint64_t* h_mask;       /* Host copy of violation mask */
    int mask_words;
    double latency_ms;
    double throughput;
};

struct CEEngine {
    CEMemoryPool* pool;
    void* stream_pool;       /* CEStreamPool* (C++ internal) */
    cudaStream_t default_stream;

    CEConfig config;
    CEStats stats;
    int device_id;
    int compute_cap;
    bool bounds_uploaded;
    int bounds_count;

    char error_buf[256];

    /* CUDA graph support */
    bool graph_ready;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    /* Host staging buffers for async upload */
    void* h_values;
    void* h_lo;
    void* h_hi;
};

/* === Memory pool C-linkage functions (memory_pool.cu) === */
extern "C" {
    struct CEMemoryPool;
    CEMemoryPool* ce_mem_pool_create(int max_constraints, int precision, cudaError_t* err);
    void ce_mem_pool_destroy(CEMemoryPool* pool);
    void* ce_mem_pool_values(CEMemoryPool* p);
    void* ce_mem_pool_lo(CEMemoryPool* p);
    void* ce_mem_pool_hi(CEMemoryPool* p);
    int32_t* ce_mem_pool_a(CEMemoryPool* p);
    int32_t* ce_mem_pool_b(CEMemoryPool* p);
    uint64_t* ce_mem_pool_mask(CEMemoryPool* p);
    int* ce_mem_pool_count(CEMemoryPool* p);
    int ce_mem_pool_capacity(CEMemoryPool* p);
    int ce_mem_pool_elem_size(CEMemoryPool* p);
    size_t ce_mem_pool_total(CEMemoryPool* p);

    void* ce_stream_pool_create(int count);
    void ce_stream_pool_destroy(void* pool);
    cudaStream_t ce_stream_get_cuda(void* pool, int index);
    int ce_stream_pool_size(void* pool);

    void ce_stats_record(CEStats* stats, double latency_ms, int count, int violations);
}

/* === Helpers === */

static void set_error(CEEngine* engine, const char* fmt, ...) {
    if (!engine) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(engine->error_buf, sizeof(engine->error_buf), fmt, args);
    va_end(args);
}

static int mask_words_for(int count) {
    return (count + 63) / 64;
}

/* === API Implementation === */

extern "C" {

void ce_config_default(CEConfig* config) {
    config->max_constraints = 1'000'000;
    config->precision = CE_INT32;
    config->check_mode = CE_MODE_BOUNDS;
    config->enable_graphs = false;
    config->stream_count = 4;
}

CEEngine* ce_create(int device_id) {
    CEConfig config;
    ce_config_default(&config);
    /* device_id -1 means auto-select; we pass it through */
    CEEngine* engine = ce_create_configured(&config);
    if (engine && device_id >= 0) {
        /* Already set by ce_create_configured with auto */
    }
    return engine;
}

CEEngine* ce_create_configured(const CEConfig* config) {
    if (!config) return nullptr;

    CEEngine* engine = new CEEngine();
    if (!engine) return nullptr;
    memset(engine, 0, sizeof(*engine));
    engine->config = *config;
    engine->error_buf[0] = '\0';

    /* Select CUDA device */
    int device_count = 0;
    cudaError_t e = cudaGetDeviceCount(&device_count);
    if (e != cudaSuccess || device_count == 0) {
        set_error(engine, "No CUDA devices found");
        delete engine;
        return nullptr;
    }

    /* Auto-select: pick device 0 (or the one with most memory) */
    int best = 0;
    size_t best_mem = 0;
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if ((size_t)prop.totalGlobalMem > best_mem) {
            best_mem = prop.totalGlobalMem;
            best = i;
        }
    }

    engine->device_id = best;
    e = cudaSetDevice(engine->device_id);
    if (e != cudaSuccess) {
        set_error(engine, "Failed to set CUDA device %d: %s", engine->device_id,
                  cudaGetErrorString(e));
        delete engine;
        return nullptr;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, engine->device_id);
    engine->compute_cap = prop.major * 10 + prop.minor;
    engine->stats.device_id = engine->device_id;
    engine->stats.compute_capability = engine->compute_cap;
    engine->stats.gpu_memory_total = prop.totalGlobalMem;

    /* Create default stream */
    engine->default_stream = 0; /* CUDA default stream */

    /* Create memory pool */
    e = cudaSuccess;
    engine->pool = ce_mem_pool_create(config->max_constraints, config->precision, &e);
    if (!engine->pool) {
        set_error(engine, "Failed to create memory pool: %s", cudaGetErrorString(e));
        delete engine;
        return nullptr;
    }
    engine->stats.gpu_memory_used = ce_mem_pool_total(engine->pool);

    /* Create stream pool */
    if (config->stream_count > 0) {
        engine->stream_pool = ce_stream_pool_create(config->stream_count);
    }

    /* Allocate host staging buffers */
    size_t val_bytes = (size_t)config->max_constraints * ce_mem_pool_elem_size(engine->pool);
    engine->h_values = malloc(val_bytes > 0 ? val_bytes : 4096);
    engine->h_lo = malloc(val_bytes > 0 ? val_bytes : 4096);
    engine->h_hi = malloc(val_bytes > 0 ? val_bytes : 4096);

    return engine;
}

void ce_destroy(CEEngine* engine) {
    if (!engine) return;
    if (engine->pool) ce_mem_pool_destroy(engine->pool);
    if (engine->stream_pool) ce_stream_pool_destroy(engine->stream_pool);
    if (engine->graph_ready) {
        cudaGraphExecDestroy(engine->graph_exec);
        cudaGraphDestroy(engine->graph);
    }
    free(engine->h_values);
    free(engine->h_lo);
    free(engine->h_hi);
    delete engine;
}

const char* ce_get_error(const CEEngine* engine) {
    return engine ? engine->error_buf : "null engine";
}

/* === Data Upload === */

static int upload_bounds_impl(CEEngine* engine, const void* lo, const void* hi,
                               int count, int expected_precision) {
    if (!engine || !lo || !hi) return CE_ERR_PARAM;
    if (count > ce_mem_pool_capacity(engine->pool)) return CE_ERR_OVERFLOW;

    cudaError_t e;
    size_t bytes = (size_t)count * ce_mem_pool_elem_size(engine->pool);

    e = cudaMemcpy(ce_mem_pool_lo(engine->pool), lo, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { set_error(engine, "H2D lo failed: %s", cudaGetErrorString(e)); return CE_ERR_CUDA; }

    e = cudaMemcpy(ce_mem_pool_hi(engine->pool), hi, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { set_error(engine, "H2D hi failed: %s", cudaGetErrorString(e)); return CE_ERR_CUDA; }

    engine->bounds_uploaded = true;
    engine->bounds_count = count;
    engine->graph_ready = false; /* Invalidate CUDA graph */
    return CE_OK;
}

int ce_upload_bounds_i8(CEEngine* engine, const int8_t* lo, const int8_t* hi, int count) {
    return upload_bounds_impl(engine, lo, hi, count, CE_INT8);
}
int ce_upload_bounds_i16(CEEngine* engine, const int16_t* lo, const int16_t* hi, int count) {
    return upload_bounds_impl(engine, lo, hi, count, CE_INT16);
}
int ce_upload_bounds_i32(CEEngine* engine, const int32_t* lo, const int32_t* hi, int count) {
    return upload_bounds_impl(engine, lo, hi, count, CE_INT32);
}
int ce_upload_bounds_f32(CEEngine* engine, const float* lo, const float* hi, int count) {
    return upload_bounds_impl(engine, lo, hi, count, CE_FP32);
}
int ce_upload_bounds_f64(CEEngine* engine, const double* lo, const double* hi, int count) {
    return upload_bounds_impl(engine, lo, hi, count, CE_FP64);
}

int ce_update_bounds(CEEngine* engine, const void* lo, const void* hi, int count) {
    if (!engine || !lo || !hi) return CE_ERR_PARAM;
    if (!engine->bounds_uploaded) return CE_ERR_NO_BOUNDS;

    size_t bytes = (size_t)count * ce_mem_pool_elem_size(engine->pool);
    /* Async copy — doesn't block the default stream */
    cudaMemcpyAsync(ce_mem_pool_lo(engine->pool), lo, bytes, cudaMemcpyHostToDevice, engine->default_stream);
    cudaMemcpyAsync(ce_mem_pool_hi(engine->pool), hi, bytes, cudaMemcpyHostToDevice, engine->default_stream);
    return CE_OK;
}

/* === Constraint Checking === */

static CEResult* do_check(CEEngine* engine, const void* values, int count,
                           cudaStream_t stream, bool is_eisenstein,
                           const int32_t* a, const int32_t* b, int32_t radius_sq) {
    if (!engine) return nullptr;
    if (!engine->bounds_uploaded && !is_eisenstein) {
        set_error(engine, "Bounds not uploaded");
        return nullptr;
    }
    if (count > ce_mem_pool_capacity(engine->pool)) {
        set_error(engine, "Count %d exceeds capacity %d", count, ce_mem_pool_capacity(engine->pool));
        return nullptr;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    int mw = mask_words_for(count);
    size_t mask_bytes = (size_t)mw * sizeof(uint64_t);

    /* Clear mask */
    cudaError_t e = cudaMemsetAsync(ce_mem_pool_mask(engine->pool), 0, mask_bytes, stream);
    if (e != cudaSuccess) { set_error(engine, "Memset failed"); return nullptr; }

    if (is_eisenstein) {
        /* Upload a, b components */
        size_t comp_bytes = (size_t)count * sizeof(int32_t);
        cudaMemcpyAsync(ce_mem_pool_a(engine->pool), a, comp_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ce_mem_pool_b(engine->pool), b, comp_bytes, cudaMemcpyHostToDevice, stream);
        launch_eisenstein_check(ce_mem_pool_a(engine->pool), ce_mem_pool_b(engine->pool),
                                radius_sq, ce_mem_pool_mask(engine->pool), count, stream);
    } else {
        /* Upload values */
        size_t val_bytes = (size_t)count * ce_mem_pool_elem_size(engine->pool);
        cudaMemcpyAsync(ce_mem_pool_values(engine->pool), values, val_bytes, cudaMemcpyHostToDevice, stream);

        /* Launch precision-specific kernel */
        switch (engine->config.precision) {
            case CE_INT8:
                launch_bounds_check_i8((const int8_t*)ce_mem_pool_values(engine->pool),
                    (const int8_t*)ce_mem_pool_lo(engine->pool),
                    (const int8_t*)ce_mem_pool_hi(engine->pool),
                    ce_mem_pool_mask(engine->pool), count, stream);
                break;
            case CE_INT16:
                launch_bounds_check_i16((const int16_t*)ce_mem_pool_values(engine->pool),
                    (const int16_t*)ce_mem_pool_lo(engine->pool),
                    (const int16_t*)ce_mem_pool_hi(engine->pool),
                    ce_mem_pool_mask(engine->pool), count, stream);
                break;
            case CE_INT32:
                launch_bounds_check_i32((const int32_t*)ce_mem_pool_values(engine->pool),
                    (const int32_t*)ce_mem_pool_lo(engine->pool),
                    (const int32_t*)ce_mem_pool_hi(engine->pool),
                    ce_mem_pool_mask(engine->pool), count, stream);
                break;
            case CE_FP32:
                launch_bounds_check_f32((const float*)ce_mem_pool_values(engine->pool),
                    (const float*)ce_mem_pool_lo(engine->pool),
                    (const float*)ce_mem_pool_hi(engine->pool),
                    ce_mem_pool_mask(engine->pool), count, stream);
                break;
            case CE_FP64:
                launch_bounds_check_f64((const double*)ce_mem_pool_values(engine->pool),
                    (const double*)ce_mem_pool_lo(engine->pool),
                    (const double*)ce_mem_pool_hi(engine->pool),
                    ce_mem_pool_mask(engine->pool), count, stream);
                break;
        }
    }

    /* Count violations on GPU */
    launch_violation_count(ce_mem_pool_mask(engine->pool), mw, ce_mem_pool_count(engine->pool), stream);

    /* Synchronize */
    cudaStreamSynchronize(stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /* Download results */
    CEResult* result = new CEResult();
    result->check_count = count;
    result->mask_words = mw;
    result->h_mask = (uint64_t*)malloc(mask_bytes);
    result->latency_ms = latency_ms;
    result->throughput = (latency_ms > 0) ? (count / (latency_ms / 1000.0)) : 0;

    cudaMemcpy(result->h_mask, ce_mem_pool_mask(engine->pool), mask_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&result->violation_count, ce_mem_pool_count(engine->pool), sizeof(int), cudaMemcpyDeviceToHost);

    /* Update stats */
    ce_stats_record(&engine->stats, latency_ms, count, result->violation_count);

    return result;
}

CEResult* ce_check_i8(CEEngine* engine, const int8_t* values, int count) {
    return do_check(engine, values, count, engine->default_stream, false, nullptr, nullptr, 0);
}
CEResult* ce_check_i16(CEEngine* engine, const int16_t* values, int count) {
    return do_check(engine, values, count, engine->default_stream, false, nullptr, nullptr, 0);
}
CEResult* ce_check_i32(CEEngine* engine, const int32_t* values, int count) {
    return do_check(engine, values, count, engine->default_stream, false, nullptr, nullptr, 0);
}
CEResult* ce_check_f32(CEEngine* engine, const float* values, int count) {
    return do_check(engine, values, count, engine->default_stream, false, nullptr, nullptr, 0);
}
CEResult* ce_check_f64(CEEngine* engine, const double* values, int count) {
    return do_check(engine, values, count, engine->default_stream, false, nullptr, nullptr, 0);
}

CEResult* ce_check_eisenstein(CEEngine* engine, const int32_t* a, const int32_t* b,
                               int count, int32_t radius_squared) {
    return do_check(engine, nullptr, count, engine->default_stream, true, a, b, radius_squared);
}

/* === Async API === */

struct CEStream_internal {
    CEEngine* engine;
    cudaStream_t cuda_stream;
    CEResult* pending;
};

CEStream* ce_stream_create(CEEngine* engine) {
    if (!engine) return nullptr;
    CEStream_internal* s = new CEStream_internal();
    s->engine = engine;
    s->pending = nullptr;
    cudaStreamCreateWithFlags(&s->cuda_stream, cudaStreamNonBlocking);
    return (CEStream*)s;
}

static CEStream_internal* unwrap(CEStream* s) { return (CEStream_internal*)s; }

int ce_stream_check_i8(CEStream* stream, const int8_t* values, int count) {
    CEStream_internal* s = unwrap(stream);
    if (s->pending) ce_result_destroy(s->pending);
    s->pending = do_check(s->engine, values, count, s->cuda_stream, false, nullptr, nullptr, 0);
    return s->pending ? CE_OK : CE_ERR_CUDA;
}

int ce_stream_check_i32(CEStream* stream, const int32_t* values, int count) {
    CEStream_internal* s = unwrap(stream);
    if (s->pending) ce_result_destroy(s->pending);
    s->pending = do_check(s->engine, values, count, s->cuda_stream, false, nullptr, nullptr, 0);
    return s->pending ? CE_OK : CE_ERR_CUDA;
}

int ce_stream_check_eisenstein(CEStream* stream, const int32_t* a, const int32_t* b,
                                int count, int32_t radius_squared) {
    CEStream_internal* s = unwrap(stream);
    if (s->pending) ce_result_destroy(s->pending);
    s->pending = do_check(s->engine, nullptr, count, s->cuda_stream, true, a, b, radius_squared);
    return s->pending ? CE_OK : CE_ERR_CUDA;
}

CEResult* ce_stream_result(CEStream* stream) {
    CEStream_internal* s = unwrap(stream);
    CEResult* r = s->pending;
    s->pending = nullptr;
    return r;
}

void ce_stream_destroy(CEStream* stream) {
    CEStream_internal* s = unwrap(stream);
    if (!s) return;
    if (s->pending) ce_result_destroy(s->pending);
    cudaStreamSynchronize(s->cuda_stream);
    cudaStreamDestroy(s->cuda_stream);
    delete s;
}

/* === Result Access === */

int ce_result_violation_count(const CEResult* r) { return r ? r->violation_count : 0; }
const uint64_t* ce_result_violation_mask(const CEResult* r) { return r ? r->h_mask : nullptr; }
double ce_result_throughput(const CEResult* r) { return r ? r->throughput : 0; }
double ce_result_latency_ms(const CEResult* r) { return r ? r->latency_ms : 0; }
int ce_result_count(const CEResult* r) { return r ? r->check_count : 0; }

void ce_result_destroy(CEResult* result) {
    if (!result) return;
    if (result->h_mask) free(result->h_mask);
    delete result;
}

/* === Statistics === */

const CEStats* ce_get_stats(const CEEngine* engine) {
    return engine ? &engine->stats : nullptr;
}

void ce_reset_stats(CEEngine* engine) {
    if (!engine) return;
    memset(&engine->stats, 0, sizeof(engine->stats));
    engine->stats.device_id = engine->device_id;
    engine->stats.compute_capability = engine->compute_cap;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, engine->device_id);
    engine->stats.gpu_memory_total = prop.totalGlobalMem;
    engine->stats.gpu_memory_used = ce_mem_pool_total(engine->pool);
}

/* === Device Query === */

int ce_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

static char device_name_buf[256];

const char* ce_device_name(int device_id) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return "Unknown";
    }
    snprintf(device_name_buf, sizeof(device_name_buf), "%s (sm_%d%d)",
             prop.name, prop.major, prop.minor);
    return device_name_buf;
}

} /* extern "C" */
