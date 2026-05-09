/**
 * constraint_engine.h — GPU Constraint Checking Engine
 *
 * Public API for cuda-constraint-engine.
 * Check billions of constraints per second on NVIDIA GPUs.
 *
 * Usage:
 *   #include "constraint_engine.h"
 *   CEEngine* engine = ce_create(-1);
 *   ce_upload_bounds_i32(engine, lo, hi, count);
 *   CEResult* result = ce_check_i32(engine, values, count);
 *   printf("Violations: %d\n", ce_result_violation_count(result));
 *
 * Copyright (c) 2026 SuperInstance — MIT License
 */

#ifndef CONSTRAINT_ENGINE_H
#define CONSTRAINT_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Opaque Handles === */

/** Opaque engine handle. Manages GPU memory pool, streams, and state. */
typedef struct CEEngine CEEngine;

/** Opaque async stream handle. For concurrent constraint checks. */
typedef struct CEStream CEStream;

/** Opaque result handle. Contains violation mask, count, and timing. */
typedef struct CEResult CEResult;

/* === Precision Modes === */

#define CE_INT8    0
#define CE_INT16   1
#define CE_INT32   2
#define CE_FP32    3
#define CE_FP64    4

/* === Check Modes === */

#define CE_MODE_BOUNDS      0   /**< Standard lo ≤ value ≤ hi checking */
#define CE_MODE_NORM        1   /**< Norm-based constraint checking */
#define CE_MODE_EISENSTEIN  2   /**< Eisenstein integer disk bounds */

/* === Error Codes === */

#define CE_OK               0
#define CE_ERR_CUDA        -1    /**< CUDA runtime error */
#define CE_ERR_PARAM       -2    /**< Invalid parameter */
#define CE_ERR_MEMORY      -3    /**< GPU memory allocation failure */
#define CE_ERR_OVERFLOW    -4    /**< Count exceeds pre-allocated capacity */
#define CE_ERR_NO_BOUNDS   -5    /**< Bounds not uploaded yet */
#define CE_ERR_DEVICE      -6    /**< No suitable CUDA device */

/* === Engine Lifecycle === */

/**
 * Create a constraint engine with default configuration.
 *
 * @param device_id  CUDA device index (-1 = auto-select best GPU)
 * @return Engine handle, or NULL on failure
 *
 * Default config: 1M constraints, INT32, bounds mode, no graphs, 4 streams.
 */
CEEngine* ce_create(int device_id);

/**
 * Create a constraint engine with custom configuration.
 *
 * @param config  Configuration struct (see CEConfig)
 * @return Engine handle, or NULL on failure
 */
CEEngine* ce_create_configured(const struct CEConfig* config);

/**
 * Destroy engine and free all GPU resources.
 * Safe to call with NULL (no-op).
 */
void ce_destroy(CEEngine* engine);

/* === Configuration === */

/** Engine configuration. Passed to ce_create_configured(). */
typedef struct CEConfig {
    int max_constraints;       /**< Pre-allocate GPU memory for this many constraints */
    int precision;             /**< CE_INT8, CE_INT16, CE_INT32, CE_FP32, or CE_FP64 */
    int check_mode;            /**< CE_MODE_BOUNDS, CE_MODE_NORM, or CE_MODE_EISENSTEIN */
    bool enable_graphs;        /**< Use CUDA graphs for fixed-size workloads (18x launch speedup) */
    int stream_count;          /**< Async stream pool size (0 = synchronous only) */
} CEConfig;

/**
 * Get default configuration.
 * @param config  Struct to populate with defaults
 */
void ce_config_default(struct CEConfig* config);

/**
 * Get the last error message for this engine.
 * @return Human-readable error string, valid until next API call on this engine.
 */
const char* ce_get_error(const CEEngine* engine);

/* === Data Upload === */

/**
 * Upload lower/upper bounds (INT8). Stays on GPU until replaced.
 * @return CE_OK on success, error code on failure
 */
int ce_upload_bounds_i8(CEEngine* engine, const int8_t* lo, const int8_t* hi, int count);

/**
 * Upload lower/upper bounds (INT16).
 * @return CE_OK on success, error code on failure
 */
int ce_upload_bounds_i16(CEEngine* engine, const int16_t* lo, const int16_t* hi, int count);

/**
 * Upload lower/upper bounds (INT32).
 * @return CE_OK on success, error code on failure
 */
int ce_upload_bounds_i32(CEEngine* engine, const int32_t* lo, const int32_t* hi, int count);

/**
 * Upload lower/upper bounds (FP32).
 * @return CE_OK on success, error code on failure
 */
int ce_upload_bounds_f32(CEEngine* engine, const float* lo, const float* hi, int count);

/**
 * Upload lower/upper bounds (FP64).
 * @return CE_OK on success, error code on failure
 */
int ce_upload_bounds_f64(CEEngine* engine, const double* lo, const double* hi, int count);

/* === Constraint Checking === */

/**
 * Check INT8 values against uploaded bounds.
 * @param engine  Engine handle
 * @param values  Host array of values to check
 * @param count   Number of values
 * @return Result handle (must be freed with ce_result_destroy), or NULL on error
 */
CEResult* ce_check_i8(CEEngine* engine, const int8_t* values, int count);

/**
 * Check INT16 values against uploaded bounds.
 */
CEResult* ce_check_i16(CEEngine* engine, const int16_t* values, int count);

/**
 * Check INT32 values against uploaded bounds.
 */
CEResult* ce_check_i32(CEEngine* engine, const int32_t* values, int count);

/**
 * Check FP32 values against uploaded bounds.
 */
CEResult* ce_check_f32(CEEngine* engine, const float* values, int count);

/**
 * Check FP64 values against uploaded bounds.
 */
CEResult* ce_check_f64(CEEngine* engine, const double* values, int count);

/* === Eisenstein-Specific === */

/**
 * Check Eisenstein integers against disk bounds (norm ≤ r²).
 *
 * For each pair (a[i], b[i]), computes norm = a² + ab + b²
 * and checks norm ≤ radius_squared.
 *
 * @param engine          Engine handle (must be in CE_MODE_EISENSTEIN or auto-detects)
 * @param a               Real components (host array)
 * @param b               Eisenstein omega components (host array)
 * @param count           Number of integer pairs
 * @param radius_squared  Radius squared (r²) for the disk bound
 * @return Result handle, or NULL on error
 */
CEResult* ce_check_eisenstein(CEEngine* engine, const int32_t* a, const int32_t* b,
                               int count, int32_t radius_squared);

/**
 * Hot-swap bounds without re-uploading structure.
 * Updates the GPU bounds in-place via async memcpy.
 * Thread-safe: can be called while checks are in flight on other streams.
 *
 * @return CE_OK on success
 */
int ce_update_bounds(CEEngine* engine, const void* lo, const void* hi, int count);

/* === Async API === */

/**
 * Create an async stream for concurrent constraint checking.
 * Streams can run checks in parallel on the same engine.
 */
CEStream* ce_stream_create(CEEngine* engine);

/**
 * Async check INT8 values. Non-blocking; use ce_stream_result() to wait.
 */
int ce_stream_check_i8(CEStream* stream, const int8_t* values, int count);

/**
 * Async check INT32 values. Non-blocking; use ce_stream_result() to wait.
 */
int ce_stream_check_i32(CEStream* stream, const int32_t* values, int count);

/**
 * Async check Eisenstein integers. Non-blocking.
 */
int ce_stream_check_eisenstein(CEStream* stream, const int32_t* a, const int32_t* b,
                                int count, int32_t radius_squared);

/**
 * Block until the last check on this stream completes, then return result.
 * Caller must free with ce_result_destroy().
 */
CEResult* ce_stream_result(CEStream* stream);

/**
 * Destroy an async stream. Blocks until any in-flight work completes.
 */
void ce_stream_destroy(CEStream* stream);

/* === Result Access === */

/** Number of constraint violations found. */
int ce_result_violation_count(const CEResult* result);

/**
 * Violation mask — bit i is set if constraint i was violated.
 * Mask length is ceil(count/64) uint64_t values.
 * Valid until ce_result_destroy() is called.
 */
const uint64_t* ce_result_violation_mask(const CEResult* result);

/** Throughput in constraints/second for this check. */
double ce_result_throughput(const CEResult* result);

/** Wall-clock latency in milliseconds for this check (including H2D + kernel + D2H). */
double ce_result_latency_ms(const CEResult* result);

/** Number of constraints checked. */
int ce_result_count(const CEResult* result);

/** Free result resources. Safe to call with NULL. */
void ce_result_destroy(CEResult* result);

/* === Statistics === */

/** Cumulative engine statistics. */
typedef struct CEStats {
    double throughput_avg;       /**< Running average throughput (constraints/sec) */
    double latency_avg_ms;      /**< Running average latency (ms) */
    double latency_p99_ms;      /**< P99 latency (ms) */
    int64_t total_checks;       /**< Total constraint checks performed */
    int64_t total_violations;   /**< Total violations found */
    size_t gpu_memory_used;     /**< Current GPU memory allocated by engine */
    size_t gpu_memory_total;    /**< Total GPU memory on device */
    int device_id;              /**< CUDA device index in use */
    int compute_capability;     /**< Compute capability (e.g., 86 for sm_86) */
} CEStats;

/**
 * Get cumulative engine statistics. Pointer valid for engine lifetime.
 */
const CEStats* ce_get_stats(const CEEngine* engine);

/**
 * Reset cumulative statistics to zero.
 */
void ce_reset_stats(CEEngine* engine);

/* === Device Query === */

/**
 * Get number of available CUDA devices.
 */
int ce_device_count(void);

/**
 * Get device name string. Caller must NOT free.
 * Valid until next ce_device_name() call.
 */
const char* ce_device_name(int device_id);

#ifdef __cplusplus
}
#endif

#endif /* CONSTRAINT_ENGINE_H */
