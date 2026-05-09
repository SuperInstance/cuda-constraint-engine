/**
 * batch_check.cu — Batch constraint checking at scale
 *
 * Demonstrates checking millions of constraints in a single call.
 */

#include "constraint_engine.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

int main() {
    const int N = 10'000'000;  /* 10M constraints */

    CEConfig config;
    ce_config_default(&config);
    config.max_constraints = N;
    config.precision = CE_INT32;

    CEEngine* engine = ce_create_configured(&config);
    if (!engine) { fprintf(stderr, "Engine creation failed\n"); return 1; }
    printf("Device: %s\n", ce_device_name(0));
    printf("Checking %d constraints...\n", N);

    /* Generate random bounds and values */
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);

    std::vector<int32_t> lo(N), hi(N), values(N);
    for (int i = 0; i < N; i++) {
        lo[i] = dist(rng);
        hi[i] = lo[i] + 100;  /* bounds window of 100 */
        values[i] = dist(rng);
    }

    ce_upload_bounds_i32(engine, lo.data(), hi.data(), N);
    CEResult* result = ce_check_i32(engine, values.data(), N);

    printf("Violations: %d / %d\n", ce_result_violation_count(result), N);
    printf("Latency: %.3f ms\n", ce_result_latency_ms(result));
    printf("Throughput: %.0f constraints/sec\n", ce_result_throughput(result));

    auto stats = ce_get_stats(engine);
    printf("GPU memory: %.1f MB\n", stats->gpu_memory_used / 1e6);

    ce_result_destroy(result);
    ce_destroy(engine);
    return 0;
}
