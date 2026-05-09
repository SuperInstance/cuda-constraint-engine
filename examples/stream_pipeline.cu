/**
 * stream_pipeline.cu — Async streaming with CUDA streams
 *
 * Demonstrates concurrent constraint checks on multiple streams.
 */

#include "constraint_engine.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

int main() {
    const int N = 1'000'000;
    const int STREAMS = 4;

    CEConfig config;
    ce_config_default(&config);
    config.max_constraints = N;
    config.precision = CE_INT32;
    config.stream_count = STREAMS;

    CEEngine* engine = ce_create_configured(&config);
    if (!engine) { fprintf(stderr, "Engine creation failed\n"); return 1; }
    printf("Device: %s\n", ce_device_name(0));
    printf("Running %d concurrent streams, %d constraints each\n", STREAMS, N);

    /* Generate test data */
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);

    std::vector<int32_t> lo(N), hi(N), values(N);
    for (int i = 0; i < N; i++) {
        lo[i] = dist(rng);
        hi[i] = lo[i] + 100;
        values[i] = dist(rng);
    }
    ce_upload_bounds_i32(engine, lo.data(), hi.data(), N);

    /* Create streams and launch concurrent checks */
    CEStream* streams[STREAMS];
    for (int s = 0; s < STREAMS; s++) {
        streams[s] = ce_stream_create(engine);
        ce_stream_check_i32(streams[s], values.data(), N);
        printf("  Stream %d: launched\n", s);
    }

    /* Collect results */
    double total_throughput = 0;
    int total_violations = 0;
    for (int s = 0; s < STREAMS; s++) {
        CEResult* r = ce_stream_result(streams[s]);
        if (r) {
            printf("  Stream %d: %d violations, %.3f ms, %.0f c/s\n",
                   s, ce_result_violation_count(r),
                   ce_result_latency_ms(r), ce_result_throughput(r));
            total_throughput += ce_result_throughput(r);
            total_violations += ce_result_violation_count(r);
            ce_result_destroy(r);
        }
        ce_stream_destroy(streams[s]);
    }

    printf("Total: %d violations, aggregate throughput: %.0f c/s\n",
           total_violations, total_throughput);

    ce_destroy(engine);
    return 0;
}
