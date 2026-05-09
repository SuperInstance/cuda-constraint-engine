/**
 * quickstart.cu — 10-line "hello constraints"
 *
 * The simplest possible usage of cuda-constraint-engine.
 * Compile: make examples/quickstart
 * Run:     LD_LIBRARY_PATH=. ./examples/quickstart
 */

#include "constraint_engine.h"
#include <cstdio>
#include <cstdlib>

int main() {
    /* 1. Create engine (auto-select GPU) */
    CEEngine* engine = ce_create(-1);
    if (!engine) { fprintf(stderr, "Failed to create engine\n"); return 1; }
    printf("Device: %s\n", ce_device_name(0));

    /* 2. Define bounds: values must be in [10, 50] */
    int32_t lo[] = {10, 10, 10, 10, 10};
    int32_t hi[] = {50, 50, 50, 50, 50};
    ce_upload_bounds_i32(engine, lo, hi, 5);

    /* 3. Check some values */
    int32_t values[] = {15, 99, 25, -1, 40};  /* 99 and -1 violate */
    CEResult* result = ce_check_i32(engine, values, 5);

    /* 4. Print results */
    printf("Checked %d constraints\n", ce_result_count(result));
    printf("Violations: %d\n", ce_result_violation_count(result));
    printf("Latency: %.3f ms\n", ce_result_latency_ms(result));
    printf("Throughput: %.0f constraints/sec\n", ce_result_throughput(result));

    const uint64_t* mask = ce_result_violation_mask(result);
    for (int i = 0; i < 5; i++) {
        bool violated = (mask[i / 64] >> (i % 64)) & 1;
        printf("  value[%d] = %d → %s\n", i, values[i], violated ? "VIOLATED" : "OK");
    }

    /* 5. Cleanup */
    ce_result_destroy(result);
    ce_destroy(engine);
    printf("Done!\n");
    return 0;
}
