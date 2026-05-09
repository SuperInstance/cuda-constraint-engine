/**
 * statistics.cu — Aggregation kernels and stats tracking
 */

#include "constraint_engine.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <cstdio>

extern "C" {

/* Internal stats update — called after each check */
void ce_stats_record(CEStats* stats, double latency_ms, int count, int violations) {
    if (!stats) return;

    double alpha = 0.01; /* EMA smoothing factor */

    /* Update running average */
    if (stats->total_checks == 0) {
        stats->throughput_avg = (latency_ms > 0) ? (count / (latency_ms / 1000.0)) : 0;
        stats->latency_avg_ms = latency_ms;
        stats->latency_p99_ms = latency_ms;
    } else {
        double throughput = (latency_ms > 0) ? (count / (latency_ms / 1000.0)) : 0;
        stats->throughput_avg = stats->throughput_avg * (1.0 - alpha) + throughput * alpha;
        stats->latency_avg_ms = stats->latency_avg_ms * (1.0 - alpha) + latency_ms * alpha;
        /* Approximate P99 with EMA of high-latency events */
        if (latency_ms > stats->latency_p99_ms) {
            stats->latency_p99_ms = stats->latency_p99_ms * 0.95 + latency_ms * 0.05;
        }
    }

    stats->total_checks += count;
    stats->total_violations += violations;
}

} /* extern "C" */
