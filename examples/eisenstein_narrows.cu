/**
 * eisenstein_narrows.cu — The Narrows 3-boat demo on GPU
 *
 * The Eisenstein integers are complex numbers a + bω where ω = e^(2πi/3).
 * Their norm is a² + ab + b² — this kernel checks whether Eisenstein
 * integers fall within a disk of radius √(r²).
 *
 * "The Narrows" demo: three boats passing through a narrow channel,
 * each represented as an Eisenstein integer. The channel is a disk
 * constraint — boats inside the disk are "in the channel."
 */

#include "constraint_engine.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

/* Visualize an Eisenstein integer as a point in the complex plane */
static void print_eisenstein_point(int32_t a, int32_t b, bool violated, int radius_sq) {
    double norm = (double)a * a + (double)a * b + (double)b * b;
    /* Convert to Cartesian: z = a + b*ω, ω = (-1+i√3)/2 */
    double x = a - 0.5 * b;
    double y = b * sqrt(3.0) / 2.0;
    double r = sqrt((double)radius_sq);
    const char* status = violated ? "✗ OUTSIDE" : "✓ INSIDE";
    printf("  (%+4d %+4dω) → (%.1f, %.1f)  norm²=%-6.0f  r²=%d  %s\n",
           a, b, x, y, norm, radius_sq, status);
}

int main() {
    CEEngine* engine = ce_create(-1);
    if (!engine) return 1;
    printf("=== The Narrows — Eisenstein Disk Constraint Demo ===\n");
    printf("Device: %s\n\n", ce_device_name(0));

    /* Channel: disk of radius 10 (r² = 100) */
    int32_t radius_squared = 100;
    printf("Channel constraint: norm(a + bω)² ≤ %d\n\n", radius_squared);

    /* The three boats — Eisenstein integer positions */
    int32_t boat_a[] = {3,   5,   15};   /* Boat 1 near origin, Boat 2 close, Boat 3 far */
    int32_t boat_b[] = {2,  -4,   -10};
    const char* names[] = {"SS Forgemaster", "SS Oracle", "SS Cocapn"};
    int count = 3;

    printf("--- Initial positions ---\n");
    for (int i = 0; i < count; i++) {
        double norm = (double)boat_a[i]*boat_a[i] + (double)boat_a[i]*boat_b[i] + (double)boat_b[i]*boat_b[i];
        print_eisenstein_point(boat_a[i], boat_b[i], norm > radius_squared, radius_squared);
    }

    /* GPU check */
    CEResult* r = ce_check_eisenstein(engine, boat_a, boat_b, count, radius_squared);
    printf("\nGPU check: %d/%d boats in channel\n", count - ce_result_violation_count(r), count);
    printf("Latency: %.3f ms\n", ce_result_latency_ms(r));
    ce_result_destroy(r);

    /* Now check a larger batch — grid of Eisenstein integers */
    printf("\n--- Grid scan: all (a,b) with |a|,|b| ≤ 15 ---\n");
    std::vector<int32_t> grid_a, grid_b;
    for (int a = -15; a <= 15; a++)
        for (int b = -15; b <= 15; b++) {
            grid_a.push_back(a);
            grid_b.push_back(b);
        }
    int grid_count = (int)grid_a.size();
    printf("Scanning %d Eisenstein integers...\n", grid_count);

    CEResult* rg = ce_check_eisenstein(engine, grid_a.data(), grid_b.data(), grid_count, radius_squared);
    printf("Inside disk: %d / %d\n", grid_count - ce_result_violation_count(rg), grid_count);
    printf("Throughput: %.0f Eisenstein checks/sec\n", ce_result_throughput(rg));
    printf("Latency: %.3f ms\n", ce_result_latency_ms(rg));
    ce_result_destroy(rg);

    ce_destroy(engine);
    printf("\n⚓ The Narrows demo complete!\n");
    return 0;
}
