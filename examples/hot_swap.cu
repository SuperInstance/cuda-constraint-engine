/**
 * hot_swap.cu — Hot-swap bounds without re-uploading structure
 *
 * Demonstrates updating bounds in-place via async memcpy,
 * then immediately re-checking with the same values.
 */

#include "constraint_engine.h"
#include <cstdio>
#include <cstdlib>

int main() {
    CEEngine* engine = ce_create(-1);
    if (!engine) return 1;
    printf("Device: %s\n", ce_device_name(0));

    /* Initial bounds: [0, 100] */
    int32_t lo[] = {0, 0, 0, 0, 0};
    int32_t hi[] = {100, 100, 100, 100, 100};
    ce_upload_bounds_i32(engine, lo, hi, 5);

    int32_t values[] = {50, 75, 99, 100, 101};  /* 101 violates */

    /* Check with original bounds */
    CEResult* r1 = ce_check_i32(engine, values, 5);
    printf("Bounds [0,100]: %d violations\n", ce_result_violation_count(r1));
    ce_result_destroy(r1);

    /* Hot-swap to tighter bounds: [0, 50] */
    int32_t new_hi[] = {50, 50, 50, 50, 50};
    ce_update_bounds(engine, lo, new_hi, 5);

    /* Re-check same values — now 75, 99, 100, 101 all violate */
    CEResult* r2 = ce_check_i32(engine, values, 5);
    printf("Bounds [0,50]:  %d violations\n", ce_result_violation_count(r2));
    ce_result_destroy(r2);

    /* Hot-swap again: [-10, 200] — all pass */
    int32_t lo2[] = {-10, -10, -10, -10, -10};
    int32_t hi2[] = {200, 200, 200, 200, 200};
    ce_update_bounds(engine, lo2, hi2, 5);

    CEResult* r3 = ce_check_i32(engine, values, 5);
    printf("Bounds [-10,200]: %d violations\n", ce_result_violation_count(r3));
    ce_result_destroy(r3);

    ce_destroy(engine);
    printf("Hot-swap demo complete!\n");
    return 0;
}
