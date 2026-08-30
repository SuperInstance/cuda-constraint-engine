/**
 * stream_pool.cu — CUDA stream pool for async operations
 *
 * Pre-created stream pool allows concurrent constraint checks
 * without stream creation overhead.
 */

#include "constraint_engine.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

/* Private to this file, and deliberately NOT named CEStream.
 *
 * `CEStream` is the public opaque handle: constraint_engine.h declares
 * `typedef struct CEStream CEStream;` and never defines it, and the handle a
 * caller actually receives from ce_stream_create() is a CEStream_internal,
 * which has three members. Giving that same tag a second, four-member
 * definition here meant the public type had two different layouts in one
 * program -- and the extra `in_use` sits at an offset past the end of what
 * ce_stream_create() allocates, so any crossing access reads out of bounds.
 *
 * Nothing crosses today. The types were kept apart by accident rather than by
 * design, which is the kind of thing that stops being true quietly. */
struct CEPooledStream {
    CEEngine* engine;
    cudaStream_t stream;
    CEResult* pending_result;
    bool in_use;
};

struct CEStreamPool {
    std::vector<CEPooledStream*> streams;
    int next_free;
};

extern "C" {

/* These are called from constraint_engine.cu via internal API */

void* ce_stream_pool_create(int count) {
    CEStreamPool* pool = new CEStreamPool();
    pool->next_free = 0;
    pool->streams.reserve(count);
    for (int i = 0; i < count; i++) {
        CEPooledStream* s = new CEPooledStream();
        s->stream = nullptr;
        s->pending_result = nullptr;
        s->in_use = false;
        s->engine = nullptr;
        cudaError_t e = cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            fprintf(stderr, "[CE] Failed to create stream %d: %s\n", i, cudaGetErrorString(e));
            s->stream = nullptr;
        }
        pool->streams.push_back(s);
    }
    return pool;
}

void ce_stream_pool_destroy(void* vpool) {
    CEStreamPool* pool = (CEStreamPool*)vpool;
    if (!pool) return;
    for (auto* s : pool->streams) {
        if (s) {
            if (s->stream) cudaStreamDestroy(s->stream);
            delete s;
        }
    }
    delete pool;
}

cudaStream_t ce_stream_get_cuda(void* vpool, int index) {
    CEStreamPool* pool = (CEStreamPool*)vpool;
    if (!pool || index < 0 || index >= (int)pool->streams.size()) return nullptr;
    return pool->streams[index]->stream;
}

int ce_stream_pool_size(void* vpool) {
    CEStreamPool* pool = (CEStreamPool*)vpool;
    return pool ? (int)pool->streams.size() : 0;
}

} /* extern "C" */
