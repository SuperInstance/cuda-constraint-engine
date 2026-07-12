# cuda-constraint-engine ⚡

**GPU constraint checking at over a billion constraints/second.**

A developer-facing CUDA library for checking massive batches of constraints on NVIDIA GPUs. Multi-precision (INT8/INT16/INT32/FP32/FP64), Eisenstein integer support, async streams, CUDA graphs, and a zero-dep Python wrapper.

## Quick Start

### C

```c
#include "constraint_engine.h"

int main() {
    // Create engine (auto-selects GPU)
    CEEngine* engine = ce_create(-1);

    // Upload bounds: values must be in [10, 50]
    int32_t lo[] = {10, 10, 10, 10, 10};
    int32_t hi[] = {50, 50, 50, 50, 50};
    ce_upload_bounds_i32(engine, lo, hi, 5);

    // Check values
    int32_t values[] = {15, 99, 25, -1, 40};
    CEResult* result = ce_check_i32(engine, values, 5);

    printf("Violations: %d\n", ce_result_violation_count(result));
    printf("Throughput: %.0f c/s\n", ce_result_throughput(result));

    ce_result_destroy(result);
    ce_destroy(engine);
}
```

### Python

```python
from constraint_engine import ConstraintEngine
import numpy as np

engine = ConstraintEngine(max_constraints=100_000, precision='int32')

values = np.array([15, 99, 25, -1, 40], dtype=np.int32)
lo = np.array([10, 10, 10, 10, 10], dtype=np.int32)
hi = np.array([50, 50, 50, 50, 50], dtype=np.int32)

violations, mask = engine.check(values, lo, hi)
print(f"Violations: {violations}")  # 2 (99 and -1)
print(f"Stats: {engine.stats}")
```

## Build

```bash
make            # Build shared library + examples
make lib        # Build library only
make examples   # Build all examples
make install    # Install to /usr/local (needs sudo)
```

Requires: CUDA Toolkit 11.5+, GCC, `sm_86+` GPU (RTX 30-series / 40-series).

Change GPU architecture: `make ARCH=sm_80`

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Your Code                       │
│  ce_check_i32() / engine.check() / etc.         │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│              CEEngine                            │
│  ┌─────────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Memory Pool │  │ Kernels  │  │ Stream Pool│ │
│  │ (pre-alloc) │  │ (GPU)    │  │ (async)    │ │
│  └─────────────┘  └──────────┘  └────────────┘ │
│                                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │ CUDA Graphs (optional, 18x launch speedup)  │ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

**Key design decisions:**

1. **Memory pool** — All GPU buffers pre-allocated on `ce_create()`. No `cudaMalloc` in the hot path.
2. **CUDA graphs** — Capture the check kernel + memcpy and replay. 18x launch speedup for fixed-size workloads.
3. **Hot-swap bounds** — `cudaMemcpyAsync` updates bounds without touching the check kernel. Thread-safe.
4. **Multi-precision** — Template kernels for INT8, INT16, INT32, FP32, FP64.
5. **Eisenstein** — Specialized kernel computes `a² + ab + b²` and compares to `r²` in one pass.
6. **Thread-safe** — Multiple `CEStream` instances run checks concurrently on separate CUDA streams.

## Performance

### RTX 4050 Laptop (sm_86)

| Precision | Constraints | Latency (ms) | Throughput       |
|-----------|------------|---------------|------------------|
| INT32     | 10M        | 8.5           | 1.17B/sec        |
| INT32     | 1M/stream  | 0.9           | 4.4B/sec (4 streams) |
| Eisenstein| 961        | 0.4           | 2.4M/sec         |

*Benchmarks from actual RTX 4050 Laptop GPU (sm_89). May 2026.*

### CPU Comparison (single-thread, INT32)

| Platform | 10M constraints | Throughput |
|----------|----------------|------------|
| GPU (RTX 4050) | 8.5 ms | 1.17B/sec |
| CPU (est.) | ~40 ms | 250M/sec |
| **Speedup** | **4.7x** | |

## Eisenstein Mode

Eisenstein integers are complex numbers of the form `a + bω` where `ω = e^(2πi/3)`. Their norm is:

```
norm(a + bω) = a² + ab + b²
```

The constraint engine can check whether Eisenstein integers fall within a disk of radius `√(r²)`:

```c
CEResult* r = ce_check_eisenstein(engine, a, b, count, radius_squared);
```

This is useful in:
- **Lattice cryptography** — Point-in-lattice-bounds checking
- **Crystallography** — Hexagonal lattice constraint verification
- **Signal processing** — Eisenstein integer constellation checking

## API Reference

### Engine Lifecycle

| Function | Description |
|----------|-------------|
| `ce_create(device_id)` | Create engine with default config |
| `ce_create_configured(config)` | Create with custom config |
| `ce_destroy(engine)` | Free all GPU resources |
| `ce_get_error(engine)` | Last error message |

### Data Upload

| Function | Description |
|----------|-------------|
| `ce_upload_bounds_i8/i16/i32/f32/f64` | Upload lo/hi bounds to GPU |
| `ce_update_bounds(engine, lo, hi, count)` | Hot-swap bounds (async) |

### Constraint Checking

| Function | Description |
|----------|-------------|
| `ce_check_i8/i16/i32/f32/f64` | Synchronous check |
| `ce_check_eisenstein(engine, a, b, count, r²)` | Eisenstein disk check |

### Async API

| Function | Description |
|----------|-------------|
| `ce_stream_create(engine)` | Create async stream |
| `ce_stream_check_i8/i32(stream, ...)` | Non-blocking check |
| `ce_stream_check_eisenstein(stream, ...)` | Non-blocking Eisenstein |
| `ce_stream_result(stream)` | Block and get result |
| `ce_stream_destroy(stream)` | Free stream |

### Results

| Function | Description |
|----------|-------------|
| `ce_result_violation_count(r)` | Number of violations |
| `ce_result_violation_mask(r)` | Bit mask (bit i = violation at index i) |
| `ce_result_throughput(r)` | Constraints/sec for this check |
| `ce_result_latency_ms(r)` | Wall-clock latency |
| `ce_result_destroy(r)` | Free result |

### Statistics

| Function | Description |
|----------|-------------|
| `ce_get_stats(engine)` | Cumulative stats (throughput, latency, memory) |
| `ce_reset_stats(engine)` | Reset stats to zero |

## File Structure

```
cuda-constraint-engine/
├── include/
│   └── constraint_engine.h     # Public API — include this
├── src/
│   ├── constraint_engine.cu    # Core engine implementation
│   ├── kernels.cu              # CUDA kernels (multi-precision + Eisenstein)
│   ├── kernels.cuh             # Kernel declarations
│   ├── memory_pool.cu          # GPU memory pool manager
│   ├── stream_pool.cu          # CUDA stream pool
│   └── statistics.cu           # Aggregation / stats tracking
├── python/
│   └── constraint_engine.py    # ctypes Python wrapper (zero deps)
├── examples/
│   ├── quickstart.cu           # 5-minute intro
│   ├── batch_check.cu          # 10M constraint benchmark
│   ├── stream_pipeline.cu      # Async multi-stream
│   ├── hot_swap.cu             # Live bounds update
│   └── eisenstein_narrows.cu   # Eisenstein disk demo
├── Makefile
├── LICENSE (MIT)
└── README.md
```

## License

MIT — use it however you want. Attribution appreciated.

---

Built by [SuperInstance](https://github.com/SuperInstance) — constraint-theory specialists.

## Ecosystem

This repo is part of the **SuperInstance** flagship ecosystem — agent-first computation, constraint theory, and self-improving runtimes.

### FLUX Runtime Family

| Repo | Language | Description |
|------|----------|-------------|
| [flux-runtime](https://github.com/SuperInstance/flux-runtime) | Python | Full FLUX runtime: markdown→bytecode, 2037 tests, zero deps |
| [flux-core](https://github.com/SuperInstance/flux-core) | Rust | Register-based bytecode VM, deterministic agent computation |
| [flux-js](https://github.com/SuperInstance/flux-js) | JavaScript | FLUX VM for Node.js and browsers, ~400ns/iter |
| [flux-compiler](https://github.com/SuperInstance/flux-compiler) | Rust/Python | Formal-methods compiler for safety-critical codegen |
| [flux-vm](https://github.com/SuperInstance/flux-vm) | Rust | Stack-based constraint-checking VM, 50 opcodes, Turing-incomplete |

### PLATO Engine Family

| Repo | Language | Description |
|------|----------|-------------|
| [plato-server](https://github.com/SuperInstance/plato-server) | Python | Knowledge tiles, fleet sync via Matrix, HTTP API |
| [plato-engine-block](https://github.com/SuperInstance/plato-engine-block) | Rust | Original room runtime: no_std + alloc, builder pattern |
| [plato-engine-block-c](https://github.com/SuperInstance/plato-engine-block-c) | C99 | Embedded reference: zero heap alloc, bare-metal portable |
| [plato-engine-block-elixir](https://github.com/SuperInstance/plato-engine-block-elixir) | Elixir | BEAM supervision trees, fault tolerance, hot reload |
| [plato-runtime-kernel](https://github.com/SuperInstance/plato-runtime-kernel) | Rust | Spatial model: tensor grid, batons, assertion traps |

### Constraint / Theory Family

| Repo | Language | Description |
|------|----------|-------------|
| [categorical-agents](https://github.com/SuperInstance/categorical-agents) | Rust | Category theory for agent composition (functors, naturality) |
| [cuda-constraint-engine](https://github.com/SuperInstance/cuda-constraint-engine) | CUDA/C | GPU constraint checking at 1B+ constraints/sec |
| [grand-pattern-rs](https://github.com/SuperInstance/grand-pattern-rs) | Rust | Fibonacci dual-direction cellular graph architecture |
| [lau-hodge-theory](https://github.com/SuperInstance/lau-hodge-theory) | Rust | Hodge decomposition, Betti numbers, spectral sequences |
| [ternary-science](https://github.com/SuperInstance/ternary-science) | Rust | Experimental evidence for ternary intelligence, 5 conservation laws |

### Agent / Infrastructure Family

| Repo | Language | Description |
|------|----------|-------------|
| [construct-core](https://github.com/SuperInstance/construct-core) | Rust | Layered trait system: bare-metal → alloc → async agent runtime |
| [crab](https://github.com/SuperInstance/crab) | Bash | Agent shell for repo entry/leave (MUD-room metaphor) |
| [exocortex](https://github.com/SuperInstance/exocortex) | Rust | Persistent cognitive substrate, S3-compatible memory |
| [git-agent](https://github.com/SuperInstance/git-agent) | Python | The repo IS the agent — autonomous lifecycle via Git |
| [capitaine-1](https://github.com/SuperInstance/capitaine-1) | TypeScript | Git-native repo-agent, Cloudflare Workers heartbeat |
| [codespace-edge-rd](https://github.com/SuperInstance/codespace-edge-rd) | Research | Codespace→Edge agent lifecycle and yoke transfer protocols |
| [git-agent-codespace](https://github.com/SuperInstance/git-agent-codespace) | DevContainer | One-click Codespace template for Git-Agent runtimes |

### Registries

| Registry | Package | Install |
|----------|---------|---------|
| **PyPI** | `flux-vm` | `pip install flux-vm` |
| **crates.io** | `fluxvm` | `cargo add fluxvm` |
| **npm** | `flux-js` | `npm install flux-js` *(coming soon)* |

### Philosophy & Architecture

- 📖 [AI-Writings](https://github.com/SuperInstance/AI-Writings) — Philosophy, essays, and design rationale
- 📦 [PACKAGES.md](https://github.com/SuperInstance/SuperInstance/blob/main/PACKAGES.md) — Full package index
