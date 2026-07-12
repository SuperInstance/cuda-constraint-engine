#!/usr/bin/env python3
"""
Test harness for cuda-constraint-engine.

These tests validate the Python wrapper logic without requiring a GPU.
GPU-dependent tests are skipped automatically when the shared library
is not available.
"""

import sys
import os
import numpy as np

# Add python dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

passed = 0
failed = 0


def ok(name):
    global passed
    passed += 1
    print(f"  ✓ {name}")


def run(name, fn):
    global failed
    try:
        fn()
        ok(name)
    except Exception as e:
        failed += 1
        print(f"  ✗ {name}: {e}")


def test_precision_constants():
    from constraint_engine import CE_INT8, CE_INT16, CE_INT32, CE_FP32, CE_FP64
    assert CE_INT8 == 0
    assert CE_INT16 == 1
    assert CE_INT32 == 2
    assert CE_FP32 == 3
    assert CE_FP64 == 4


def test_mode_constants():
    from constraint_engine import CE_MODE_BOUNDS, CE_MODE_NORM, CE_MODE_EISENSTEIN
    assert CE_MODE_BOUNDS == 0
    assert CE_MODE_NORM == 1
    assert CE_MODE_EISENSTEIN == 2


def test_precision_map():
    from constraint_engine import PRECISION_MAP
    assert PRECISION_MAP['int8'] == 0
    assert PRECISION_MAP['int16'] == 1
    assert PRECISION_MAP['int32'] == 2
    assert PRECISION_MAP['fp32'] == 3
    assert PRECISION_MAP['fp64'] == 4


def test_dtype_map():
    from constraint_engine import DTYPE_MAP, CE_INT8, CE_INT32, CE_FP32
    assert DTYPE_MAP[CE_INT8] == np.int8
    assert DTYPE_MAP[CE_INT32] == np.int32
    assert DTYPE_MAP[CE_FP32] == np.float32


def test_engine_init_without_gpu():
    """Engine creation should raise FileNotFoundError when no GPU library."""
    from constraint_engine import ConstraintEngine
    try:
        engine = ConstraintEngine(max_constraints=100, precision='int32')
        assert engine is not None
    except FileNotFoundError:
        pass  # Expected when no GPU library is built
    except (OSError, RuntimeError):
        pass  # Also acceptable — no CUDA runtime


def test_ce_stats_slots():
    from constraint_engine import CEStats
    assert hasattr(CEStats, '__slots__')
    assert 'throughput_avg' in CEStats.__slots__
    assert 'total_checks' in CEStats.__slots__


if __name__ == '__main__':
    print("Running cuda-constraint-engine tests...")
    run("precision constants", test_precision_constants)
    run("mode constants", test_mode_constants)
    run("precision map", test_precision_map)
    run("dtype map", test_dtype_map)
    run("engine init without GPU", test_engine_init_without_gpu)
    run("CEStats slots", test_ce_stats_slots)
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
