"""
constraint_engine.py — Python wrapper for cuda-constraint-engine (ctypes, zero deps)

Usage:
    from constraint_engine import ConstraintEngine

    engine = ConstraintEngine(max_constraints=100_000, precision='int8')
    violations, mask = engine.check(values, lo, hi)
    print(f"Violations: {violations}")

    # Eisenstein mode
    engine_eis = ConstraintEngine(precision='int32', mode='eisenstein')
    violations, mask = engine_eis.check_eisenstein(a, b, radius_squared=100)
    print(engine.stats)
"""

import ctypes
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# === Constants ===
CE_INT8, CE_INT16, CE_INT32, CE_FP32, CE_FP64 = 0, 1, 2, 3, 4
CE_MODE_BOUNDS, CE_MODE_NORM, CE_MODE_EISENSTEIN = 0, 1, 2

PRECISION_MAP = {
    'int8': CE_INT8, 'int16': CE_INT16, 'int32': CE_INT32,
    'float32': CE_FP32, 'fp32': CE_FP32, 'float64': CE_FP64, 'fp64': CE_FP64,
}

DTYPE_MAP = {
    CE_INT8: np.int8, CE_INT16: np.int16, CE_INT32: np.int32,
    CE_FP32: np.float32, CE_FP64: np.float64,
}


def _find_lib() -> str:
    """Find libconstraint_engine.so"""
    # Check relative to this file first
    here = Path(__file__).parent
    candidates = [
        here / "libconstraint_engine.so",
        here.parent / "libconstraint_engine.so",
        here.parent / "build" / "libconstraint_engine.so",
        Path("/usr/local/lib/libconstraint_engine.so"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"libconstraint_engine.so not found. Searched: {[str(c) for c in candidates]}"
    )


class CEStats:
    """Engine statistics."""
    __slots__ = ('throughput_avg', 'latency_avg_ms', 'latency_p99_ms',
                 'total_checks', 'total_violations', 'gpu_memory_used',
                 'gpu_memory_total', 'device_id', 'compute_capability')

    def __init__(self, ptr):
        # The C struct fields are: double, double, double, int64, int64, size_t, size_t, int, int
        self.throughput_avg = ptr.contents.throughput_avg
        self.latency_avg_ms = ptr.contents.latency_avg_ms
        self.latency_p99_ms = ptr.contents.latency_p99_ms
        self.total_checks = ptr.contents.total_checks
        self.total_violations = ptr.contents.total_violations
        self.gpu_memory_used = ptr.contents.gpu_memory_used
        self.gpu_memory_total = ptr.contents.gpu_memory_total
        self.device_id = ptr.contents.device_id
        self.compute_capability = ptr.contents.compute_capability

    def __repr__(self):
        return (f"CEStats(checks={self.total_checks:,}, violations={self.total_violations:,}, "
                f"throughput={self.throughput_avg:,.0f} c/s, "
                f"latency={self.latency_avg_ms:.3f}ms, "
                f"gpu_mem={self.gpu_memory_used / 1e6:.1f}MB)")


class ConstraintEngine:
    """
    GPU Constraint Checking Engine.

    Args:
        max_constraints: Pre-allocate GPU memory for this many constraints.
        precision: 'int8', 'int16', 'int32', 'float32', 'float64'.
        mode: 'bounds', 'norm', or 'eisenstein'.
        enable_graphs: Use CUDA graphs for fixed-size workloads.
        stream_count: Number of async streams.
        device_id: CUDA device (-1 = auto).
        lib_path: Path to libconstraint_engine.so (auto-detect if None).
    """

    def __init__(self, max_constraints: int = 1_000_000,
                 precision: str = 'int32',
                 mode: str = 'bounds',
                 enable_graphs: bool = False,
                 stream_count: int = 4,
                 device_id: int = -1,
                 lib_path: Optional[str] = None):

        lib_path = lib_path or _find_lib()
        self._lib = ctypes.CDLL(lib_path)

        # Setup C struct types
        self._setup_ctypes()

        prec = PRECISION_MAP.get(precision.lower())
        if prec is None:
            raise ValueError(f"Unknown precision '{precision}'. Use: {list(PRECISION_MAP)}")

        mode_val = {'bounds': CE_MODE_BOUNDS, 'norm': CE_MODE_NORM,
                    'eisenstein': CE_MODE_EISENSTEIN}[mode.lower()]

        # Configure
        config = self._CEConfig()
        config.max_constraints = max_constraints
        config.precision = prec
        config.check_mode = mode_val
        config.enable_graphs = enable_graphs
        config.stream_count = stream_count

        self._engine = self._lib.ce_create_configured(ctypes.byref(config))
        if not self._engine:
            raise RuntimeError("Failed to create constraint engine. Is CUDA available?")

        self._precision = prec
        self._np_dtype = DTYPE_MAP[prec]
        self._ctype = {np.int8: ctypes.c_int8, np.int16: ctypes.c_int16,
                       np.int32: ctypes.c_int32, np.float32: ctypes.c_float,
                       np.float64: ctypes.c_double}[self._np_dtype]

    def _setup_ctypes(self):
        lib = self._lib

        # CEConfig struct
        class CEConfig(ctypes.Structure):
            _fields_ = [
                ("max_constraints", ctypes.c_int),
                ("precision", ctypes.c_int),
                ("check_mode", ctypes.c_int),
                ("enable_graphs", ctypes.c_bool),
                ("stream_count", ctypes.c_int),
            ]
        self._CEConfig = CEConfig

        # Setup function signatures
        lib.ce_create_configured.argtypes = [ctypes.POINTER(CEConfig)]
        lib.ce_create_configured.restype = ctypes.c_void_p

        lib.ce_destroy.argtypes = [ctypes.c_void_p]
        lib.ce_destroy.restype = None

        lib.ce_upload_bounds_i32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                              ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        lib.ce_upload_bounds_i32.restype = ctypes.c_int

        lib.ce_upload_bounds_i8.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8),
                                             ctypes.POINTER(ctypes.c_int8), ctypes.c_int]
        lib.ce_upload_bounds_i8.restype = ctypes.c_int

        lib.ce_upload_bounds_f32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                              ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ce_upload_bounds_f32.restype = ctypes.c_int

        lib.ce_check_i32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        lib.ce_check_i32.restype = ctypes.c_void_p

        lib.ce_check_i8.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int]
        lib.ce_check_i8.restype = ctypes.c_void_p

        lib.ce_check_f32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ce_check_f32.restype = ctypes.c_void_p

        lib.ce_check_eisenstein.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                             ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int32]
        lib.ce_check_eisenstein.restype = ctypes.c_void_p

        lib.ce_result_violation_count.argtypes = [ctypes.c_void_p]
        lib.ce_result_violation_count.restype = ctypes.c_int

        lib.ce_result_violation_mask.argtypes = [ctypes.c_void_p]
        lib.ce_result_violation_mask.restype = ctypes.POINTER(ctypes.c_uint64)

        lib.ce_result_throughput.argtypes = [ctypes.c_void_p]
        lib.ce_result_throughput.restype = ctypes.c_double

        lib.ce_result_latency_ms.argtypes = [ctypes.c_void_p]
        lib.ce_result_latency_ms.restype = ctypes.c_double

        lib.ce_result_destroy.argtypes = [ctypes.c_void_p]
        lib.ce_result_destroy.restype = None

        lib.ce_get_stats.argtypes = [ctypes.c_void_p]
        lib.ce_get_stats.restype = ctypes.c_void_p

        lib.ce_get_error.argtypes = [ctypes.c_void_p]
        lib.ce_get_error.restype = ctypes.c_char_p

    def _to_carray(self, arr, dtype=None):
        """Convert numpy array to ctypes pointer."""
        arr = np.ascontiguousarray(arr, dtype=dtype or self._np_dtype)
        return arr.ctypes.data_as(ctypes.POINTER(self._ctype))

    def check(self, values, lo, hi) -> Tuple[int, np.ndarray]:
        """
        Check values against bounds.

        Args:
            values: Array of values to check.
            lo: Array of lower bounds.
            hi: Array of upper bounds.

        Returns:
            (violation_count, violation_mask_as_bool_array)
        """
        values = np.asarray(values, dtype=self._np_dtype)
        lo = np.asarray(lo, dtype=self._np_dtype)
        hi = np.asarray(hi, dtype=self._np_dtype)
        count = len(values)

        # Upload bounds
        lo_ptr = lo.ctypes.data_as(ctypes.POINTER(self._ctype))
        hi_ptr = hi.ctypes.data_as(ctypes.POINTER(self._ctype))
        upload_fn = {
            CE_INT8: self._lib.ce_upload_bounds_i8,
            CE_INT16: self._lib.ce_upload_bounds_i16,
            CE_INT32: self._lib.ce_upload_bounds_i32,
            CE_FP32: self._lib.ce_upload_bounds_f32,
            CE_FP64: self._lib.ce_upload_bounds_f64,
        }[self._precision]
        rc = upload_fn(self._engine, lo_ptr, hi_ptr, count)
        if rc != 0:
            raise RuntimeError(f"Upload bounds failed: {self._lib.ce_get_error(self._engine).decode()}")

        # Check
        val_ptr = values.ctypes.data_as(ctypes.POINTER(self._ctype))
        check_fn = {
            CE_INT8: self._lib.ce_check_i8,
            CE_INT16: self._lib.ce_check_i16,
            CE_INT32: self._lib.ce_check_i32,
            CE_FP32: self._lib.ce_check_f32,
            CE_FP64: self._lib.ce_check_f64,
        }[self._precision]
        result = check_fn(self._engine, val_ptr, count)
        if not result:
            raise RuntimeError(f"Check failed: {self._lib.ce_get_error(self._engine).decode()}")

        try:
            vcount = self._lib.ce_result_violation_count(result)
            mask_ptr = self._lib.ce_result_violation_mask(result)

            # Convert mask to bool array
            mask_words = (count + 63) // 64
            mask_arr = np.zeros(mask_words, dtype=np.uint64)
            ctypes.memmove(mask_arr.ctypes.data, mask_ptr, mask_words * 8)

            # Unpack to bool
            bool_mask = np.unpackbits(mask_arr.view(np.uint8), bitorder='little')[:count].astype(bool)
            return vcount, bool_mask
        finally:
            self._lib.ce_result_destroy(result)

    def check_eisenstein(self, a, b, radius_squared: int) -> Tuple[int, np.ndarray]:
        """
        Check Eisenstein integers against disk bounds (a² + ab + b² ≤ r²).

        Args:
            a: Real components (int32 array).
            b: Eisenstein omega components (int32 array).
            radius_squared: r² value for the disk bound.

        Returns:
            (violation_count, violation_mask_as_bool_array)
        """
        a = np.ascontiguousarray(a, dtype=np.int32)
        b = np.ascontiguousarray(b, dtype=np.int32)
        count = len(a)

        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        result = self._lib.ce_check_eisenstein(
            self._engine, a_ptr, b_ptr, count, ctypes.c_int32(radius_squared)
        )
        if not result:
            raise RuntimeError(f"Eisenstein check failed: {self._lib.ce_get_error(self._engine).decode()}")

        try:
            vcount = self._lib.ce_result_violation_count(result)
            mask_ptr = self._lib.ce_result_violation_mask(result)

            mask_words = (count + 63) // 64
            mask_arr = np.zeros(mask_words, dtype=np.uint64)
            ctypes.memmove(mask_arr.ctypes.data, mask_ptr, mask_words * 8)
            bool_mask = np.unpackbits(mask_arr.view(np.uint8), bitorder='little')[:count].astype(bool)
            return vcount, bool_mask
        finally:
            self._lib.ce_result_destroy(result)

    @property
    def stats(self) -> dict:
        """Engine statistics as a dict."""
        stats_ptr = self._lib.ce_get_stats(self._engine)
        if not stats_ptr:
            return {}
        # Read the C struct — use ctypes struct definition
        class CEStatsC(ctypes.Structure):
            _fields_ = [
                ("throughput_avg", ctypes.c_double),
                ("latency_avg_ms", ctypes.c_double),
                ("latency_p99_ms", ctypes.c_double),
                ("total_checks", ctypes.c_int64),
                ("total_violations", ctypes.c_int64),
                ("gpu_memory_used", ctypes.c_size_t),
                ("gpu_memory_total", ctypes.c_size_t),
                ("device_id", ctypes.c_int),
                ("compute_capability", ctypes.c_int),
            ]
        stats = ctypes.cast(stats_ptr, ctypes.POINTER(CEStatsC)).contents
        return {
            'throughput_avg': stats.throughput_avg,
            'latency_avg_ms': stats.latency_avg_ms,
            'latency_p99_ms': stats.latency_p99_ms,
            'total_checks': stats.total_checks,
            'total_violations': stats.total_violations,
            'gpu_memory_used': stats.gpu_memory_used,
            'gpu_memory_total': stats.gpu_memory_total,
            'device_id': stats.device_id,
            'compute_capability': stats.compute_capability,
        }

    def __del__(self):
        if hasattr(self, '_engine') and self._engine:
            self._lib.ce_destroy(self._engine)

    def __repr__(self):
        return f"ConstraintEngine(precision={self._precision}, stats={self.stats})"
