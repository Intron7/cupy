"""Benchmark minor-axis fancy indexing for sparse CSR matrices.

Compares int32 (histogram kernel) vs int64 (searchsorted/lexsort) paths
in _minor_index_fancy / _minor_index_fancy_sorted.
"""

import cupy
import cupyx.scipy.sparse as sp
import time
import sys


def make_csr(m, n, density, idx_dtype):
    """Create a CSR matrix with the given index dtype."""
    nnz_approx = int(m * n * density)
    if nnz_approx == 0:
        nnz_approx = 1
    rows = cupy.random.randint(0, m, size=nnz_approx, dtype=cupy.int64)
    cols = cupy.random.randint(0, n, size=nnz_approx, dtype=cupy.int64)
    data = cupy.random.rand(nnz_approx, dtype=cupy.float64)
    coo = sp.coo_matrix((data, (rows, cols)), shape=(m, n))
    csr = coo.tocsr()
    # Force index dtype
    csr.indptr = csr.indptr.astype(idx_dtype)
    csr.indices = csr.indices.astype(idx_dtype)
    return csr


def bench_minor_fancy(mat, idx, n_warmup=3, n_iter=10):
    """Benchmark mat[:, idx] (minor-axis fancy indexing for CSR)."""
    # Warmup
    for _ in range(n_warmup):
        _ = mat[:, idx]
    cupy.cuda.Device().synchronize()

    times = []
    for _ in range(n_iter):
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = mat[:, idx]
        cupy.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_minor_slice(mat, sl, n_warmup=3, n_iter=10):
    """Benchmark mat[:, start:stop] (minor-axis contiguous slice for CSR)."""
    for _ in range(n_warmup):
        _ = mat[:, sl]
    cupy.cuda.Device().synchronize()

    times = []
    for _ in range(n_iter):
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = mat[:, sl]
        cupy.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def fmt_time(times):
    import statistics
    median = statistics.median(times)
    if median < 1e-3:
        return f"{median*1e6:8.1f} us"
    elif median < 1:
        return f"{median*1e3:8.2f} ms"
    else:
        return f"{median:8.3f} s "


def report(label, times):
    import statistics
    median = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    print(f"  {label:40s}  median={fmt_time(times)}  std={fmt_time([std])}")


def main():
    print(f"CuPy {cupy.__version__}")
    print(f"GPU: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"CUDA: {cupy.cuda.runtime.runtimeGetVersion()}")
    print()

    # Configuration matrix
    configs = [
        # (rows, cols, density, n_select, description)
        (10_000,    10_000,   0.01,   100,    "10K x 10K, 1% dense, select 100 cols"),
        (10_000,    10_000,   0.01,   1000,   "10K x 10K, 1% dense, select 1K cols"),
        (10_000,    10_000,   0.01,   5000,   "10K x 10K, 1% dense, select 5K cols"),
        (100_000,   100_000,  0.0001, 100,    "100K x 100K, 0.01% dense, select 100 cols"),
        (100_000,   100_000,  0.0001, 1000,   "100K x 100K, 0.01% dense, select 1K cols"),
        (100_000,   100_000,  0.0001, 10000,  "100K x 100K, 0.01% dense, select 10K cols"),
        (1_000_000, 100_000,  0.00001, 100,   "1M x 100K, 0.001% dense, select 100 cols"),
        (1_000_000, 100_000,  0.00001, 1000,  "1M x 100K, 0.001% dense, select 1K cols"),
        (1_000_000, 100_000,  0.00001, 10000, "1M x 100K, 0.001% dense, select 10K cols"),
        (50_000,    1_000_000, 0.00001, 100,  "50K x 1M, 0.001% dense, select 100 cols"),
        (50_000,    1_000_000, 0.00001, 1000, "50K x 1M, 0.001% dense, select 1K cols"),
    ]

    for rows, cols, density, n_select, desc in configs:
        nnz_approx = int(rows * cols * density)
        print(f"=== {desc} (nnz ~{nnz_approx:,}) ===")

        idx = cupy.random.choice(cols, size=n_select, replace=False)
        idx32 = idx.astype(cupy.int32)
        idx64 = idx.astype(cupy.int64)

        try:
            mat32 = make_csr(rows, cols, density, cupy.int32)
            print(f"  Matrix: {mat32.shape}, nnz={mat32.nnz:,}, idx_dtype=int32")
            times32 = bench_minor_fancy(mat32, idx32)
            report("int32 (histogram kernel)", times32)
        except Exception as e:
            print(f"  int32 FAILED: {e}")
            times32 = None

        try:
            mat64 = make_csr(rows, cols, density, cupy.int64)
            print(f"  Matrix: {mat64.shape}, nnz={mat64.nnz:,}, idx_dtype=int64")
            times64 = bench_minor_fancy(mat64, idx64)
            report("int64 (searchsorted/lexsort)", times64)
        except Exception as e:
            print(f"  int64 FAILED: {e}")
            times64 = None

        if times32 and times64:
            import statistics
            m32 = statistics.median(times32)
            m64 = statistics.median(times64)
            ratio = m64 / m32 if m32 > 0 else float('inf')
            print(f"  {'int64/int32 ratio':40s}  {ratio:.2f}x")

        # Also bench contiguous minor slice
        sl = slice(0, n_select)
        try:
            times_sl32 = bench_minor_slice(mat32, sl)
            report("int32 contiguous slice [:, 0:N]", times_sl32)
        except Exception as e:
            print(f"  int32 slice FAILED: {e}")

        try:
            times_sl64 = bench_minor_slice(mat64, sl)
            report("int64 contiguous slice [:, 0:N]", times_sl64)
        except Exception as e:
            print(f"  int64 slice FAILED: {e}")

        print()


if __name__ == "__main__":
    main()
