"""Test whether cuSPARSE SpMV actually works with uint16 indices."""
import cupy
import numpy as np
from cupy.cuda import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device

print("=== Testing cuSPARSE SpMV with uint16 indices ===\n")

# Create a simple 4x4 CSR matrix
data = cupy.array([1, 2, 3, 4, 5, 6, 7], dtype=cupy.float32)
indices_np = np.array([0, 2, 2, 0, 1, 2, 3], dtype=np.uint16)
indptr_np = np.array([0, 2, 3, 6, 7], dtype=np.uint16)

rows, cols, nnz = 4, 4, 7
idx_base = _cusparse.CUSPARSE_INDEX_BASE_ZERO
cuda_dtype = _dtype.to_cuda_dtype(data.dtype)

# Test: SpMV with both uint16
print("Test: SpMV with both uint16 indptr and indices")
try:
    indices_u16 = cupy.array(indices_np, dtype=cupy.uint16)
    indptr_u16 = cupy.array(indptr_np, dtype=cupy.uint16)

    desc = _cusparse.createCsr(
        rows, cols, nnz,
        indptr_u16.data.ptr, indices_u16.data.ptr, data.data.ptr,
        _cusparse.CUSPARSE_INDEX_16U,
        _cusparse.CUSPARSE_INDEX_16U,
        idx_base, cuda_dtype)
    print(f"  createCsr OK")

    x = cupy.ones(4, dtype=cupy.float32)
    y = cupy.zeros(4, dtype=cupy.float32)

    x_desc = _cusparse.createDnVec(4, x.data.ptr, cuda_dtype)
    y_desc = _cusparse.createDnVec(4, y.data.ptr, cuda_dtype)

    alpha = cupy.array(1.0, dtype=cupy.float32)
    beta = cupy.array(0.0, dtype=cupy.float32)

    op = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    algo = 0  # CUSPARSE_SPMV_ALG_DEFAULT

    # Get handle from cusparse module
    handle = _device.get_cusparse_handle()

    buf_size = _cusparse.spMV_bufferSize(
        handle, op, alpha.data.ptr, desc, x_desc, beta.data.ptr,
        y_desc, cuda_dtype, algo)
    print(f"  bufferSize OK: {buf_size}")
    buf = cupy.empty(buf_size, dtype=cupy.uint8)

    _cusparse.spMV(
        handle, op, alpha.data.ptr, desc, x_desc, beta.data.ptr,
        y_desc, cuda_dtype, algo, buf.data.ptr)

    print(f"  SpMV result = {y}")
    expected = cupy.array([3, 3, 15, 7], dtype=cupy.float32)
    assert cupy.allclose(y, expected), f"Expected {expected}, got {y}"
    print(f"  Values CORRECT!")

    _cusparse.destroySpMat(desc)
    _cusparse.destroyDnVec(x_desc)
    _cusparse.destroyDnVec(y_desc)
except Exception as e:
    import traceback
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test: SpMM with both uint16
print("\nTest: SpMM with both uint16")
try:
    indices_u16 = cupy.array(indices_np, dtype=cupy.uint16)
    indptr_u16 = cupy.array(indptr_np, dtype=cupy.uint16)

    desc = _cusparse.createCsr(
        rows, cols, nnz,
        indptr_u16.data.ptr, indices_u16.data.ptr, data.data.ptr,
        _cusparse.CUSPARSE_INDEX_16U,
        _cusparse.CUSPARSE_INDEX_16U,
        idx_base, cuda_dtype)

    B = cupy.eye(4, dtype=cupy.float32)
    C = cupy.zeros((4, 4), dtype=cupy.float32)

    B_desc = _cusparse.createDnMat(4, 4, 4, B.data.ptr, cuda_dtype, _cusparse.CUSPARSE_ORDER_ROW)
    C_desc = _cusparse.createDnMat(4, 4, 4, C.data.ptr, cuda_dtype, _cusparse.CUSPARSE_ORDER_ROW)

    alpha = cupy.array(1.0, dtype=cupy.float32)
    beta = cupy.array(0.0, dtype=cupy.float32)

    op = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    algo = 0

    handle = _device.get_cusparse_handle()

    buf_size = _cusparse.spMM_bufferSize(
        handle, op, op, alpha.data.ptr, desc, B_desc, beta.data.ptr,
        C_desc, cuda_dtype, algo)
    print(f"  bufferSize OK: {buf_size}")
    buf = cupy.empty(max(buf_size, 1), dtype=cupy.uint8)

    _cusparse.spMM(
        handle, op, op, alpha.data.ptr, desc, B_desc, beta.data.ptr,
        C_desc, cuda_dtype, algo, buf.data.ptr)

    print(f"  SpMM result:\n{C}")

    _cusparse.destroySpMat(desc)
    _cusparse.destroyDnMat(B_desc)
    _cusparse.destroyDnMat(C_desc)
except Exception as e:
    import traceback
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

# Summary
print("\n=== Summary ===")
print("cuSPARSE requires indptr and indices to have the SAME index type size.")
print("Mixed sizes (e.g., int32 indptr + int64 indices) are NOT SUPPORTED.")
print("uint16 (CUSPARSE_INDEX_16U) works when BOTH indptr AND indices are uint16.")
print("uint16 max = 65535, so this limits matrix dimensions and nnz to 65535.")
