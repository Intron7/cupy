"""Test whether cuSPARSE supports uint16 index types and mixed index dtypes."""
import cupy
import numpy as np

print("=== Testing cuSPARSE uint16 and mixed-dtype index support ===\n")

# Create a simple 4x4 CSR matrix
# [[1, 0, 2, 0],
#  [0, 0, 3, 0],
#  [4, 5, 6, 0],
#  [0, 0, 0, 7]]
data = cupy.array([1, 2, 3, 4, 5, 6, 7], dtype=cupy.float32)
indices_np = np.array([0, 2, 2, 0, 1, 2, 3], dtype=np.int32)
indptr_np = np.array([0, 2, 3, 6, 7], dtype=np.int32)

# Test 1: Normal int32 CSR (baseline)
print("Test 1: int32/int32 CSR (baseline)")
try:
    from cupyx.scipy.sparse import csr_matrix
    m = csr_matrix((data, cupy.array(indices_np), cupy.array(indptr_np)), shape=(4, 4))
    result = m.dot(cupy.ones(4, dtype=cupy.float32))
    print(f"  OK: indices={m.indices.dtype}, indptr={m.indptr.dtype}, result={result}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: Create SpMatDescriptor with uint16 indices directly
print("\nTest 2: cuSPARSE createCsr with uint16 indices")
try:
    from cupy.cuda import cusparse as _cusparse
    from cupy._core import _dtype

    indices_u16 = cupy.array(indices_np, dtype=cupy.uint16)
    indptr_i32 = cupy.array(indptr_np, dtype=cupy.int32)

    rows, cols, nnz = 4, 4, 7
    idx_base = _cusparse.CUSPARSE_INDEX_BASE_ZERO
    cuda_dtype = _dtype.to_cuda_dtype(data.dtype)

    desc = _cusparse.createCsr(
        rows, cols, nnz,
        indptr_i32.data.ptr, indices_u16.data.ptr, data.data.ptr,
        _cusparse.CUSPARSE_INDEX_32I,   # indptr type
        _cusparse.CUSPARSE_INDEX_16U,   # indices type
        idx_base, cuda_dtype)
    print(f"  OK: createCsr succeeded with int32 indptr + uint16 indices")
    _cusparse.destroySpMat(desc)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 3: Create SpMatDescriptor with uint16 indptr
print("\nTest 3: cuSPARSE createCsr with uint16 indptr")
try:
    indptr_u16 = cupy.array(indptr_np, dtype=cupy.uint16)
    indices_i32 = cupy.array(indices_np, dtype=cupy.int32)

    desc = _cusparse.createCsr(
        rows, cols, nnz,
        indptr_u16.data.ptr, indices_i32.data.ptr, data.data.ptr,
        _cusparse.CUSPARSE_INDEX_16U,   # indptr type
        _cusparse.CUSPARSE_INDEX_32I,   # indices type
        idx_base, cuda_dtype)
    print(f"  OK: createCsr succeeded with uint16 indptr + int32 indices")
    _cusparse.destroySpMat(desc)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 4: Both uint16
print("\nTest 4: cuSPARSE createCsr with both uint16")
try:
    indices_u16 = cupy.array(indices_np, dtype=cupy.uint16)
    indptr_u16 = cupy.array(indptr_np, dtype=cupy.uint16)

    desc = _cusparse.createCsr(
        rows, cols, nnz,
        indptr_u16.data.ptr, indices_u16.data.ptr, data.data.ptr,
        _cusparse.CUSPARSE_INDEX_16U,
        _cusparse.CUSPARSE_INDEX_16U,
        idx_base, cuda_dtype)
    print(f"  OK: createCsr succeeded with both uint16")
    _cusparse.destroySpMat(desc)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 5: Mixed int32/int64
print("\nTest 5: cuSPARSE createCsr with int32 indptr + int64 indices")
try:
    indices_i64 = cupy.array(indices_np, dtype=cupy.int64)
    indptr_i32 = cupy.array(indptr_np, dtype=cupy.int32)

    desc = _cusparse.createCsr(
        rows, cols, nnz,
        indptr_i32.data.ptr, indices_i64.data.ptr, data.data.ptr,
        _cusparse.CUSPARSE_INDEX_32I,
        _cusparse.CUSPARSE_INDEX_64I,
        idx_base, cuda_dtype)
    print(f"  OK: createCsr succeeded with int32 indptr + int64 indices")
    _cusparse.destroySpMat(desc)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 6: Actually run SpMV with uint16 indices
print("\nTest 6: SpMV with uint16 indices (the real test)")
try:
    from cupyx import cusparse

    indices_u16 = cupy.array(indices_np, dtype=cupy.uint16)
    indptr_i32 = cupy.array(indptr_np, dtype=cupy.int32)

    # Manually create descriptor and try spmv
    handle = _cusparse.getHandle()

    desc = _cusparse.createCsr(
        rows, cols, nnz,
        indptr_i32.data.ptr, indices_u16.data.ptr, data.data.ptr,
        _cusparse.CUSPARSE_INDEX_32I,
        _cusparse.CUSPARSE_INDEX_16U,
        idx_base, cuda_dtype)

    x = cupy.ones(4, dtype=cupy.float32)
    y = cupy.zeros(4, dtype=cupy.float32)

    x_desc = _cusparse.createDnVec(4, x.data.ptr, cuda_dtype)
    y_desc = _cusparse.createDnVec(4, y.data.ptr, cuda_dtype)

    alpha = cupy.array(1.0, dtype=cupy.float32)
    beta = cupy.array(0.0, dtype=cupy.float32)

    op = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    algo = 0  # CUSPARSE_SPMV_ALG_DEFAULT

    buf_size = _cusparse.spMV_bufferSize(
        handle, op, alpha.data.ptr, desc, x_desc, beta.data.ptr,
        y_desc, cuda_dtype, algo)
    buf = cupy.empty(buf_size, dtype=cupy.uint8)

    _cusparse.spMV(
        handle, op, alpha.data.ptr, desc, x_desc, beta.data.ptr,
        y_desc, cuda_dtype, algo, buf.data.ptr)

    print(f"  OK: SpMV result = {y}")
    expected = cupy.array([3, 3, 15, 7], dtype=cupy.float32)
    assert cupy.allclose(y, expected), f"Expected {expected}, got {y}"
    print(f"  Values correct!")

    _cusparse.destroySpMat(desc)
    _cusparse.destroyDnVec(x_desc)
    _cusparse.destroyDnVec(y_desc)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 7: SpMV with mixed int32 indptr + int64 indices
print("\nTest 7: SpMV with int32 indptr + int64 indices")
try:
    indices_i64 = cupy.array(indices_np, dtype=cupy.int64)
    indptr_i32 = cupy.array(indptr_np, dtype=cupy.int32)

    desc = _cusparse.createCsr(
        rows, cols, nnz,
        indptr_i32.data.ptr, indices_i64.data.ptr, data.data.ptr,
        _cusparse.CUSPARSE_INDEX_32I,
        _cusparse.CUSPARSE_INDEX_64I,
        idx_base, cuda_dtype)

    x = cupy.ones(4, dtype=cupy.float32)
    y = cupy.zeros(4, dtype=cupy.float32)

    x_desc = _cusparse.createDnVec(4, x.data.ptr, cuda_dtype)
    y_desc = _cusparse.createDnVec(4, y.data.ptr, cuda_dtype)

    alpha = cupy.array(1.0, dtype=cupy.float32)
    beta = cupy.array(0.0, dtype=cupy.float32)

    buf_size = _cusparse.spMV_bufferSize(
        handle, op, alpha.data.ptr, desc, x_desc, beta.data.ptr,
        y_desc, cuda_dtype, algo)
    buf = cupy.empty(buf_size, dtype=cupy.uint8)

    _cusparse.spMV(
        handle, op, alpha.data.ptr, desc, x_desc, beta.data.ptr,
        y_desc, cuda_dtype, algo, buf.data.ptr)

    print(f"  OK: SpMV result = {y}")
    expected = cupy.array([3, 3, 15, 7], dtype=cupy.float32)
    assert cupy.allclose(y, expected), f"Expected {expected}, got {y}"
    print(f"  Values correct!")

    _cusparse.destroySpMat(desc)
    _cusparse.destroyDnVec(x_desc)
    _cusparse.destroyDnVec(y_desc)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print("\n=== Done ===")
