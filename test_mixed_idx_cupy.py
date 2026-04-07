"""
CuPy-level test: bypass dtype unification and test mixed indptr/indices dtypes.

This constructs a CSR matrix with int64 indptr + int32 indices by monkey-patching
past CuPy's validation, then tests which operations work.
"""
import cupy
import numpy as np
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse._compressed import _compressed_sparse_matrix

print("=== CuPy-level mixed index dtype test ===\n")

# Build a valid CSR matrix first, then swap dtypes
data = cupy.array([1, 2, 3, 4, 5, 6, 7], dtype=cupy.float32)
indices_i32 = cupy.array([0, 2, 2, 0, 1, 2, 3], dtype=cupy.int32)
indptr_i64 = cupy.array([0, 2, 3, 6, 7], dtype=cupy.int64)

# --- Method: Bypass _from_parts validation by directly setting attributes ---
print("Constructing CSR with int64 indptr + int32 indices (bypass validation)...")
try:
    # First create a normal matrix
    m = csr_matrix((data, indices_i32.astype(cupy.int32),
                     indptr_i64.astype(cupy.int32)), shape=(4, 4))
    # Now forcibly replace indptr with int64 version
    m.indptr = indptr_i64
    print(f"  OK: indices.dtype={m.indices.dtype}, indptr.dtype={m.indptr.dtype}")
    print(f"  shape={m.shape}, nnz={m.nnz}")
except Exception as e:
    print(f"  Construction FAIL: {type(e).__name__}: {e}")
    import sys; sys.exit(1)

# Test 1: .dot() (SpMV path)
print("\nTest 1: m.dot(vector) — SpMV")
try:
    x = cupy.ones(4, dtype=cupy.float32)
    result = m.dot(x)
    print(f"  OK: result={result}")
    expected = cupy.array([3, 3, 15, 7], dtype=cupy.float32)
    assert cupy.allclose(result, expected)
    print(f"  Values correct!")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 2: .dot() matrix (SpMM path)
print("\nTest 2: m.dot(matrix) — SpMM")
try:
    B = cupy.eye(4, dtype=cupy.float32)
    result = m.dot(B)
    print(f"  OK: result type={type(result)}")
    if hasattr(result, 'toarray'):
        print(f"  result=\n{result.toarray()}")
    else:
        print(f"  result=\n{result}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 3: .toarray() (SparseToDense)
print("\nTest 3: m.toarray() — SparseToDense")
try:
    dense = m.toarray()
    print(f"  OK: dense=\n{dense}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 4: todense()
print("\nTest 4: m.todense()")
try:
    dense = m.todense()
    print(f"  OK: dense=\n{dense}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 5: Slicing
print("\nTest 5: m[1:3, :] — row slice")
try:
    sub = m[1:3, :]
    print(f"  OK: shape={sub.shape}, nnz={sub.nnz}")
    print(f"  indices.dtype={sub.indices.dtype}, indptr.dtype={sub.indptr.dtype}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 6: Scalar indexing
print("\nTest 6: m[2, 1] — scalar index")
try:
    val = m[2, 1]
    print(f"  OK: val={val}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 7: Matrix multiply (@ operator, SpGEMM)
print("\nTest 7: m @ m.T — SpGEMM")
try:
    result = m @ m.T
    print(f"  OK: shape={result.shape}, nnz={result.nnz}")
    print(f"  indices.dtype={result.indices.dtype}, indptr.dtype={result.indptr.dtype}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 8: Sum along axis
print("\nTest 8: m.sum(axis=1)")
try:
    s = m.sum(axis=1)
    print(f"  OK: sum={s.T}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 9: max/min
print("\nTest 9: m.max(axis=1)")
try:
    mx = m.max(axis=1)
    print(f"  OK: max={mx.toarray().T}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 10: Transpose
print("\nTest 10: m.T")
try:
    mt = m.T
    print(f"  OK: shape={mt.shape}")
    print(f"  indices.dtype={mt.indices.dtype}, indptr.dtype={mt.indptr.dtype}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 11: tocsc
print("\nTest 11: m.tocsc()")
try:
    csc = m.tocsc()
    print(f"  OK: shape={csc.shape}")
    print(f"  indices.dtype={csc.indices.dtype}, indptr.dtype={csc.indptr.dtype}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

# Test 12: Copy
print("\nTest 12: m.copy()")
try:
    c = m.copy()
    print(f"  OK: indices.dtype={c.indices.dtype}, indptr.dtype={c.indptr.dtype}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print("\n=== Done ===")
