"""Test mixed index dtype support for CSR/CSC sparse matrices."""
import cupy
import numpy as np
from cupyx.scipy.sparse import csr_matrix

print("=== Mixed Index Dtype Tests ===\n")

# Create test data
data = cupy.array([1, 2, 3, 4, 5, 6, 7], dtype=cupy.float32)

def test_combo(indptr_dtype, indices_dtype, label):
    print(f"\n--- {label}: indptr={indptr_dtype.__name__}, "
          f"indices={indices_dtype.__name__} ---")

    indices = cupy.array([0, 2, 2, 0, 1, 2, 3], dtype=indices_dtype)
    indptr = cupy.array([0, 2, 3, 6, 7], dtype=indptr_dtype)

    # Test 1: Construction
    try:
        m = csr_matrix((data, indices, indptr), shape=(4, 4))
        print(f"  Construction: OK (indptr={m.indptr.dtype}, "
              f"indices={m.indices.dtype})")
        assert m.indptr.dtype == indptr_dtype
        assert m.indices.dtype == indices_dtype
    except Exception as e:
        print(f"  Construction: FAIL - {e}")
        return

    # Test 2: idx_dtype property
    try:
        print(f"  idx_dtype: {m.idx_dtype}")
    except Exception as e:
        print(f"  idx_dtype: FAIL - {e}")

    # Test 3: has_sorted_indices
    try:
        val = m.has_sorted_indices
        print(f"  has_sorted_indices: {val}")
    except Exception as e:
        print(f"  has_sorted_indices: FAIL - {e}")

    # Test 4: has_canonical_format
    try:
        val = m.has_canonical_format
        print(f"  has_canonical_format: {val}")
    except Exception as e:
        print(f"  has_canonical_format: FAIL - {e}")

    # Test 5: dot(vector) — SpMV
    try:
        x = cupy.ones(4, dtype=cupy.float32)
        result = m.dot(x)
        expected = cupy.array([3, 3, 15, 7], dtype=cupy.float32)
        assert cupy.allclose(result, expected)
        print(f"  dot(vector): OK result={result}")
    except Exception as e:
        print(f"  dot(vector): FAIL - {e}")

    # Test 6: dot(matrix) — SpMM
    try:
        B = cupy.eye(4, dtype=cupy.float32)
        result = m.dot(B)
        print(f"  dot(matrix): OK shape={result.shape}")
    except Exception as e:
        print(f"  dot(matrix): FAIL - {e}")

    # Test 7: toarray
    try:
        dense = m.toarray()
        print(f"  toarray: OK shape={dense.shape}")
    except Exception as e:
        print(f"  toarray: FAIL - {e}")

    # Test 8: row slice
    try:
        sub = m[1:3, :]
        print(f"  row slice: OK shape={sub.shape}, "
              f"indptr={sub.indptr.dtype}, indices={sub.indices.dtype}")
    except Exception as e:
        print(f"  row slice: FAIL - {e}")

    # Test 9: scalar index
    try:
        val = m[2, 1]
        print(f"  scalar index m[2,1]: OK val={float(val)}")
    except Exception as e:
        print(f"  scalar index: FAIL - {e}")

    # Test 10: fancy indexing
    try:
        rows = cupy.array([0, 2])
        sub = m[rows, :]
        print(f"  fancy row index: OK shape={sub.shape}")
    except Exception as e:
        print(f"  fancy row index: FAIL - {e}")

    # Test 11: sum
    try:
        s = m.sum(axis=1)
        print(f"  sum(axis=1): OK = {s.T}")
    except Exception as e:
        print(f"  sum(axis=1): FAIL - {e}")

    # Test 12: max
    try:
        mx = m.max(axis=1)
        print(f"  max(axis=1): OK = {mx.toarray().T}")
    except Exception as e:
        print(f"  max(axis=1): FAIL - {e}")

    # Test 13: argmax
    try:
        amx = m.argmax(axis=1)
        print(f"  argmax(axis=1): OK = {amx.T}")
    except Exception as e:
        print(f"  argmax(axis=1): FAIL - {e}")

    # Test 14: transpose
    try:
        mt = m.T
        print(f"  transpose: OK shape={mt.shape}, "
              f"indptr={mt.indptr.dtype}, indices={mt.indices.dtype}")
    except Exception as e:
        print(f"  transpose: FAIL - {e}")

    # Test 15: tocsc
    try:
        csc = m.tocsc()
        print(f"  tocsc: OK shape={csc.shape}, "
              f"indptr={csc.indptr.dtype}, indices={csc.indices.dtype}")
    except Exception as e:
        print(f"  tocsc: FAIL - {e}")

    # Test 16: copy
    try:
        c = m.copy()
        assert c.indptr.dtype == m.indptr.dtype
        assert c.indices.dtype == m.indices.dtype
        print(f"  copy: OK (dtypes preserved)")
    except Exception as e:
        print(f"  copy: FAIL - {e}")

    # Test 17: m @ m.T (SpGEMM)
    try:
        result = m @ m.T
        print(f"  m @ m.T: OK shape={result.shape}, "
              f"indptr={result.indptr.dtype}, indices={result.indices.dtype}")
    except Exception as e:
        print(f"  m @ m.T: FAIL - {e}")

    # Test 18: minor index fancy (column select)
    try:
        cols = cupy.array([0, 2], dtype=indices_dtype)
        sub = m[:, cols]
        print(f"  col fancy index: OK shape={sub.shape}")
    except Exception as e:
        print(f"  col fancy index: FAIL - {e}")

    # Test 19: _from_parts
    try:
        m2 = csr_matrix._from_parts(
            data.copy(), indices.copy(), indptr.copy(), (4, 4))
        print(f"  _from_parts: OK (indptr={m2.indptr.dtype}, "
              f"indices={m2.indices.dtype})")
    except Exception as e:
        print(f"  _from_parts: FAIL - {e}")

    # Test 20: _empty_like
    try:
        e = m._empty_like((2, 4))
        assert e.indptr.dtype == m.indptr.dtype
        assert e.indices.dtype == m.indices.dtype
        print(f"  _empty_like: OK (dtypes preserved)")
    except Exception as e:
        print(f"  _empty_like: FAIL - {e}")


# Test uniform (baseline)
test_combo(cupy.int32, cupy.int32, "Baseline int32+int32")

# Test mixed combos
test_combo(cupy.int64, cupy.int32, "Mixed int64+int32")
test_combo(cupy.int64, cupy.uint16, "Mixed int64+uint16")
test_combo(cupy.int32, cupy.uint16, "Mixed int32+uint16")

print("\n=== Done ===")
