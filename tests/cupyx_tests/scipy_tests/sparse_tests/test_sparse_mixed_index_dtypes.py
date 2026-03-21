"""Tests for mixed index dtypes (indptr.dtype != indices.dtype).

The mixed-dtype feature allows CSR/CSC matrices to store indptr and indices
with independent dtypes (e.g., int64 indptr + int32 indices, or uint16 + uint16).
This saves memory when nnz > INT32_MAX but the minor dimension fits in a
smaller type.
"""
from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _sputils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mixed_csr(indptr_dtype, indices_dtype, data_dtype='float64'):
    """Build a small 3×4 CSR matrix with explicit index dtypes.

    [[1, 0, 2, 0],
     [0, 0, 3, 0],
     [4, 5, 6, 0]]
    """
    data = cupy.array([1, 2, 3, 4, 5, 6], dtype=data_dtype)
    indices = cupy.array([0, 2, 2, 0, 1, 2], dtype=indices_dtype)
    indptr = cupy.array([0, 2, 3, 6], dtype=indptr_dtype)
    m = sparse.csr_matrix._from_parts(data, indices, indptr, (3, 4))
    return m


def _make_mixed_csc(indptr_dtype, indices_dtype, data_dtype='float64'):
    """Build a small 3×4 CSC matrix with explicit index dtypes.

    [[1, 0, 2, 0],
     [0, 0, 3, 0],
     [4, 5, 6, 0]]
    """
    data = cupy.array([1, 4, 5, 2, 3, 6], dtype=data_dtype)
    indices = cupy.array([0, 2, 2, 0, 1, 2], dtype=indices_dtype)
    indptr = cupy.array([0, 2, 3, 6, 6], dtype=indptr_dtype)
    m = sparse.csc_matrix._from_parts(data, indices, indptr, (3, 4))
    return m


def _to_dense(m):
    """Convert sparse matrix to dense numpy array for comparison."""
    return cupy.asnumpy(m.toarray())


# Reference dense matrix for the helpers above.
_DENSE = numpy.array([
    [1, 0, 2, 0],
    [0, 0, 3, 0],
    [4, 5, 6, 0],
], dtype='float64')


# All mixed-dtype combos to test.
_MIXED_DTYPES = [
    (cupy.int64, cupy.int32),
    (cupy.int32, cupy.int64),
    (cupy.int64, cupy.uint16),
    (cupy.uint16, cupy.uint16),
    (cupy.uint16, cupy.int32),
    (cupy.int32, cupy.uint16),
]


# ---------------------------------------------------------------------------
# 1. Dtype inference helpers
# ---------------------------------------------------------------------------

class TestDtypeInference:
    """get_indptr_dtype / get_indices_dtype pick optimal types."""

    def test_indptr_uint16_small_nnz(self):
        assert _sputils.get_indptr_dtype(100) == cupy.uint16

    def test_indptr_uint16_max(self):
        assert _sputils.get_indptr_dtype(65535) == cupy.uint16

    def test_indptr_int32_above_uint16(self):
        assert _sputils.get_indptr_dtype(65536) == cupy.int32

    def test_indptr_int32_max(self):
        assert _sputils.get_indptr_dtype(2**31 - 1) == cupy.int32

    def test_indptr_int64_above_int32(self):
        assert _sputils.get_indptr_dtype(2**31) == cupy.int64

    def test_indices_uint16_small_dim(self):
        assert _sputils.get_indices_dtype(1000) == cupy.uint16

    def test_indices_uint16_max(self):
        assert _sputils.get_indices_dtype(65535) == cupy.uint16

    def test_indices_int32_above_uint16(self):
        assert _sputils.get_indices_dtype(65536) == cupy.int32

    def test_indices_int32_max(self):
        assert _sputils.get_indices_dtype(2**31 - 1) == cupy.int32

    def test_indices_int64_above_int32(self):
        assert _sputils.get_indices_dtype(2**31) == cupy.int64


# ---------------------------------------------------------------------------
# 2. Construction — mixed dtypes preserved
# ---------------------------------------------------------------------------

class TestMixedConstruction:
    """CSR/CSC constructed via _from_parts preserve separate dtypes."""

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csr_from_parts_preserves_dtypes(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        assert m.indptr.dtype == ptr_dt
        assert m.indices.dtype == idx_dt
        testing.assert_array_equal(m.toarray(), _DENSE)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csc_from_parts_preserves_dtypes(self, ptr_dt, idx_dt):
        m = _make_mixed_csc(ptr_dt, idx_dt)
        assert m.indptr.dtype == ptr_dt
        assert m.indices.dtype == idx_dt
        testing.assert_array_equal(m.toarray(), _DENSE)

    def test_shape_constructor_infers_uint16(self):
        # Small shape → both indptr and indices should be uint16.
        m = sparse.csr_matrix((100, 200))
        assert m.indptr.dtype == cupy.uint16
        assert m.indices.dtype == cupy.uint16

    def test_shape_constructor_large_minor_uses_int32(self):
        # minor dim > 65535 → indices = int32, indptr still uint16 (nnz=0).
        m = sparse.csr_matrix((10, 100_000))
        assert m.indptr.dtype == cupy.uint16
        assert m.indices.dtype == cupy.int32

    def test_tuple3_constructor_preserves_mixed(self):
        # Pass int64 indptr + int32 indices explicitly.
        data = cupy.array([1.0, 2.0])
        indices = cupy.array([0, 3], dtype=cupy.int32)
        indptr = cupy.array([0, 1, 2], dtype=cupy.int64)
        m = sparse.csr_matrix((data, indices, indptr), shape=(2, 10))
        # indptr may be downcast to int32 by check_contents since values
        # fit, but indices should stay int32.
        assert m.indices.dtype == cupy.int32

    def test_copy_from_mixed_csr_preserves_dtypes(self):
        m = _make_mixed_csr(cupy.int64, cupy.uint16)
        m2 = sparse.csr_matrix(m)
        assert m2.indptr.dtype == cupy.int64
        assert m2.indices.dtype == cupy.uint16
        testing.assert_array_equal(m2.toarray(), _DENSE)


# ---------------------------------------------------------------------------
# 3. toarray / todense
# ---------------------------------------------------------------------------

class TestMixedToarray:
    """toarray() works correctly with mixed dtypes."""

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csr_toarray(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        testing.assert_array_equal(m.toarray(), _DENSE)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csc_toarray(self, ptr_dt, idx_dt):
        m = _make_mixed_csc(ptr_dt, idx_dt)
        testing.assert_array_equal(m.toarray(), _DENSE)


# ---------------------------------------------------------------------------
# 4. Format conversion (tocsr / tocsc / tocoo)
# ---------------------------------------------------------------------------

class TestMixedFormatConversion:
    """Format conversions work with mixed dtypes."""

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csr_to_csc(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        c = m.tocsc()
        testing.assert_array_equal(c.toarray(), _DENSE)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csc_to_csr(self, ptr_dt, idx_dt):
        m = _make_mixed_csc(ptr_dt, idx_dt)
        c = m.tocsr()
        testing.assert_array_equal(c.toarray(), _DENSE)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csr_to_coo(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        c = m.tocoo()
        testing.assert_array_equal(c.toarray(), _DENSE)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csc_to_coo(self, ptr_dt, idx_dt):
        m = _make_mixed_csc(ptr_dt, idx_dt)
        c = m.tocoo()
        testing.assert_array_equal(c.toarray(), _DENSE)


# ---------------------------------------------------------------------------
# 5. Transpose
# ---------------------------------------------------------------------------

class TestMixedTranspose:
    """Transpose works with mixed dtypes."""

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csr_transpose(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        t = m.T
        testing.assert_array_equal(t.toarray(), _DENSE.T)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_csc_transpose(self, ptr_dt, idx_dt):
        m = _make_mixed_csc(ptr_dt, idx_dt)
        t = m.T
        testing.assert_array_equal(t.toarray(), _DENSE.T)


# ---------------------------------------------------------------------------
# 6. Arithmetic
# ---------------------------------------------------------------------------

class TestMixedArithmetic:
    """Addition and multiplication between mixed-dtype matrices."""

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_add_same_mixed_dtypes(self, ptr_dt, idx_dt):
        a = _make_mixed_csr(ptr_dt, idx_dt)
        b = _make_mixed_csr(ptr_dt, idx_dt)
        c = a + b
        testing.assert_array_equal(c.toarray(), _DENSE + _DENSE)

    def test_add_different_mixed_dtypes(self):
        a = _make_mixed_csr(cupy.int64, cupy.uint16)
        b = _make_mixed_csr(cupy.int32, cupy.int32)
        c = a + b
        testing.assert_array_equal(c.toarray(), _DENSE + _DENSE)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_scalar_multiply(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        c = m * 3.0
        testing.assert_array_equal(c.toarray(), _DENSE * 3.0)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_matmul(self, ptr_dt, idx_dt):
        a = _make_mixed_csr(ptr_dt, idx_dt)
        b = _make_mixed_csr(ptr_dt, idx_dt).T.tocsr()
        c = a @ b
        expected = _DENSE @ _DENSE.T
        testing.assert_array_almost_equal(c.toarray(), expected)

    def test_matmul_different_mixed_dtypes(self):
        a = _make_mixed_csr(cupy.int64, cupy.uint16)
        b = _make_mixed_csr(cupy.int32, cupy.int32).T.tocsr()
        c = a @ b
        expected = _DENSE @ _DENSE.T
        testing.assert_array_almost_equal(c.toarray(), expected)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_subtract(self, ptr_dt, idx_dt):
        a = _make_mixed_csr(ptr_dt, idx_dt)
        b = _make_mixed_csr(ptr_dt, idx_dt)
        c = a - b
        testing.assert_array_equal(
            c.toarray(), numpy.zeros_like(_DENSE))


# ---------------------------------------------------------------------------
# 7. Reductions
# ---------------------------------------------------------------------------

class TestMixedReductions:
    """sum, min, max, argmin, argmax with mixed dtypes."""

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_sum_no_axis(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        assert int(m.sum()) == int(_DENSE.sum())

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_sum_axis0(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        testing.assert_array_equal(
            m.sum(axis=0), _DENSE.sum(axis=0))

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_sum_axis1(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        testing.assert_array_equal(
            m.sum(axis=1).ravel(), _DENSE.sum(axis=1))

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_max_axis1(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        result = m.max(axis=1).toarray().ravel()
        expected = _DENSE.max(axis=1)
        testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_min_axis1(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        result = m.min(axis=1).toarray().ravel()
        expected = _DENSE.min(axis=1)
        testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_argmax_axis1(self, ptr_dt, idx_dt):
        # Use a matrix without zeros in any row to avoid ambiguity.
        data = cupy.array([1, 2, 3, 4], dtype='float64')
        indices = cupy.array([0, 1, 0, 1], dtype=idx_dt)
        indptr = cupy.array([0, 2, 4], dtype=ptr_dt)
        m = sparse.csr_matrix._from_parts(data, indices, indptr, (2, 2))
        dense = numpy.array([[1, 2], [3, 4]], dtype='float64')
        result = cupy.asnumpy(m.argmax(axis=1)).ravel()
        expected = dense.argmax(axis=1)
        testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_argmin_axis1(self, ptr_dt, idx_dt):
        data = cupy.array([1, 2, 3, 4], dtype='float64')
        indices = cupy.array([0, 1, 0, 1], dtype=idx_dt)
        indptr = cupy.array([0, 2, 4], dtype=ptr_dt)
        m = sparse.csr_matrix._from_parts(data, indices, indptr, (2, 2))
        dense = numpy.array([[1, 2], [3, 4]], dtype='float64')
        result = cupy.asnumpy(m.argmin(axis=1)).ravel()
        expected = dense.argmin(axis=1)
        testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 8. Indexing
# ---------------------------------------------------------------------------

class TestMixedIndexing:
    """Scalar and fancy indexing with mixed dtypes."""

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_scalar_getitem(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        assert float(m[0, 0]) == 1.0
        assert float(m[0, 2]) == 2.0
        assert float(m[2, 1]) == 5.0
        assert float(m[0, 1]) == 0.0

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_row_slice(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        sub = m[1:3]
        testing.assert_array_equal(sub.toarray(), _DENSE[1:3])

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_fancy_row_index(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        rows = cupy.array([0, 2])
        sub = m[rows]
        testing.assert_array_equal(sub.toarray(), _DENSE[[0, 2]])

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_fancy_col_index(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        cols = cupy.array([0, 2])
        sub = m[:, cols]
        testing.assert_array_equal(sub.toarray(), _DENSE[:, [0, 2]])


# ---------------------------------------------------------------------------
# 9. Structural operations
# ---------------------------------------------------------------------------

class TestMixedStructural:
    """eliminate_zeros, sort_indices, has_sorted_indices with mixed dtypes."""

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_eliminate_zeros(self, ptr_dt, idx_dt):
        # Insert an explicit zero and eliminate it.
        data = cupy.array([1, 0, 2, 3], dtype='float64')
        indices = cupy.array([0, 1, 2, 2], dtype=idx_dt)
        indptr = cupy.array([0, 3, 4], dtype=ptr_dt)
        m = sparse.csr_matrix._from_parts(data, indices, indptr, (2, 4))
        m.eliminate_zeros()
        assert m.nnz == 3
        expected = numpy.array([[1, 0, 2, 0], [0, 0, 3, 0]], dtype='float64')
        testing.assert_array_equal(m.toarray(), expected)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_has_sorted_indices(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        assert m.has_sorted_indices

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_has_canonical_format(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        assert m.has_canonical_format

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_sort_indices_unsorted(self, ptr_dt, idx_dt):
        # Build a CSR with unsorted indices within a row.
        data = cupy.array([2, 1, 3], dtype='float64')
        indices = cupy.array([2, 0, 1], dtype=idx_dt)
        indptr = cupy.array([0, 2, 3], dtype=ptr_dt)
        m = sparse.csr_matrix._from_parts(data, indices, indptr, (2, 4))
        m._has_sorted_indices = False
        m.sort_indices()
        assert m.has_sorted_indices
        expected = numpy.array([[1, 0, 2, 0], [0, 3, 0, 0]], dtype='float64')
        testing.assert_array_equal(m.toarray(), expected)


# ---------------------------------------------------------------------------
# 10. Diagonal extraction
# ---------------------------------------------------------------------------

class TestMixedDiagonal:

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_diagonal(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        diag = cupy.asnumpy(m.diagonal())
        expected = numpy.diag(_DENSE)
        testing.assert_array_equal(diag, expected)

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_diagonal_k1(self, ptr_dt, idx_dt):
        m = _make_mixed_csr(ptr_dt, idx_dt)
        diag = cupy.asnumpy(m.diagonal(k=1))
        expected = numpy.diag(_DENSE, k=1)
        testing.assert_array_equal(diag, expected)


# ---------------------------------------------------------------------------
# 11. Real / Imag (complex data, mixed index dtypes)
# ---------------------------------------------------------------------------

class TestMixedComplex:

    @pytest.mark.parametrize('ptr_dt,idx_dt', _MIXED_DTYPES)
    def test_real_imag(self, ptr_dt, idx_dt):
        data = cupy.array([1+2j, 3+4j, 5+6j], dtype='complex128')
        indices = cupy.array([0, 1, 2], dtype=idx_dt)
        indptr = cupy.array([0, 1, 2, 3], dtype=ptr_dt)
        m = sparse.csr_matrix._from_parts(data, indices, indptr, (3, 3))
        testing.assert_array_equal(
            m.real.toarray(),
            numpy.diag([1.0, 3.0, 5.0]))
        testing.assert_array_equal(
            m.imag.toarray(),
            numpy.diag([2.0, 4.0, 6.0]))


# ---------------------------------------------------------------------------
# 12. Memory savings verification
# ---------------------------------------------------------------------------

class TestMemorySavings:
    """Verify that mixed dtypes actually save memory."""

    def test_int64_uint16_vs_uniform_int64(self):
        m_mixed = _make_mixed_csr(cupy.int64, cupy.uint16)
        m_uniform = _make_mixed_csr(cupy.int64, cupy.int64)
        mixed_bytes = m_mixed.indptr.nbytes + m_mixed.indices.nbytes
        uniform_bytes = m_uniform.indptr.nbytes + m_uniform.indices.nbytes
        assert mixed_bytes < uniform_bytes

    def test_uint16_uint16_vs_int32_int32(self):
        m_small = _make_mixed_csr(cupy.uint16, cupy.uint16)
        m_int32 = _make_mixed_csr(cupy.int32, cupy.int32)
        small_bytes = m_small.indptr.nbytes + m_small.indices.nbytes
        int32_bytes = m_int32.indptr.nbytes + m_int32.indices.nbytes
        assert small_bytes < int32_bytes
