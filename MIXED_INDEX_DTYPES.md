# CuPy Sparse Mixed-Dtype Index Support

**Date**: 2026-04-07  
**Status**: Planning phase  
**Branch**: Update-subsetting-int64  

---

## Confirmed Facts (cuSPARSE team, #swdl-cusparse, Feb 2026)

- cuSPARSE does **NOT** currently support mixed index types for any operation
- Error at descriptor creation: `"different index sizes for 'csrRowOffsetsType' (64I) and 'csrColIndType' (32I) is not currently supported"`
- Confirmed by Quang Anh Pham: *"We haven't added support for different index types to any cuSPARSE routine."*
- **CTK 13.3** will add mixed index support for **SpMV CSR only** (confirmed by Gunja Pandav)
- SpMM, Csr2cscEx2, SparseToDense/DenseToSparse — requested, nothing committed beyond SpMV
- uint16 indices — requested but nothing committed; `CUSPARSE_INDEX_16U` enum exists but is non-functional for compute ops

---

## Target Dtype Combinations

For rapids-singlecell (1M+ cells × 30k-60k genes, 1-3B+ nnz):

| Combo | indptr | indices | When |
|-------|--------|---------|------|
| int64 + int32 | nnz > 2^31 | ncols < 2^31 | Large datasets |
| int64 + uint16 | nnz > 2^31 | ncols < 65k | Very common in scRNA-seq |
| int32 + uint16 | nnz < 2^31 | ncols < 65k | Medium datasets |

### Memory Impact (1.3M cells × 28k genes, 2.6B nnz)

| Component | uniform int64 | int64 + int32 | int64 + uint16 |
|-----------|--------------|---------------|----------------|
| indptr | 10.4 MB | 10.4 MB | 10.4 MB |
| indices | 19.8 GB | 9.9 GB | 4.95 GB |
| data (f32) | 9.9 GB | 9.9 GB | 9.9 GB |
| **total** | **~29.7 GB** | **~19.8 GB** | **~14.9 GB** |

---

## RSC Operations Needing Mixed-Type Support

| Operation | RSC Usage | cuSPARSE Status | Fallback Strategy |
|-----------|-----------|-----------------|-------------------|
| SpMV (CSR) | Throughout CuPy sparse | CTK 13.3 native | Upcast until 13.3 |
| Csr2cscEx2 | CSR↔CSC conversion | No support planned | Custom kernel or upcast at boundary |
| SpMM (CSR × dense) | PCA: Z = XV | No support planned | Upcast at boundary or custom kernel |
| SparseToDense | Format conversion | No support planned | RSC has custom `_sparse_to_dense` |
| DenseToSparse | Format conversion | No support planned | Custom kernel |
| Custom kernels | normalize, scale, HVG, filter | N/A (not cuSPARSE) | Extend templates for mixed types |

---

## CuPy Index Type Enforcement (5 Layers)

### Layer 1: Constructor — `_compressed.py:315-316`
```python
self.indices = indices.astype(idx_dtype, copy=copy)
self.indptr = indptr.astype(idx_dtype, copy=copy)
```

### Layer 2: `_from_parts()` — `_compressed.py:351-354`
```python
if indices.dtype != indptr.dtype:
    raise ValueError(...)
```

### Layer 3: ElementwiseKernels — `_compressed.py:175-204`
```python
_has_sorted_indices_kern = ElementwiseKernel('raw T indptr, raw T indices', ...)
```
Shared `T` requires matching dtypes. Triggered by `sum_duplicates()`.

### Layer 4: RawModule CUDA templates — `_compressed.py:32-68`
```c
template<typename TI> __global__ void max_reduction(..., TI* x, TI* y, ...)
```
Single `TI` for both indptr and indices.

### Layer 5: cuSPARSE C API
`cusparseCreateCsr()` rejects mixed sizes (until CTK 13.3 SpMV).

---

## Implementation Plan

### Phase 1: Mixed-Type Storage Layer
- Split `idx_dtype` into `indptr_dtype` + `indices_dtype` in constructor
- Remove/relax `_from_parts()` matching check
- Add `get_indptr_dtype(nnz)` / `get_indices_dtype(minor_dim)` helpers
- Auto-narrowing: choose smallest dtype that fits, separately for each

### Phase 2: CuPy Kernel Updates
- Split `TI` → `TPtr` + `TIdx` in all CUDA kernel templates
- Rewrite ElementwiseKernels to use separate type params (or use RawKernel)
- Update `_index.py` kernels similarly

### Phase 3: cuSPARSE Call Boundary
- For operations with cuSPARSE mixed support (SpMV in CTK 13.3): pass through
- For operations without: upcast narrower index to wider at call boundary, call cuSPARSE, discard temp
- `SpMatDescriptor.create()` already passes separate types — no change needed

### Phase 4: Custom Fallback Kernels (priority order)
1. CSR↔CSC conversion — custom kernel or upcast boundary
2. SpMM — upcast at boundary (or custom for CSR × tall-skinny dense)
3. SparseToDense/DenseToSparse — RSC already has custom kernels to extend
4. SpMV — native in CTK 13.3; upcast boundary until then

### Phase 5: Testing & Benchmarking
- Test matrix generator matching RSC workloads
- Correctness: compare mixed-type results vs uniform int64
- Memory: validate savings on real workloads
- Performance: measure upcast overhead at call boundaries

---

## Test Files
- `test_mixed_idx.cu` — C API test (nvcc -lcusparse)
- `test_mixed_idx_cupy.py` — CuPy bypass test
- `test_uint16_cusparse.py` — uint16 investigation
- `test_uint16_cusparse2.py` — uint16 SpMV test
