/*
 * test_mixed_idx.cu
 * Minimal test: does cuSPARSE support mixed index dtypes for CSR?
 *
 * Tests all 4 combos of (indptr_type, indices_type):
 *   (int32, int32), (int64, int64), (int64, int32), (int32, int64)
 *
 * For each combo, tests:
 *   1. cusparseCreateCsr       — descriptor creation
 *   2. cusparseSpMV            — sparse matrix × dense vector
 *   3. cusparseSpMM            — sparse matrix × dense matrix
 *   4. cusparseSparseToDense   — convert to dense
 *
 * Compile: nvcc -lcusparse test_mixed_idx.cu -o test_mixed_idx
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("  CUDA error %d: %s\n", err, cudaGetErrorString(err)); \
            return;                                                      \
        }                                                                \
    } while (0)

const char* statusName(cusparseStatus_t s) {
    switch (s) {
        case CUSPARSE_STATUS_SUCCESS:                return "SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:        return "NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:           return "ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:          return "INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:          return "ARCH_MISMATCH";
        case CUSPARSE_STATUS_EXECUTION_FAILED:       return "EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:         return "INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_NOT_SUPPORTED:          return "NOT_SUPPORTED";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES: return "INSUFFICIENT_RESOURCES";
        default:                                     return "UNKNOWN";
    }
}

const char* idxName(cusparseIndexType_t t) {
    switch (t) {
        case CUSPARSE_INDEX_16U: return "uint16";
        case CUSPARSE_INDEX_32I: return "int32";
        case CUSPARSE_INDEX_64I: return "int64";
        default:                 return "???";
    }
}

/*
 * Small 4×4 CSR matrix:
 *   [[1, 0, 2, 0],
 *    [0, 0, 3, 0],
 *    [4, 5, 6, 0],
 *    [0, 0, 0, 7]]
 *
 * data    = [1, 2, 3, 4, 5, 6, 7]
 * indices = [0, 2, 2, 0, 1, 2, 3]
 * indptr  = [0, 2, 3, 6, 7]
 */

// Copy indptr/indices to device with the right byte width
template<typename T>
void copyToDevice(T** d_ptr, const int* src, int n) {
    T* h = (T*)malloc(n * sizeof(T));
    for (int i = 0; i < n; i++) h[i] = (T)src[i];
    cudaMalloc(d_ptr, n * sizeof(T));
    cudaMemcpy(*d_ptr, h, n * sizeof(T), cudaMemcpyHostToDevice);
    free(h);
}

void test_combo(cusparseHandle_t handle,
                cusparseIndexType_t indptrType,
                cusparseIndexType_t indicesType) {

    const int rows = 4, cols = 4, nnz = 7;
    int h_indices[] = {0, 2, 2, 0, 1, 2, 3};
    int h_indptr[]  = {0, 2, 3, 6, 7};
    float h_data[]  = {1, 2, 3, 4, 5, 6, 7};

    printf("\n=== indptr=%s, indices=%s ===\n",
           idxName(indptrType), idxName(indicesType));

    // Allocate device data
    float *d_data, *d_x, *d_y, *d_B, *d_C, *d_dense;
    CHECK_CUDA(cudaMalloc(&d_data, nnz * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate indptr with correct type
    void *d_indptr = nullptr, *d_indices = nullptr;
    if (indptrType == CUSPARSE_INDEX_32I) {
        int* p; copyToDevice(&p, h_indptr, rows + 1); d_indptr = p;
    } else {
        long long* p; copyToDevice(&p, h_indptr, rows + 1); d_indptr = p;
    }
    if (indicesType == CUSPARSE_INDEX_32I) {
        int* p; copyToDevice(&p, h_indices, nnz); d_indices = p;
    } else {
        long long* p; copyToDevice(&p, h_indices, nnz); d_indices = p;
    }

    // --- 1. createCsr ---
    cusparseSpMatDescr_t matA;
    cusparseStatus_t st = cusparseCreateCsr(
        &matA, rows, cols, nnz,
        d_indptr, d_indices, d_data,
        indptrType, indicesType,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    printf("  createCsr:       %s\n", statusName(st));
    if (st != CUSPARSE_STATUS_SUCCESS) goto cleanup;

    // --- 2. SpMV ---
    {
        float h_x[] = {1, 1, 1, 1};
        CHECK_CUDA(cudaMalloc(&d_x, cols * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, rows * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_y, 0, rows * sizeof(float)));

        cusparseDnVecDescr_t vecX, vecY;
        cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F);

        float alpha = 1.0f, beta = 0.0f;
        size_t bufSize = 0;

        st = cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
        if (st != CUSPARSE_STATUS_SUCCESS) {
            printf("  SpMV bufSize:    %s\n", statusName(st));
        } else {
            void* buf; cudaMalloc(&buf, bufSize);
            st = cusparseSpMV(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf);
            printf("  SpMV:            %s", statusName(st));
            if (st == CUSPARSE_STATUS_SUCCESS) {
                float h_y[4];
                cudaMemcpy(h_y, d_y, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                printf("  result=[%.0f, %.0f, %.0f, %.0f]", h_y[0], h_y[1], h_y[2], h_y[3]);
            }
            printf("\n");
            cudaFree(buf);
        }
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // --- 3. SpMM ---
    {
        // B = 4x2 ones, C = 4x2 zeros
        float h_B[8]; for (int i = 0; i < 8; i++) h_B[i] = 1.0f;
        CHECK_CUDA(cudaMalloc(&d_B, 8 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, 8 * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, 8 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C, 0, 8 * sizeof(float)));

        cusparseDnMatDescr_t matB, matC;
        cusparseCreateDnMat(&matB, cols, 2, 2, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&matC, rows, 2, 2, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

        float alpha = 1.0f, beta = 0.0f;
        size_t bufSize = 0;

        st = cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize);
        if (st != CUSPARSE_STATUS_SUCCESS) {
            printf("  SpMM bufSize:    %s\n", statusName(st));
        } else {
            void* buf; cudaMalloc(&buf, bufSize);
            st = cusparseSpMM(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
            printf("  SpMM:            %s", statusName(st));
            if (st == CUSPARSE_STATUS_SUCCESS) {
                float h_C[8];
                cudaMemcpy(h_C, d_C, 8 * sizeof(float), cudaMemcpyDeviceToHost);
                printf("  C[0..3]=[%.0f, %.0f, %.0f, %.0f]", h_C[0], h_C[1], h_C[2], h_C[3]);
            }
            printf("\n");
            cudaFree(buf);
        }
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // --- 4. SparseToDense ---
    {
        CHECK_CUDA(cudaMalloc(&d_dense, rows * cols * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_dense, 0, rows * cols * sizeof(float)));

        cusparseDnMatDescr_t matDense;
        cusparseCreateDnMat(&matDense, rows, cols, cols, d_dense,
                            CUDA_R_32F, CUSPARSE_ORDER_ROW);

        size_t bufSize = 0;
        st = cusparseSparseToDense_bufferSize(
            handle, matA, matDense,
            CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufSize);
        if (st != CUSPARSE_STATUS_SUCCESS) {
            printf("  SparseToDense bufSize: %s\n", statusName(st));
        } else {
            void* buf; cudaMalloc(&buf, bufSize);
            st = cusparseSparseToDense(
                handle, matA, matDense,
                CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buf);
            printf("  SparseToDense:   %s", statusName(st));
            if (st == CUSPARSE_STATUS_SUCCESS) {
                float h_dense[16];
                cudaMemcpy(h_dense, d_dense, 16 * sizeof(float), cudaMemcpyDeviceToHost);
                printf("  [0,0]=%.0f [2,1]=%.0f [3,3]=%.0f",
                       h_dense[0], h_dense[2*4+1], h_dense[3*4+3]);
            }
            printf("\n");
            cudaFree(buf);
        }
        cusparseDestroyDnMat(matDense);
        cudaFree(d_dense);
    }

    cusparseDestroySpMat(matA);

cleanup:
    cudaFree(d_data);
    cudaFree(d_indptr);
    cudaFree(d_indices);
}

int main() {
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    int cusparseVer;
    cusparseGetVersion(handle, &cusparseVer);

    printf("CUDA driver: %d.%d, runtime: %d.%d, cuSPARSE: %d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10,
           cusparseVer);

    // Test all 4 combinations
    cusparseIndexType_t types[] = {CUSPARSE_INDEX_32I, CUSPARSE_INDEX_64I};
    const char* names[] = {"int32", "int64"};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            test_combo(handle, types[i], types[j]);
        }
    }

    // Bonus: test uint16 for both
    printf("\n=== BONUS: indptr=uint16, indices=uint16 ===\n");
    {
        const int rows = 4, cols = 4, nnz = 7;
        int h_indices[] = {0, 2, 2, 0, 1, 2, 3};
        int h_indptr[]  = {0, 2, 3, 6, 7};
        float h_data[]  = {1, 2, 3, 4, 5, 6, 7};

        unsigned short *d_indptr_u16, *d_indices_u16;
        float *d_data2;

        unsigned short h_indptr_u16[] = {0, 2, 3, 6, 7};
        unsigned short h_indices_u16[] = {0, 2, 2, 0, 1, 2, 3};

        cudaMalloc(&d_indptr_u16, (rows + 1) * sizeof(unsigned short));
        cudaMalloc(&d_indices_u16, nnz * sizeof(unsigned short));
        cudaMalloc(&d_data2, nnz * sizeof(float));
        cudaMemcpy(d_indptr_u16, h_indptr_u16, (rows+1)*sizeof(unsigned short), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices_u16, h_indices_u16, nnz*sizeof(unsigned short), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data2, h_data, nnz*sizeof(float), cudaMemcpyHostToDevice);

        cusparseSpMatDescr_t matA;
        cusparseStatus_t st = cusparseCreateCsr(
            &matA, rows, cols, nnz,
            d_indptr_u16, d_indices_u16, d_data2,
            CUSPARSE_INDEX_16U, CUSPARSE_INDEX_16U,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        printf("  createCsr:       %s\n", statusName(st));

        if (st == CUSPARSE_STATUS_SUCCESS) {
            // Try SpMV
            float h_x[] = {1,1,1,1};
            float *d_x, *d_y;
            cudaMalloc(&d_x, 4*sizeof(float));
            cudaMalloc(&d_y, 4*sizeof(float));
            cudaMemcpy(d_x, h_x, 4*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_y, 0, 4*sizeof(float));

            cusparseDnVecDescr_t vecX, vecY;
            cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F);
            cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F);

            float alpha=1, beta=0;
            size_t bufSize=0;
            st = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
            if (st != CUSPARSE_STATUS_SUCCESS) {
                printf("  SpMV bufSize:    %s\n", statusName(st));
            } else {
                void* buf; cudaMalloc(&buf, bufSize);
                st = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                    CUSPARSE_SPMV_ALG_DEFAULT, buf);
                printf("  SpMV:            %s\n", statusName(st));
                cudaFree(buf);
            }
            cusparseDestroyDnVec(vecX);
            cusparseDestroyDnVec(vecY);
            cudaFree(d_x);
            cudaFree(d_y);
            cusparseDestroySpMat(matA);
        }
        cudaFree(d_indptr_u16);
        cudaFree(d_indices_u16);
        cudaFree(d_data2);
    }

    cusparseDestroy(handle);
    printf("\nDone.\n");
    return 0;
}
