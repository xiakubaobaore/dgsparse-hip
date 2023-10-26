#ifndef CSR2CSC_H
#define CSR2CSC_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>

#include "hip_util.hpp"

void csr2cscKernel(int m, int n, int nnz, int devid, int *csrRowPtr,
                   int *csrColInd, float *csrVal, int *cscColPtr,
                   int *cscRowInd, float *cscVal) {
  hipsparseHandle_t handle;
  checkHipError(hipSetDevice(devid));
  checkHipSparseError(hipsparseCreate(&handle));
  size_t bufferSize = 0;
  void *buffer = NULL;
  checkHipSparseError(hipsparseCsr2cscEx2_bufferSize(
      handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
      cscRowInd, HIP_R_32F, HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_INDEX_BASE_ZERO,
      HIPSPARSE_CSR2CSC_ALG1, &bufferSize));
  checkHipError(hipMalloc((void **)&buffer, bufferSize));
  checkHipSparseError(hipsparseCsr2cscEx2(
      handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
      cscRowInd, HIP_R_32F, HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_INDEX_BASE_ZERO,
      HIPSPARSE_CSR2CSC_ALG1, buffer));
  checkHipError(hipFree(buffer));
}

#endif
