
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>

using namespace std;

/*
    test Matrix
    A:
        [
            [1.0, 0.0, 2.0, 3.0],
            [0.0, 4.0, 0.0, 0.0],
            [5.0, 0.0, 6.0, 7.0],
            [0.0, 8.0, 0.0, 9.0],
        ]

    B:
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
            [4, 5, 6],
        ]
    C:
        [
            [162, 195, 228],
            [160, 200, 240],
            [498, 615, 732],
            [356, 445, 534],
        ]

    rowptr = [0, 3, 4, 7, 9]
    col = [0, 2, 3, 1, 0, 2, 3, 1, 3]
    values = [1., 2., 3., 4., 5., 6., 7., 8., 9.]
*/

int main() {
  int m = 4;
  int n = 3;
  int k = 4;
  int nnz = 9;
  float alpha = 1;
  float beta = 0;

  int *h_rowptr = new int[m + 1]{0, 3, 4, 7, 9};
  int *h_col = new int[nnz]{0, 2, 3, 1, 0, 2, 3, 1, 3};
  float *h_val = new float[nnz]{1., 2., 3., 4., 5., 6., 7., 8., 9.};

  // float *h_b = new float[k * n]{10, 40, 70, 4, 20, 50, 80, 5, 30, 60, 90, 6};
  float *h_b = new float[k * n]{10, 20, 30, 40, 50, 60, 70, 80, 90, 4, 5, 6};

  float *h_c = new float[m * n];

  int *d_rowptr;
  int *d_col;
  float *d_val;
  float *d_b;
  float *d_c;
  hipMalloc((void **)&d_rowptr, (m + 1) * sizeof(int));
  hipMalloc((void **)&d_col, (nnz) * sizeof(int));
  hipMalloc((void **)&d_val, (nnz) * sizeof(float));
  hipMalloc((void **)&d_b, (k * n) * sizeof(float));
  hipMalloc((void **)&d_c, (m * n) * sizeof(float));
  hipMemcpy(d_rowptr, h_rowptr, (m + 1) * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_col, h_col, (nnz) * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_val, h_val, (nnz) * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, (k * n) * sizeof(float), hipMemcpyHostToDevice);
  hipMemset((void *)d_c, 0, (m * n) * sizeof(float));

  hipsparseHandle_t handle;
  hipsparseCreate(&handle);
  hipsparseMatDescr_t descr;
  hipsparseCreateMatDescr(&descr);
  hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL);
  hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO);

  hipsparseScsrmm(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, nnz,
                  &alpha, descr, d_val, d_rowptr, d_col, d_b, k, &beta, d_c, m);
  hipMemcpy(h_c, d_c, (m * n) * sizeof(float), hipMemcpyDeviceToHost);
  for (int i = 0; i < m * n; i++) {
    cout << h_c[i] << " ";
  }
  cout << endl;
}
