#ifndef SPMM_HIP
#define SPMM_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../gspmm.h"
#include "hip_util.hpp"

template <typename Index, typename DType, typename REDUCE, typename COMPUTE>
__global__ void
csrspmm_seqreduce_rowbalance_kernel(const Index nr, const Index feature_size,
                                    const Index rowPtr[], const Index colIdx[],
                                    const DType values[], const DType dnInput[],
                                    DType dnOutput[], Index E[]) {
  Index row_tile = blockDim.y; // 8
  Index subwarp_id = threadIdx.y;
  Index stride = row_tile * gridDim.x; // 8 * (m/8)
  Index row = blockIdx.x * row_tile + subwarp_id;
  Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  // if(row == 0 && v_id == 0){
  //   printf("stride = %d\n", stride);
  //   printf("HIP kernel launch with %d*%d blocks of %d*%d threads\n",
  //   int(gridDim.x), int(gridDim.y), int(blockDim.x), int(blockDim.y));
  // }

  dnInput += v_id;
  dnOutput += v_id;
  E += v_id;
  DType val;
  // DType res = init(REDUCE::Op);
  Index col;
  for (; row < nr; row += stride) {
    DType res = init(REDUCE::Op);
    Index E_k_idx = -1;
    Index start = __ldg(rowPtr + row);
    Index end = __ldg(rowPtr + row + 1);
    if ((end - start) > 0) {
      for (Index p = start; p < end; p++) {
        DType val_pre_red;
        col = __ldg(colIdx + p);
        val = __guard_load_default_one<DType>(values, p);
        val_pre_red = val * __ldg(dnInput + col * feature_size);
        if ((REDUCE::Op == MAX && (res < val_pre_red)) ||
            ((REDUCE::Op == MIN) && (res > val_pre_red))) {
          E_k_idx = col;
        }
        res = REDUCE::reduce(res, val_pre_red);

        // res += val * __ldg(dnInput + col * feature_size);
      }
      if (REDUCE::Op == MEAN) {
        res /= (end - start);
      }
    } else {
      res = 0;
    }
    dnOutput[row * feature_size] = res;
    // dnOutput[row * feature_size] = 1000;
    E[row * feature_size] = E_k_idx;
  }
}

__global__ void csrspmm_seqreduce_rowbalance_kernel_without_template(
    const int nr, const int feature_size, const int rowPtr[],
    const int colIdx[], const float values[], const float dnInput[],
    float dnOutput[], int E[]) {
  int row_tile = blockDim.y; // 8
  int subwarp_id = threadIdx.y;
  int stride = row_tile * gridDim.x; // 8 * (m/8)
  int row = blockIdx.x * row_tile + subwarp_id;
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  dnInput += v_id;
  dnOutput += v_id;
  E += v_id;
  float val;
  // DType res = init(REDUCE::Op);
  int col;
  for (; row < nr; row += stride) {
    float res = 0;
    int E_k_idx = -1;
    int start = __ldg(rowPtr + row);
    int end = __ldg(rowPtr + row + 1);
    if ((end - start) > 0) {
      for (int p = start; p < end; p++) {
        float val_pre_red;
        col = __ldg(colIdx + p);
        val = __guard_load_default_one<float>(values, p);
        val_pre_red = val * __ldg(dnInput + col * feature_size);
        res += val_pre_red;
      }
    }
    dnOutput[row * feature_size] = res;
    E[row * feature_size] = E_k_idx;
  }
}
// template <typename Index, typename DType>
// __global__ void csrspmm_seqreduce_rowbalance_with_mask_kernel(
//     const Index nr, const Index feature_size, const Index rowPtr[],
//     const Index colIdx[], const DType values[], const DType dnInput[],
//     const Index E[], DType dnOutput[]) {
//   Index row_tile = blockDim.y; // 8
//   Index subwarp_id = threadIdx.y;
//   Index stride = row_tile * gridDim.x; // 8 * (m/8)
//   Index row = blockIdx.x * row_tile + subwarp_id;
//   Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
//   dnInput += v_id;
//   dnOutput += v_id;
//   E += v_id;
//   DType res = 0, val;
//   Index col;
//   for (; row < nr; row += stride) {
//     Index E_k_idx;
//     Index start = __ldg(rowPtr + row);
//     Index end = __ldg(rowPtr + row + 1);
//     for (Index p = start; p < end; p++) {
//       DType val_pre_red;
//       col = __ldg(colIdx + p);
//       val = __guard_load_default_one<DType>(values, p);
//       E_k_idx = __ldg(E + col * feature_size);
//       if (E_k_idx == row) {
//         val_pre_red = val * __ldg(dnInput + col * feature_size);
//       }
//       res += val_pre_red;

//       // res += val * __ldg(dnInput + col * feature_size);
//     }
//     dnOutput[row * feature_size] = res;
//   }
// }

template <typename Index, typename DType, typename REDUCE, typename COMPUTE>
__global__ void csrspmm_seqreduce_nnzbalance_kernel(
    const Index nr, const Index feature_size, const Index nnz_,
    const Index rowPtr[], const Index colIdx[], const DType values[],
    const DType dnInput[], DType dnOutput[], Index E[]) {
  Index nnz = nnz_;
  if (nnz < 0)
    nnz = rowPtr[nr];

  Index Nnzdim_thread = blockDim.y * gridDim.x;
  Index NE_PER_THREAD = CEIL(nnz, Nnzdim_thread);
  Index eid = (blockIdx.x * blockDim.y + threadIdx.y) * NE_PER_THREAD;
  Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  Index col = 0;
  DType val = 0.0;

  if (v_id < feature_size) {
    if (eid < nnz) {
      Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, eid);
      Index step = __ldg(rowPtr + row + 1) - eid;

      for (Index ii = 0; ii < NE_PER_THREAD; ii++) {
        if (eid >= nnz)
          break;
        if (ii < step) {
          col = __ldg(colIdx + eid) * feature_size;
          val += __guard_load_default_one<DType>(values, eid) *
                 __ldg(dnInput + col + v_id);

          eid++;
        } else {
          atomicAdd(&dnOutput[row * feature_size + v_id], val);

          row = binary_search_segment_number<Index>(rowPtr, nr, nnz, eid);
          step = __ldg(rowPtr + row + 1) - eid;
          col = __ldg(colIdx + eid) * feature_size;
          val = __guard_load_default_one<DType>(values, eid) *
                __ldg(dnInput + col + v_id);

          eid++;
        }
      }
      // REDUCE::atomic_reduce(&dnOutput[row * feature_size + v_id], val);
      atomicAdd(&dnOutput[row * feature_size + v_id], val);
    }
  }
}

#endif
