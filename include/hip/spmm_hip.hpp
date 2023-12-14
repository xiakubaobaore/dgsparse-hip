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

template <typename Index, typename DType, typename access_t, typename REDUCE,
          typename COMPUTE>
__global__ void
csrspmm_parreduce_rowbalance_kernel(const Index nr, const Index feature_size,
                                    const Index rowPtr[], const Index colIdx[],
                                    const DType values[], const DType dnInput[],
                                    DType dnOutput[], Index E[]) {
  constexpr Index CoarsenFactor = sizeof(access_t) / sizeof(DType);

  Index lane_id = (threadIdx.x & (64 - 1));
  Index stride = gridDim.x * blockDim.y;
  Index row = blockIdx.x * blockDim.y + threadIdx.y;

  // get the dense column offset
  Index col_offset = blockIdx.y * 64 + (threadIdx.x >> 6) * CoarsenFactor;
  const DType *B_panel = dnInput + col_offset;
  DType *C_panel = dnOutput + col_offset;
  Index ldB = feature_size;
  Index ldC = feature_size;

  if (col_offset >= feature_size)
    return;
  if (col_offset + CoarsenFactor >= feature_size)
    goto Ndim_Residue;

  for (; row < nr; row += stride) {
    // declare accumulators
    DType c[CoarsenFactor];
#pragma unroll
    for (Index j = 0; j < CoarsenFactor; j++) {
      c[j] = init(REDUCE::Op);
    }
    DType buffer[CoarsenFactor];

    Index start = rowPtr[row];
    Index end = rowPtr[row + 1];
    Index k;
    DType val;
    DType val_pre_red;
    Index E_k_idx[CoarsenFactor] = {0};
    // DType res = init(REDUCE::Op);

    for (Index jj = start + lane_id; jj < end; jj += 64) {
      k = colIdx[jj];
      val = __guard_load_default_one<DType>(values, jj);

      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        val_pre_red = val * buffer[i];
        if ((REDUCE::Op == REDUCEOP::MAX && val_pre_red > c[i]) ||
            (REDUCE::Op == REDUCEOP::MIN && val_pre_red < c[i])) {
          c[i] = val_pre_red;
          E_k_idx[i] = k;
        } else if (REDUCE::Op == REDUCEOP::SUM ||
                   REDUCE::Op == REDUCEOP::MEAN) {
          c[i] += val_pre_red;
        }
      }
    }

#pragma unroll
    for (Index i = 0; i < CoarsenFactor; i++) {
      DType temp_c = c[i];
      // row-wise reduction is a simple merge-tree
      SHFL_DOWN_REDUCE(c[i], temp_c, REDUCE::Op, E_k_idx[i])
    }

    // store to C in vector-type
    if (lane_id == 0) {
      *(access_t *)(C_panel + row * ldC) = *(access_t *)c;
      *(access_t *)(E + col_offset + row * ldC) = *(access_t *)E_k_idx;
    }
  }
  return;

Ndim_Residue:
  Index valid_lane_num = feature_size - col_offset;

  for (; row < nr; row += stride) {
    // get row offsets
    DType c[CoarsenFactor];
#pragma unroll
    for (Index j = 0; j < CoarsenFactor; j++) {
      c[j] = init(REDUCE::Op);
    }
    DType buffer[CoarsenFactor];
    // access_t res = init_zeros<access_t>();

    Index start = rowPtr[row];
    Index end = rowPtr[row + 1];
    Index k;
    DType val;
    DType val_pre_red;
    Index E_k_idx[CoarsenFactor] = {0};

    for (Index jj = start + lane_id; jj < end; jj += 64) {
      k = colIdx[jj];
      val = __guard_load_default_one<DType>(values, jj);

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          buffer[i] = B_panel[k * ldB + i];
        }
      }

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        val_pre_red = val * buffer[i];
        if ((REDUCE::Op == REDUCEOP::MAX && val_pre_red > c[i]) ||
            (REDUCE::Op == REDUCEOP::MIN && val_pre_red < c[i])) {
          c[i] = val_pre_red;
          E_k_idx[i] = k;
        } else if (REDUCE::Op == REDUCEOP::SUM ||
                   REDUCE::Op == REDUCEOP::MEAN) {
          c[i] += val_pre_red;
        }
      }
    }

#pragma unroll
    for (Index i = 0; i < CoarsenFactor; i++) {
      DType temp_c = c[i];
      // row-wise reduction is a simple merge-tree
      SHFL_DOWN_REDUCE(c[i], temp_c, REDUCE::Op, E_k_idx[i])
    }

    if (lane_id == 0) {
#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          C_panel[row * ldC + i] = c[i];
          E[col_offset + row * ldC + i] = E_k_idx[i];
        }
      }
    }
  }
}

template <typename Index, typename DType, typename access_t>
__global__ void
csrspmm_parreduce_nnzbalance_kernel(const Index nr, const Index feature_size,
                                    const Index nnz_, const Index rowPtr[],
                                    const Index colIdx[], const DType values[],
                                    const DType dnInput[], DType dnOutput[]) {
  constexpr Index CoarsenFactor = sizeof(access_t) / sizeof(DType);
  Index nnz = nnz_;
  if (nnz < 0)
    nnz = rowPtr[nr];

  Index lane_id = (threadIdx.x & (64 - 1));
  Index Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  Index nz_start = Nnzdim_warp_id * 64;
  Index stride = gridDim.x * (blockDim.y * 64);

  // get the dense column offset
  Index col_offset = blockIdx.y * 64 + (threadIdx.x >> 6) * CoarsenFactor;
  const DType *B_panel = dnInput + col_offset;
  DType *C_panel = dnOutput + col_offset;
  Index ldB = feature_size;
  Index ldC = feature_size;

  Index k;
  DType v;
  Index E_k_idx[CoarsenFactor] = {0};
  DType c[CoarsenFactor] = {0};
  DType buffer[CoarsenFactor] = {0};

  if (col_offset >= feature_size)
    return;
  if (col_offset + CoarsenFactor >= feature_size)
    goto Ndim_Residue;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, nz_id);

    if (nz_id < nnz) {
      k = colIdx[nz_id];
      v = __guard_load_default_one<DType>(values, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

    // load B-elements in vector-type
    *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = buffer[i] * v;
    }

    // reduction
    Index row_intv = __shfl(row, (64 - 1)) - __shfl(row, 0);
    if (row_intv == 0) {
// if all non-zeros in this warp belong to the same row, use a simple reduction
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        DType temp_c = c[i];
        int idx = 0;
        SHFL_DOWN_REDUCE(c[i], temp_c, REDUCEOP::SUM, idx);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    } else {
      // if non-zeros belong to different rows, use a parallel-scan primitive
      // thread that holds the start of each segment are responsible for writing
      // results
      bool is_seg_start = ((__shfl_up(row, 1) != row) || (lane_id == 0));
      DType tmpv;
      Index tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
// atomic add has no vector-type form.
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    }
  }
  return;
Ndim_Residue:
  Index valid_lane_num = feature_size - col_offset;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, nz_id);

    if (nz_id < nnz) {
      k = colIdx[nz_id];
      v = __guard_load_default_one<DType>(values, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = B_panel[k * ldB + i] * v;
      }
    }

    // reduction
    Index row_intv = __shfl(row, (64 - 1)) - __shfl(row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        DType temp_c = c[i];
        int idx = 0;
        SHFL_DOWN_REDUCE(c[i], temp_c, REDUCEOP::SUM, idx);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    } else {
      bool is_seg_start = ((__shfl_up(row, 1) != row) || (lane_id == 0));
      DType tmpv;
      Index tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
  }
  return;
}

template <typename Index, typename DType>
__global__ void
csrspmm_neighbor_group_kernel(const Index edge_groups, const Index feature_size,
                              const Index group_key[], const Index group_row[],
                              const Index colIdx[], const DType values[],
                              const DType dnInput[], DType dnOutput[]) {
  Index group_tile = blockDim.y; // combine a set of groups together
  Index subwarp_id = threadIdx.y;
  Index group = blockIdx.x * group_tile + subwarp_id; // which node_group
  Index v_id = threadIdx.x;

  if (group < edge_groups) {
    Index row = group_row[group]; // get the specific row of each node group
    dnInput += v_id;
    dnOutput += v_id;
    DType res = 0, val;
    Index col;
    Index start = __ldg(group_key + group);
    Index end = __ldg(group_key + group + 1);
    for (Index p = start; p < end; p++) {
      DType val_pre_red = 0;
      col = __ldg(colIdx + p);
      val = __guard_load_default_one<DType>(values, p);
      val_pre_red = val * __ldg(dnInput + col * feature_size);
      res += val_pre_red;
    }
    atomicAdd(dnOutput + row * feature_size,
              res); // atomic, cuz different node group -> same row
  }
}

template <typename Index, typename DType>
__global__ void csrspmm_seqreduce_rowbalance_with_mask_kernel(
    const Index nr, const Index feature_size, const Index rowPtr[],
    const Index colIdx[], const DType values[], const DType dnInput[],
    const Index E[], DType dnOutput[]) {
  Index row_tile = blockDim.y; // 8
  Index subwarp_id = threadIdx.y;
  Index stride = row_tile * gridDim.x; // 8 * (m/8)
  Index row = blockIdx.x * row_tile + subwarp_id;
  Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  dnInput += v_id;
  dnOutput += v_id;
  E += v_id;
  DType res = 0, val;
  Index col;
  for (; row < nr; row += stride) {
    Index E_k_idx;
    Index start = __ldg(rowPtr + row);
    Index end = __ldg(rowPtr + row + 1);
    for (Index p = start; p < end; p++) {
      DType val_pre_red;
      col = __ldg(colIdx + p);
      val = __guard_load_default_one<DType>(values, p);
      E_k_idx = __ldg(E + col * feature_size);
      if (E_k_idx == row) {
        val_pre_red = val * __ldg(dnInput + col * feature_size);
      }
      res += val_pre_red;

      // res += val * __ldg(dnInput + col * feature_size);
    }
    dnOutput[row * feature_size] = res;
  }
}

#endif
