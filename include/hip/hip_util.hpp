#ifndef UTIL_H
#define UTIL_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>
#include <stdio.h>

#define checkHipError(a)                                                       \
  do {                                                                         \
    if (hipSuccess != (a)) {                                                   \
      fprintf(stderr, "Hip runTime error in line %d of file %s \
            : %s \n",                                                          \
              __LINE__, __FILE__, hipGetErrorString(hipGetLastError()));       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define checkHipSparseError(a)                                                 \
  do {                                                                         \
    if (HIPSPARSE_STATUS_SUCCESS != (a)) {                                     \
      fprintf(stderr, "HipSparse runTime error in line %d of file %s \
            : %s \n",                                                          \
              __LINE__, __FILE__, hipGetErrorString(hipGetLastError()));       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CEIL(x, y) (((x) + (y)-1) / (y))

#define FULLMASK 0xffffffff
#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
}

template <typename index_t>
__device__ __forceinline__ index_t
binary_search_segment_number(const index_t *seg_offsets, const index_t n_seg,
                             const index_t n_elem, const index_t elem_id) {
  index_t lo = 1, hi = n_seg, mid;
  while (lo < hi) {
    mid = (lo + hi) >> 1;
    if (seg_offsets[mid] <= elem_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (hi - 1);
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load(ldType &tmp, data *array, int offset) {
  tmp = *(reinterpret_cast<ldType *>(array + offset));
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load(data *lhd, data *rhd, int offset) {
  *(reinterpret_cast<ldType *>(lhd)) =
      *(reinterpret_cast<ldType *>(rhd + offset));
}

__device__ __forceinline__ int findRow(const int *S_csrRowPtr, int eid,
                                       int start, int end) {
  int low = start, high = end;
  if (low == high)
    return low;
  while (low < high) {
    int mid = (low + high) >> 1;
    if (S_csrRowPtr[mid] <= eid)
      low = mid + 1;
    else
      high = mid;
  }
  if (S_csrRowPtr[high] == eid)
    return high;
  else
    return high - 1;
}

template <typename data>
__device__ __forceinline__ void selfMulConst4(data *lhd, data Const) {
  lhd[0] *= Const;
  lhd[1] *= Const;
  lhd[2] *= Const;
  lhd[3] *= Const;
}

template <typename ldType, typename data>
__device__ __forceinline__ void Load4(ldType *tmp, data *array, int *offset,
                                      int offset2 = 0) {
  Load(tmp[0], array, offset[0] + offset2);
  Load(tmp[1], array, offset[1] + offset2);
  Load(tmp[2], array, offset[2] + offset2);
  Load(tmp[3], array, offset[3] + offset2);
}

template <typename vecData, typename data>
__device__ __forceinline__ data vecDot2(vecData &lhd, vecData &rhd) {
  return lhd.x * rhd.x + lhd.y * rhd.y;
}

template <typename vecData, typename data>
__device__ __forceinline__ void vec2Dot4(data *cal, vecData *lhd,
                                         vecData *rhd) {
  cal[0] += vecDot2<vecData, data>(lhd[0], rhd[0]);
  cal[1] += vecDot2<vecData, data>(lhd[1], rhd[1]);
  cal[2] += vecDot2<vecData, data>(lhd[2], rhd[2]);
  cal[3] += vecDot2<vecData, data>(lhd[3], rhd[3]);
}

template <typename data>
__device__ __forceinline__ void Dot4(data *cal, data *lhd, data *rhd) {
  cal[0] += lhd[0] * rhd[0];
  cal[1] += lhd[1] * rhd[1];
  cal[2] += lhd[2] * rhd[2];
  cal[3] += lhd[3] * rhd[3];
}

template <typename data>
__device__ __forceinline__ void AllReduce4(data *multi, int stride,
                                           int warpSize) {
  for (; stride > 0; stride >>= 1) {
    multi[0] += __shfl_xor(multi[0], stride, warpSize);
    multi[1] += __shfl_xor(multi[1], stride, warpSize);
    multi[2] += __shfl_xor(multi[2], stride, warpSize);
    multi[3] += __shfl_xor(multi[3], stride, warpSize);
  }
}

template <typename ldType, typename data>
__device__ __forceinline__ void Store(data *lhd, data *rhd, int offset) {
  *(reinterpret_cast<ldType *>(lhd + offset)) =
      *(reinterpret_cast<ldType *>(rhd));
}

#endif
