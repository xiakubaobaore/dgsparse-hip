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
#endif
