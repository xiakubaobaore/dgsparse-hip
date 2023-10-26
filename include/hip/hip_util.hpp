#ifndef UTIL_H
#define UTIL_H

#include <hip_runtime.h>
#include <hip_runtime_api.h>
#include <hipsparse.h>
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

#endif
