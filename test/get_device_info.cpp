#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

int main() {
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  int w = props.warpSize;
  printf("warpsize = %d\n", w);

  return 0;
}
