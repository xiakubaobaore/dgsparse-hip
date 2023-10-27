#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/extension.h>
#include <torch/script.h>

#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

int64_t hip_version() noexcept {
#ifdef WITH_HIP
  return HIP_VERSION;
#else
  return -1;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hip_version", &hip_version, "hip_version");
}
