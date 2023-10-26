
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

#include "../include/hip/spmm_hip.h"

std::vector<torch::Tensor> csr2csc(torch::Tensor rowptr, torch::Tensor colind,
                                   torch::Tensor values);

std::vector<torch::Tensor> csr2csc(torch::Tensor rowptr, torch::Tensor colind,
                                   torch::Tensor values) {
  return csr2csc_hip(rowptr, colind, values);
}

TORCH_LIBRARY(dgsparse_spmm, m) { m.def("csr2csc", &csr2csc); }
