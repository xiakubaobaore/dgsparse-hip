
#include <torch/extension.h>

#include <tuple>
#include <vector>

std::vector<torch::Tensor> csr2csc_hip(torch::Tensor csrRowPtr,
                                       torch::Tensor csrColInd,
                                       torch::Tensor csrVal);
