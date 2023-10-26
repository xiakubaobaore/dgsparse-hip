
#include <torch/extension.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "../../include/hip/spmm_hip.h"
#include "../../include/hip/csr2csc.hpp"


std::vector<torch::Tensor> csr2csc_hip(
    torch::Tensor csrRowPtr,
    torch::Tensor csrColInd,
    torch::Tensor csrVal
) {
    assert(csrRowPtr.device().type() == torch::kCUDA);
    assert(csrColInd.device().type() == torch::kCUDA);
    assert(csrVal.device().type() == torch::kCUDA);
    assert(csrRowPtr.is_contiguous());
    assert(csrColInd.is_contiguous());
    assert(csrVal.is_contiguous());
    assert(csrRowPtr.dtype() == torch::kInt32);
    assert(csrColInd.dtype() == torch::kInt32);
    assert(csrVal.dtype() == torch::kFloat32);

    const auto n = csrRowPtr.size(0) - 1;
    const auto nnz = csrColInd.size(0);
    auto devid = csrRowPtr.device().index();
    auto optionsF =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto optionsI =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
    auto cscColPtr = torch::empty({n + 1}, optionsI);
    auto cscRowInd = torch::empty({nnz}, optionsI);
    auto cscVal = torch::empty({nnz}, optionsF);
    csr2cscKernel(
        n, n, nnz, devid,
        csrRowPtr.data_ptr<int>(), csrColInd.data_ptr<int>(), csrVal.data_ptr<float>(),
        cscColPtr.data_ptr<int>(), cscRowInd.data_ptr<int>(), cscVal.data_ptr<float>()
    );
    return {cscColPtr, cscRowInd, cscVal};
}
