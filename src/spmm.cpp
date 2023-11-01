
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

#include "../include/hip/spmm_hip.h"

std::vector<torch::Tensor> csr2csc(torch::Tensor rowptr, torch::Tensor colind,
                                   torch::Tensor values);

torch::Tensor spmm_sum(torch::Tensor rowptr, torch::Tensor col,
                       torch::Tensor values, torch::Tensor colptr,
                       torch::Tensor row, torch::Tensor csr2csc,
                       torch::Tensor dense, bool has_value, int64_t algorithm);

std::vector<torch::Tensor> csr2csc(torch::Tensor rowptr, torch::Tensor colind,
                                   torch::Tensor values) {
  return csr2csc_hip(rowptr, colind, values);
}

class SpMMSum : public torch::autograd::Function<SpMMSum> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor values, torch::Tensor colptr,
                               torch::Tensor row, torch::Tensor csr2csc,
                               torch::Tensor dense, bool has_value,
                               int64_t algorithm) {
    auto out = spmm_hip(rowptr, col, values, dense, has_value, algorithm,
                        REDUCEOP::SUM, COMPUTEOP::ADD);
    ctx->saved_data["has_value"] = has_value;
    ctx->saved_data["algorithm"] = algorithm;
    ctx->save_for_backward({rowptr, col, values, colptr, row, csr2csc, dense});
    return out[0];
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto algorithm = ctx->saved_data["algorithm"].toInt();
    auto saved = ctx->get_saved_variables();
    auto rowptr = saved[0], col = saved[1], values = saved[2],
         colptr = saved[3], row = saved[4], csr2csc = saved[5],
         dense = saved[6];

    auto grad_value = torch::Tensor();
    if (has_value > 0 &&
        torch::autograd::any_variable_requires_grad({values})) {
      grad_value = sddmm_cuda_csr(rowptr, col, grad_out, dense, REDUCEOP::SUM);
    }

    auto grad_mat = std::vector<torch::Tensor>();
    if (torch::autograd::any_variable_requires_grad({dense})) {
      auto t_values = torch::Tensor();
      t_values = values.view({-1, 1}).index_select(0, csr2csc).view(-1);
      grad_mat = spmm_hip(colptr, row, t_values, grad_out, has_value, algorithm,
                          SUM, ADD);
    }
    return {torch::Tensor(), torch::Tensor(), grad_value,
            torch::Tensor(), torch::Tensor(), torch::Tensor(),
            grad_mat[0],     torch::Tensor(), torch::Tensor()};
    //       has_value};
  }
};

torch::Tensor spmm_sum(torch::Tensor rowptr, torch::Tensor col,
                       torch::Tensor values, torch::Tensor colptr,
                       torch::Tensor row, torch::Tensor csr2csc,
                       torch::Tensor dense, bool has_value, int64_t algorithm) {
  return SpMMSum::apply(rowptr, col, values, colptr, row, csr2csc, dense,
                        has_value, algorithm);
}

TORCH_LIBRARY(dgsparse_spmm, m) {
  m.def("spmm_sum", &spmm_sum);
  m.def("csr2csc", &csr2csc);
}
