
#include <torch/extension.h>

#include <tuple>
#include <vector>

#include "../gspmm.h"

std::vector<torch::Tensor> csr2csc_hip(torch::Tensor csrRowPtr,
                                       torch::Tensor csrColInd,
                                       torch::Tensor csrVal);

std::vector<torch::Tensor> spmm_hip(torch::Tensor csrptr, torch::Tensor indices,
                                    torch::Tensor edge_val,
                                    torch::Tensor in_feat, bool has_value,
                                    int64_t algorithm, REDUCEOP reduce_op,
                                    COMPUTEOP compute_op);

torch::Tensor sddmm_hip_csr(torch::Tensor rowptr, torch::Tensor colind,
                            torch::Tensor D1, torch::Tensor D2,
                            REDUCEOP reduce_op);

torch::Tensor sddmm_hip_csr_with_mask(torch::Tensor rowptr,
                                      torch::Tensor colind, torch::Tensor D1,
                                      torch::Tensor D2, torch::Tensor E);

torch::Tensor spmm_hip_with_mask(torch::Tensor csrptr, torch::Tensor indices,
                                 torch::Tensor edge_val, torch::Tensor in_feat,
                                 torch::Tensor E, bool has_value);
