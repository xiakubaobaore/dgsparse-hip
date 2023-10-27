#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "../../include/hip/csr2csc.hpp"
#include "../../include/hip/hip_util.hpp"
#include "../../include/hip/sddmm_hip.hpp"
#include "../../include/hip/spmm_hip.hpp"
#include "../../include/gspmm.h"

std::vector<torch::Tensor>
spmm_cuda(torch::Tensor csrptr, torch::Tensor indices, torch::Tensor edge_val,
          torch::Tensor in_feat, bool has_value, int64_t algorithm,
          REDUCEOP reduce_op, COMPUTEOP compute_op) {
  //   assertTensor(csrptr, torch::kInt32);
  //   assertTensor(indices, torch::kInt32);
  //   assertTensor(in_feat, torch::kFloat32);
  //   assertTensor(edge_val, torch::kFloat32);
  in_feat = in_feat.contiguous();
  int v = csrptr.size(0) - 1;
  int Ndim_worker = in_feat.size(1);
  int f = Ndim_worker;
  int e = indices.size(0);
  auto devid = in_feat.device().index();

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::empty({v, f}, options);
  auto options_E =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
  auto out_E = torch::empty({v, f}, options_E);

  if (algorithm == 0) {
    int Mdim_worker = csrptr.size(0) - 1;
    // int v = Mdim_worker;
    int Ndim_worker = in_feat.size(1);
    // int f = Ndim_worker;
    // int e = indices.size(0);
    int RefThreadPerBlock = 256;
    int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
    int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
    int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
    int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

    dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

    // auto out_feat = torch::empty({v, f}, options);

    if (has_value)
      SWITCH_REDUCEOP(reduce_op, REDUCE, {
        SWITCH_COMPUTEOP(compute_op, COMPUTE, {
          hipLaunchKernelGGL(
            csrspmm_seqreduce_rowbalance_kernel<int, float, REDUCE, COMPUTE>,
            gridDim, blockDim, 0, 0,
            Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
            indices.data_ptr<int>(), edge_val.data_ptr<float>(),
            in_feat.data_ptr<float>(), out_feat.data_ptr<float>(),
            out_E.data_ptr<int>());
        });
      });

    else
      SWITCH_REDUCEOP(reduce_op, REDUCE, {
        SWITCH_COMPUTEOP(compute_op, COMPUTE, {
          hipLaunchKernelGGL(
            csrspmm_seqreduce_rowbalance_kernel<int, float, REDUCE, COMPUTE>,
            gridDim, blockDim, 0, 0,
            Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
            indices.data_ptr<int>(), (float *)nullptr,
            in_feat.data_ptr<float>(), out_feat.data_ptr<float>(),
            out_E.data_ptr<int>());
        });
      });
  }

  SWITCH_REDUCEOP(reduce_op, REDUCE, {
    if (REDUCE::Op == MAX || REDUCE::Op == MIN) {
      return {out_feat, out_E};
    } else {
      return {out_feat};
    }
  });
}

// torch::Tensor spmm_cuda_with_mask(torch::Tensor csrptr, torch::Tensor indices,
//                                   torch::Tensor edge_val, torch::Tensor in_feat,
//                                   torch::Tensor E, bool has_value,
//                                   int64_t algorithm, REDUCEOP reduce_op,
//                                   COMPUTEOP compute_op) {
//   in_feat = in_feat.contiguous();
//   int v = csrptr.size(0) - 1;
//   int Ndim_worker = in_feat.size(1);
//   int f = Ndim_worker;
//   int e = indices.size(0);
//   auto devid = in_feat.device().index();
//   auto options =
//       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);

//   auto out_feat = torch::empty({v, f}, options);

//   if (algorithm == 0) {
//     int Mdim_worker = csrptr.size(0) - 1;
//     // int v = Mdim_worker;
//     int Ndim_worker = in_feat.size(1);
//     // int f = Ndim_worker;
//     // int e = indices.size(0);
//     int RefThreadPerBlock = 256;
//     int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
//     int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
//     int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
//     int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

//     dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
//     dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

//     // auto out_feat = torch::empty({v, f}, options);

//     if (has_value)
//       csrspmm_seqreduce_rowbalance_with_mask_kernel<<<gridDim, blockDim>>>(
//           Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
//           indices.data_ptr<int>(), edge_val.data_ptr<float>(),
//           in_feat.data_ptr<float>(), E.data_ptr<int>(),
//           out_feat.data_ptr<float>());

//     else
//       csrspmm_seqreduce_rowbalance_with_mask_kernel<<<gridDim, blockDim>>>(
//           Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
//           indices.data_ptr<int>(), (float *)nullptr, in_feat.data_ptr<float>(),
//           E.data_ptr<int>(), out_feat.data_ptr<float>());
//   }

//   return out_feat;
// };

// std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
//                                         torch::Tensor csrColInd,
//                                         torch::Tensor csrVal) {
//   assert(csrRowPtr.device().type() == torch::kCUDA);
//   assert(csrColInd.device().type() == torch::kCUDA);
//   assert(csrVal.device().type() == torch::kCUDA);
//   assert(csrRowPtr.is_contiguous());
//   assert(csrColInd.is_contiguous());
//   assert(csrVal.is_contiguous());
//   assert(csrRowPtr.dtype() == torch::kInt32);
//   assert(csrColInd.dtype() == torch::kInt32);
//   assert(csrVal.dtype() == torch::kFloat32);
//   const at::cuda::OptionalCUDAGuard device_guard1(device_of(csrRowPtr));
//   const at::cuda::OptionalCUDAGuard device_guard2(device_of(csrColInd));
//   const at::cuda::OptionalCUDAGuard device_guard3(device_of(csrVal));
//   const auto n = csrRowPtr.size(0) - 1;
//   const auto nnz = csrColInd.size(0);
//   auto devid = csrRowPtr.device().index();
//   auto optionsF =
//       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
//   auto optionsI =
//       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
//   auto cscColPtr = torch::empty({n + 1}, optionsI);
//   auto cscRowInd = torch::empty({nnz}, optionsI);
//   auto cscVal = torch::empty({nnz}, optionsF);
//   csr2cscKernel(n, n, nnz, devid, csrRowPtr.data_ptr<int>(),
//                 csrColInd.data_ptr<int>(), csrVal.data_ptr<float>(),
//                 cscColPtr.data_ptr<int>(), cscRowInd.data_ptr<int>(),
//                 cscVal.data_ptr<float>());
//   return {cscColPtr, cscRowInd, cscVal};
// }
