import torch
from dgsparse import spmm_sum
from dgsparse import SparseTensor
# import pytest
from utils import GraphDataset


class SpMMSum:

    def __init__(self, data, in_dim, device, algorithm) -> None:
        # prepare for torch and dgsparse
        self.tcsr = data.tcsr

        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)

        self.in_dim = in_dim
        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand((data.num_nodes, in_dim),
                                        requires_grad=True,
                                        device=device)

    def forward_check(self):
        out_check = torch.sparse.mm(self.tcsr, self.input_feature)
        out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        assert torch.allclose(out, out_check)

    def backward_check(self):
        out_check = torch.sparse.mm(self.tcsr, self.input_feature)
        out_check.sum().backward()
        dX_check = self.input_feature.grad
        dA_check = self.tcsr.grad
        out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        out.sum().backward()
        dX = self.input_feature.grad
        dA_nnz = self.dcsr.storage._values.grad

        assert torch.allclose(dX, dX_check)
        assert torch.allclose(dA_nnz, dA_check.values())


datasets = ['cora', 'citeseer', 'pubmed', 'ppi']
features = [32, 64, 128]


def test_spmm_sum(dataset, feat):
    data = GraphDataset(dataset, 0)
    gc = SpMMSum(data, feat, 0, 0)
    gc.forward_check()
    # gc.backward_check()


for data in datasets:
    for feat in features:
        test_spmm_sum(data, feat)
