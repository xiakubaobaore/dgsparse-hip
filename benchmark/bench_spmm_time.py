import torch
import time
import dgl
from utils import GraphDataset

from dgsparse import SparseTensor
from dgsparse import spmm_sum


class SpMMSum:

    def __init__(self, data, in_dim, device, algorithm) -> None:
        self.tcsr = data.tcsr
        self.dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            self.tcsr.clone().detach(), True, requires_grad=True)

        self.device = device
        self.algorithm = algorithm
        self.input_feature = torch.rand((data.num_nodes, in_dim),
                                        device=device,
                                        requires_grad=True)

    def forward_check(self):

        for _ in range(10):
            torch.sparse.mm(self.tcsr, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            torch.sparse.mm(self.tcsr, self.input_feature)
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        for _ in range(10):
            spmm_sum(self.dcsr, self.input_feature, -1)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            spmm_sum(self.dcsr, self.input_feature, -1)
        torch.cuda.synchronize()
        end = time.time()
        rocsparse_time = end - start

        for _ in range(10):
            spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start
        return torch_sparse_time, rocsparse_time, dgsparse_time

    def backward_check(self):
        for _ in range(10):
            out = torch.sparse.mm(self.tcsr, self.input_feature)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = torch.sparse.mm(self.tcsr, self.input_feature)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        torch_sparse_time = end - start

        # temp = self.tcsr.clone().detach()
        # for _ in range(10):
        #     out = torch.matmul(temp, self.input_feature)
        # torch.cuda.synchronize()
        # start = time.time()
        # for _ in range(100):
        #     out = torch.matmul(temp, self.input_feature)
        #     out.sum().backward()
        # torch.cuda.synchronize()
        # end = time.time()
        # torch_matmul_time = end - start

        for _ in range(10):
            out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = spmm_sum(self.dcsr, self.input_feature, self.algorithm)
            out.sum().backward()
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start

        return torch_sparse_time, 0, dgsparse_time


def check_time(gc, stage='forward'):
    print(f'{stage} time:')
    # torch_sparse_time_list = []
    # rocsparse_time_list = []
    # dgsparse_time_list = []
    if stage == 'forward':
        torch_sparse_time, rocsparse_time, dgsparse_time = gc.forward_check()
    elif stage == 'backward':
        torch_sparse_time, rocsparse_time, dgsparse_time = gc.backward_check()
    else:
        raise ValueError
    # torch_sparse_time_list.append(torch_sparse_time)
    # rocsparse_time_list.append(rocsparse_time)
    # dgsparse_time_list.append(dgsparse_time)
    # print(f'torch_sparse {stage} time is: {torch_sparse_time_list}')
    # print(f'rocsparse {stage} time is: {rocsparse_time_list}')
    # print(f'dgsparse {stage} time is: {dgsparse_time_list}')

    print(f'torch_sparse {stage} time is: {torch_sparse_time:.4f}')
    print(f'rocsparse {stage} time is: {rocsparse_time:.4f}')
    print(f'dgsparse {stage} time is: {dgsparse_time:.4f}')


# def check_time(gc, stage='forward'):
#     print(f'{stage} time:')
#     if stage == 'forward':
#         dgsparse_time = gc.forward_check()
#     elif stage == 'backward':
#         dgsparse_time = gc.backward_check()
#     else:
#         raise ValueError
#     print(f'dgsparse_{gc.algorithm} {stage} time is: {dgsparse_time}')
#     return dgsparse_time

# def test_spmm_time(dataset, in_dim, device, reduce='sum', algorithm=0):
#     print()
#     print(
#         f'start testing {dataset} dataset, reduce is: {reduce}, in_dim is: {in_dim}, algorithm = {algorithm}'
#     )
#     data = GraphDataset(dataset, device)
#     if (reduce == 'sum'):
#         gc = SpMMSum(data, in_dim, device, algorithm)
#     else:
#         raise ValueError
#     return check_time(gc, stage='forward')


def test_spmm_time(dataset, in_dim, device, reduce='sum'):
    print()
    print(f'start testing {dataset} dataset, \
        reduce is: {reduce}, in_dim is: {in_dim}')
    data = GraphDataset(dataset, device)
    if reduce == 'sum':
        gc = SpMMSum(data, in_dim, device, 0)
    # elif reduce == 'max':
    #     gc = SpMMMax(data, in_dim, device, 0)
    # elif reduce == 'min':
    #     gc = SpMMMin(data, in_dim, device, 0)
    # elif reduce == 'mean':
    #     gc = SpMMMean(data, in_dim, device, 0)
    else:
        raise ValueError
    # check_time(gc, stage='forward')
    check_time(gc, stage='backward')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['cora', 'citeseer', 'pubmed', 'ppi']
    features_dim = [32, 64, 128]
    for dataset in datasets:
        for in_dim in features_dim:
            test_spmm_time(dataset, in_dim, device, reduce='sum')

# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     datasets = ['cora', 'citeseer', 'pubmed', 'ppi']
#     features_dim = [32, 64, 128]
#     algorithms = [-1, 0, 1, 2, 3]
#     all_time = []
#     for algorithm in algorithms:
#         time_list = []
#         for dataset in datasets:
#             for in_dim in features_dim:
#                 time_list.append(
#                     test_spmm_time(dataset,
#                                    in_dim,
#                                    device,
#                                    reduce='sum',
#                                    algorithm=algorithm))
#         all_time.append(time_list)
#     time_result = torch.tensor(all_time)
#     print(time_result)
