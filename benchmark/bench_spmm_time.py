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
            spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            spmm_sum(self.dcsr, self.input_feature, self.algorithm)
        torch.cuda.synchronize()
        end = time.time()
        dgsparse_time = end - start
        return dgsparse_time

    def backward_check(self):
        return 0


def check_time(gc, stage='forward'):
    print(f'{stage} time:')
    if stage == 'forward':
        dgsparse_time = gc.forward_check()
    elif stage == 'backward':
        dgsparse_time = gc.backward_check()
    else:
        raise ValueError
    print(f'dgsparse_{gc.algorithm} {stage} time is: {dgsparse_time}')
    return dgsparse_time


def test_spmm_time(dataset, in_dim, device, reduce='sum', algorithm=0):
    print()
    print(
        f'start testing {dataset} dataset, reduce is: {reduce}, in_dim is: {in_dim}, algorithm = {algorithm}'
    )
    data = GraphDataset(dataset, device)
    if (reduce == 'sum'):
        gc = SpMMSum(data, in_dim, device, algorithm)
    else:
        raise ValueError
    return check_time(gc, stage='forward')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['cora', 'citeseer', 'pubmed', 'ppi']
    features_dim = [32, 64, 128]
    algorithms = [-1, 0, 1, 2, 3]
    all_time = []
    for algorithm in algorithms:
        time_list = []
        for dataset in datasets:
            for in_dim in features_dim:
                time_list.append(
                    test_spmm_time(dataset,
                                   in_dim,
                                   device,
                                   reduce='sum',
                                   algorithm=algorithm))
        all_time.append(time_list)
    time_result = torch.tensor(all_time)
    print(time_result)
