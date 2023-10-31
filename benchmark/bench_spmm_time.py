import torch
import time
import dgl
from utils import GraphDataset

from dgsparse import SparseTensor


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
        return 0

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
    print(f'dgsparse {stage} time is: {dgsparse_time}')


def test_spmm_time(dataset, in_dim, device, reduce='sum'):
    print()
    print(
        f'start testing {dataset} dataset, reduce is: {reduce}, in_dim is: {in_dim}'
    )
    data = GraphDataset(dataset, device)
    if (reduce == 'sum'):
        gc = SpMMSum(data, in_dim, device, 0)
    else:
        raise ValueError
    check_time(gc, stage='forward')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['cora', 'citeseer', 'pubmed', 'ppi']
    features_dim = [32]
    for dataset in datasets:
        for in_dim in features_dim:
            test_spmm_time(dataset, in_dim, device, reduce='sum')
