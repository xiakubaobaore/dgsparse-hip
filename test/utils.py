import torch
import dgl
import scipy
import numpy as np


class GraphDataset:

    def __init__(self, name: str, device) -> None:
        print()
        print(
            f'---------------- initing {name} dataset on {device} ----------------'
        )
        self.name = name
        self.device = device
        self.get_dataset()

    def get_dataset(self):
        if self.name == 'cora':
            dataset = dgl.data.CoraGraphDataset()
            graph = dataset[0]
        elif self.name == 'citeseer':
            dataset = dgl.data.CiteseerGraphDataset()
            graph = dataset[0]
        elif self.name == 'pubmed':
            dataset = dgl.data.PubmedGraphDataset()
            graph = dataset[0]
        elif self.name == 'ppi':
            dataset = dgl.data.PPIDataset()
            graph = dataset[0]
        elif self.name == 'reddit':
            dataset = dgl.data.RedditDataset()
            graph = dataset[0]
        else:
            raise KeyError(f'UnKnown dataset {self.name}')
        num_nodes = graph.num_nodes()
        row, col = graph.adj_tensors('coo')
        data = np.ones(col.shape)
        scipy_coo = scipy.sparse.csr_matrix((data, (row, col)),
                                            shape=(num_nodes, num_nodes))
        scipy_csr = scipy_coo.tocsr()
        rowptr = scipy_csr.indptr
        col = scipy_csr.indices
        value = scipy_csr.data
        self.num_nodes = num_nodes
        self.tcsr = torch.sparse_csr_tensor(rowptr,
                                            col,
                                            value,
                                            dtype=torch.float,
                                            size=(self.num_nodes,
                                                  self.num_nodes),
                                            requires_grad=True,
                                            device=self.device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gc = GraphDataset('cora', device)
    gc = GraphDataset('citeseer', device)
    gc = GraphDataset('pubmed', device)
