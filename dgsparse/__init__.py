import importlib
import os.path as osp
import torch

import dgsparse.tensor
from .tensor import SparseTensor
from .storage import Storage
from .ftransform import csr2csc
from .spmm import spmm_sum

__version__ = '0.1'

for library in ['_spmm']:
    hip_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_hip', [osp.dirname(__file__)])
    spec = hip_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:
        raise ImportError(f"Could not find module '{library}_hip' in "
                          f'{osp.dirname(__file__)}')

__all__ = ['Storage', 'SparseTensor', 'csr2csc', 'spmm_sum']
