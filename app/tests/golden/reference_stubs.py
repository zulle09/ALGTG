"""
Stand-ins for the heavy imports of the reference code, shared by the golden generators
(make_*_golden.py). Development tools only: nothing under app/ imports the reference code.
"""

import importlib.util
import sys
import types
from contextlib import contextmanager

import numpy as np
import torch


def install_stub(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_dal_toolbox_stubs():
    """What `from dal_toolbox... import ...` needs, without lightning, torchmetrics or the package itself."""
    class Query:   # dal_toolbox's Query base class: self.rng comes from setup_rng(seed)
        def __init__(self, random_seed=None):
            self.random_seed = random_seed
            self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random.mtrand._rand

    install_stub('dal_toolbox')
    install_stub('dal_toolbox.active_learning', ActiveLearningDataModule=object)
    install_stub('dal_toolbox.active_learning.strategies', Query=Query)
    install_stub('dal_toolbox.models')
    install_stub('dal_toolbox.models.base', BaseModule=object)


def install_pycls_stubs():
    """The authors' deep-al code imports `pycls.datasets.utils` for its feature loader."""
    install_stub('pycls')
    datasets = install_stub('pycls.datasets')
    datasets.utils = install_stub('pycls.datasets.utils', load_features=None)
    sys.modules['pycls'].datasets = datasets
    return datasets.utils


@contextmanager
def cuda_as_identity():
    """The reference graph builders call `.cuda()` unconditionally; run them on whatever device we have."""
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        yield
    finally:
        torch.Tensor.cuda = original
