"""
Dataset wrapper that makes sample-time randomness independent of iteration order.

Many AllTracker datasets draw from Python, NumPy, and Torch RNGs inside
``__getitem__`` for temporal crops, photometric/spatial augmentation, and point
subsampling.  For exact fast resume we need sample ``(epoch, index)`` to produce
the same augmented example without replaying all earlier samples in the epoch.
"""

import contextlib
import hashlib
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _stable_seed(base_seed: int, epoch: int, index: int, salt: str) -> int:
    payload = f"{base_seed}:{epoch}:{index}:{salt}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


@contextlib.contextmanager
def _temporary_rng_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    seed32 = seed % (2**32)
    random.seed(seed32)
    np.random.seed(seed32)
    torch.manual_seed(seed)

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


class DeterministicSampleDataset(Dataset):
    """Wrap a dataset so stochastic ``__getitem__`` is keyed by epoch and index."""

    def __init__(self, dataset: Dataset, base_seed: int, salt: str = ""):
        self.dataset = dataset
        self.base_seed = int(base_seed)
        self.salt = str(salt)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        seed = _stable_seed(self.base_seed, self.epoch, int(index), self.salt)
        with _temporary_rng_seed(seed):
            return self.dataset[index]

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)
