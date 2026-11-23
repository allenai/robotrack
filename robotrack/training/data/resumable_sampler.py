"""
Sampler implementations that can seek into the current epoch.

This mirrors PyTorch's DistributedSampler ordering, while allowing a resumed
training job to start at a local sample offset instead of consuming thousands
of batches just to rebuild DataLoader state.
"""

import math
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler


class ResumableDistributedSampler(Sampler[int]):
    """Distributed/random sampler with a one-shot start offset."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.start_index = 0

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_index(self, start_index: int) -> None:
        self.start_index = max(0, int(start_index))

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if self.start_index:
            indices = indices[self.start_index :]
            self.start_index = 0

        return iter(indices)

    def __len__(self) -> int:
        return max(0, self.num_samples - self.start_index)
