"""
Helpers for building train dataloaders from Hydra-instantiated datasets.
"""

import logging

import torch
from torch.utils.data import DataLoader, Dataset

from robotrack.training.distributed import seed_worker
from robotrack.training.data.deterministic_dataset import DeterministicSampleDataset
from robotrack.training.data.resumable_sampler import ResumableDistributedSampler


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = False,
    drop_last: bool = True,
    collate_fn=None,
    distributed: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    sampler_seed: int | None = None,
    deterministic_samples: bool = True,
    seed_salt: str = "",
) -> DataLoader:
    """Build a DataLoader, optionally with a DistributedSampler."""
    if sampler_seed is None:
        sampler_seed = seed

    if deterministic_samples:
        dataset = DeterministicSampleDataset(
            dataset,
            base_seed=seed,
            salt=seed_salt or f"rank{rank}",
        )

    sampler = ResumableDistributedSampler(
        dataset,
        num_replicas=world_size if distributed else 1,
        rank=rank if distributed else 0,
        shuffle=shuffle,
        seed=sampler_seed,
        drop_last=False,
    )
    shuffle = False

    g = torch.Generator()
    g.manual_seed(seed + rank)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=g,
    )

    logging.info(
        f"Created DataLoader: batch_size={batch_size} num_workers={num_workers} "
        f"distributed={distributed} world_size={world_size} rank={rank} "
        f"sampler_seed={sampler_seed} deterministic_samples={deterministic_samples} "
        f"len(dataset)={len(dataset)}"
    )
    return loader
