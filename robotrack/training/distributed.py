"""
Distributed training utilities for torch.distributed.
Replaces the lightning_fabric.Fabric setup used in the original train_stage1.py.
"""

import os
import random
import logging
from datetime import timedelta
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist


def get_machine_local_and_dist_rank() -> Tuple[int, int]:
    """Get (local_rank, dist_rank) from torchrun or SLURM env vars."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist_rank = int(os.environ.get("RANK", 0))
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
        dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    else:
        local_rank = 0
        dist_rank = 0
    return local_rank, dist_rank


def get_world_size() -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    return 1


def init_distributed(backend: str = "nccl", timeout_mins: int = 30):
    """Initialize torch.distributed; returns (rank, world_size, local_rank, device)."""
    local_rank, rank = get_machine_local_and_dist_rank()
    world_size = get_world_size()

    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=timeout_mins),
        )
        logging.info(
            f"Initialized distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}"
        )
    else:
        logging.info("Running in single-process mode (no distributed)")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def set_seeds(seed: int, rank: int = 0):
    """Set RNG seeds. Each rank gets a different seed so dataloader workers diverge."""
    seed = seed + rank
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
