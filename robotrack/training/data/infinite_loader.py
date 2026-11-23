"""
Infinite batch iterator for step-based training.

Cycles its underlying DataLoader forever, calls sampler.set_epoch each cycle,
and skips batches whose collate_fn returned gotit=False so they don't reach
the forward pass (matches the original alltracker behavior).
"""

import logging
from typing import Iterator, Any, Optional

from torch.utils.data import DataLoader


class InfiniteBatchIterator:
    """Step-based wrapper around a (possibly distributed) DataLoader."""

    def __init__(
        self,
        dataloader: DataLoader,
        start_epoch: int = 0,
        batches_to_skip: int = 0,
        name: str = "loader",
    ):
        self.dataloader = dataloader
        self.name = name
        self._epoch = start_epoch
        self._iter: Optional[Iterator] = None
        self._resume_batches_in_epoch = max(0, int(batches_to_skip))
        self._batches_in_epoch = self._resume_batches_in_epoch
        self._failed_samples = 0
        self._total_samples = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def batches_in_epoch(self) -> int:
        return self._batches_in_epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "batches_in_epoch": self._batches_in_epoch}

    def __iter__(self) -> "InfiniteBatchIterator":
        return self

    def __next__(self) -> Any:
        while True:
            if self._iter is None:
                self._start_new_epoch()

            try:
                batch, gotit = next(self._iter)
            except StopIteration:
                self._epoch += 1
                self._start_new_epoch()
                batch, gotit = next(self._iter)

            self._batches_in_epoch += 1

            n_failed = sum(1 for g in gotit if not g)
            self._total_samples += len(gotit)
            if n_failed > 0:
                self._failed_samples += n_failed
            if self._batches_in_epoch % 100 == 0 and self._total_samples > 0:
                rate = self._failed_samples / self._total_samples
                logging.info(
                    f"[{self.name}] sample failure rate: {rate:.2%} "
                    f"({self._failed_samples}/{self._total_samples})"
                )

            if not all(gotit):
                continue

            return batch

    def _start_new_epoch(self) -> None:
        batch_size = getattr(self.dataloader, "batch_size", None) or 1
        start_index = self._resume_batches_in_epoch * int(batch_size)

        if hasattr(self.dataloader.dataset, "set_epoch"):
            self.dataloader.dataset.set_epoch(self._epoch)
        if hasattr(self.dataloader, "sampler") and hasattr(
            self.dataloader.sampler, "set_epoch"
        ):
            self.dataloader.sampler.set_epoch(self._epoch)
        if start_index and hasattr(self.dataloader.sampler, "set_start_index"):
            self.dataloader.sampler.set_start_index(start_index)

        self._iter = iter(self.dataloader)
        self._batches_in_epoch = self._resume_batches_in_epoch
        self._resume_batches_in_epoch = 0
        if start_index:
            logging.info(
                f"[{self.name}] starting epoch {self._epoch} at local batch "
                f"{self._batches_in_epoch}"
            )
        else:
            logging.info(f"[{self.name}] starting epoch {self._epoch}")
