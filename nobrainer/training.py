"""Training utilities with optional multi-GPU DDP support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def fit(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 10,
    gpus: int = 1,
    checkpoint_dir: str | Path | None = None,
    callbacks: list[Any] | None = None,
) -> dict:
    """Train a model with optional multi-GPU DDP.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    loader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function.
    optimizer : Optimizer
        PyTorch optimizer.
    max_epochs : int
        Number of training epochs.
    gpus : int
        Number of GPUs to use (1 = single GPU/CPU, >1 = DDP).
    checkpoint_dir : path or None
        Directory for saving checkpoints. None disables checkpointing.
    callbacks : list or None
        Optional callback functions called after each epoch with
        signature ``callback(epoch, loss, model)``.

    Returns
    -------
    dict with keys: final_loss, best_loss, epochs_completed, checkpoint_path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gpus > 1 and torch.cuda.device_count() >= gpus:
        return _fit_ddp(
            model,
            loader,
            criterion,
            optimizer,
            max_epochs,
            gpus,
            checkpoint_dir,
            callbacks,
        )

    model = model.to(device)
    best_loss = float("inf")
    final_loss = 0.0
    ckpt_path = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        final_loss = avg_loss

        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_dir is not None:
                ckpt_path = str(checkpoint_dir / "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)

        if callbacks:
            for cb in callbacks:
                cb(epoch, avg_loss, model)

        logger.debug("Epoch %d/%d: loss=%.4f", epoch + 1, max_epochs, avg_loss)

    return {
        "final_loss": final_loss,
        "best_loss": best_loss,
        "epochs_completed": max_epochs,
        "checkpoint_path": ckpt_path,
    }


def _fit_ddp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    gpus: int,
    checkpoint_dir: str | Path | None,
    callbacks: list[Any] | None,
) -> dict:
    """Multi-GPU training via DistributedDataParallel.

    Launches ``gpus`` processes via ``mp.spawn``.  Each process trains
    on its assigned GPU with a ``DistributedSampler``.
    """
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    results: dict = {}

    def _worker(rank: int) -> None:
        dist.init_process_group("nccl", rank=rank, world_size=gpus)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        local_model = model.to(device)
        ddp_model = DDP(local_model, device_ids=[rank])

        sampler = DistributedSampler(loader.dataset, num_replicas=gpus, rank=rank)
        ddp_loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=True,
        )

        best_loss = float("inf")
        final_loss = 0.0
        ckpt_path = None

        if checkpoint_dir is not None and rank == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(max_epochs):
            sampler.set_epoch(epoch)
            ddp_model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in ddp_loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                elif isinstance(batch, (list, tuple)):
                    images = batch[0].to(device)
                    labels = batch[1].to(device)
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")

                optimizer.zero_grad()
                pred = ddp_model(images)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss

            if rank == 0 and avg_loss < best_loss:
                best_loss = avg_loss
                if checkpoint_dir is not None:
                    ckpt_path = str(Path(checkpoint_dir) / "best_model.pth")
                    torch.save(ddp_model.module.state_dict(), ckpt_path)

        if rank == 0:
            results.update(
                {
                    "final_loss": final_loss,
                    "best_loss": best_loss,
                    "epochs_completed": max_epochs,
                    "checkpoint_path": ckpt_path,
                }
            )

        dist.destroy_process_group()

    import os

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    mp.spawn(_worker, nprocs=gpus, join=True)
    return results


__all__ = ["fit"]
