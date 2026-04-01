"""Training utilities with optional multi-GPU DDP support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU.

    .. note::
       Also available as :func:`nobrainer.gpu.get_device`.
    """
    from nobrainer.gpu import get_device as _get_device

    return _get_device()


def _run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation pass. Returns dict with val_loss and val_acc."""
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                continue

            if labels.ndim == images.ndim and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype in (torch.float32, torch.float64):
                labels = labels.long()

            pred = model(images)
            total_loss += criterion(pred, labels).item()
            n_correct += (pred.argmax(1) == labels).sum().item()
            n_total += labels.numel()
            n_batches += 1

    val_loss = total_loss / max(n_batches, 1)
    val_acc = n_correct / max(n_total, 1)
    return {"val_loss": val_loss, "val_acc": val_acc}


def fit(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 10,
    gpus: int = 1,
    checkpoint_dir: str | Path | None = None,
    callbacks: list[Any] | None = None,
    val_loader: DataLoader | None = None,
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
        signature ``callback(epoch, logs, model)`` where logs is a dict
        containing at minimum ``{"loss": float}``.
    val_loader : DataLoader or None
        Validation data loader. If provided, validation loss and accuracy
        are computed each epoch and included in the logs dict.

    Returns
    -------
    dict with keys: final_loss, best_loss, epochs_completed, checkpoint_path,
    train_losses, val_losses
    """
    device = get_device()

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
            val_loader,
        )

    model = model.to(device)
    best_loss = float("inf")
    final_loss = 0.0
    ckpt_path = None
    train_losses: list[float] = []
    val_losses: list[float] = []

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

            if labels.ndim == images.ndim and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype in (torch.float32, torch.float64):
                labels = labels.long()

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        final_loss = avg_loss
        train_losses.append(avg_loss)

        # Build logs dict for callbacks
        logs: dict[str, Any] = {"loss": avg_loss, "epoch": epoch}

        # Validation (built-in, no callback needed)
        if val_loader is not None:
            val_metrics = _run_validation(model, val_loader, criterion, device)
            logs.update(val_metrics)
            val_losses.append(val_metrics["val_loss"])
            model.train()

        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_dir is not None:
                ckpt_path = str(checkpoint_dir / "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)

        if callbacks:
            for cb in callbacks:
                cb(epoch, logs, model)

        logger.debug(
            "Epoch %d/%d: loss=%.4f%s",
            epoch + 1,
            max_epochs,
            avg_loss,
            f" val_loss={logs['val_loss']:.4f} val_acc={logs['val_acc']:.4f}"
            if "val_loss" in logs
            else "",
        )

    return {
        "final_loss": final_loss,
        "best_loss": best_loss,
        "epochs_completed": max_epochs,
        "checkpoint_path": ckpt_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def _ddp_worker(
    rank: int,
    world_size: int,
    model: nn.Module,
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    checkpoint_dir: str | Path | None,
    result_dict: dict,
) -> None:
    """Single DDP worker — module-level function for mp.spawn pickling."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    local_model = model.to(device)
    ddp_model = DDP(local_model, device_ids=[rank])

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    ddp_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Validation loader on rank 0 only (no DDP sampler needed)
    val_loader = None
    if val_dataset is not None and rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    best_loss = float("inf")
    final_loss = 0.0
    ckpt_path = None
    train_losses: list[float] = []
    val_losses: list[float] = []

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

            if labels.ndim == images.ndim and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype in (torch.float32, torch.float64):
                labels = labels.long()

            optimizer.zero_grad()
            pred = ddp_model(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        final_loss = avg_loss
        train_losses.append(avg_loss)

        # Validation on rank 0
        val_msg = ""
        if rank == 0 and val_loader is not None:
            val_metrics = _run_validation(
                ddp_model.module, val_loader, criterion, device
            )
            val_losses.append(val_metrics["val_loss"])
            val_msg = (
                f" val_loss={val_metrics['val_loss']:.4f}"
                f" val_acc={val_metrics['val_acc']:.4f}"
            )
            ddp_model.train()

        if rank == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                if checkpoint_dir is not None:
                    ckpt_path = str(Path(checkpoint_dir) / "best_model.pth")
                    torch.save(ddp_model.module.state_dict(), ckpt_path)
            logger.info(
                "Epoch %d/%d: loss=%.4f%s",
                epoch + 1,
                max_epochs,
                avg_loss,
                val_msg,
            )

    if rank == 0:
        result_dict.update(
            {
                "final_loss": final_loss,
                "best_loss": best_loss,
                "epochs_completed": max_epochs,
                "checkpoint_path": ckpt_path,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
        )

    dist.destroy_process_group()


def _fit_ddp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    gpus: int,
    checkpoint_dir: str | Path | None,
    callbacks: list[Any] | None,
    val_loader: DataLoader | None = None,
) -> dict:
    """Multi-GPU training via DistributedDataParallel.

    Launches ``gpus`` processes via ``mp.spawn``.  Validation runs on
    rank 0 inside the worker — no callbacks needed for val metrics.
    """
    import os

    import torch.multiprocessing as mp

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    results: dict = mp.Manager().dict()

    # Extract datasets (picklable) — not DataLoaders (may have closures)
    train_dataset = loader.dataset
    val_dataset = val_loader.dataset if val_loader is not None else None

    mp.spawn(
        _ddp_worker,
        args=(
            gpus,
            model,
            train_dataset,
            val_dataset,
            loader.batch_size,
            loader.num_workers,
            criterion,
            optimizer,
            max_epochs,
            checkpoint_dir,
            results,
        ),
        nprocs=gpus,
        join=True,
    )

    result = dict(results)

    # Run callbacks on the returned results (in parent process, no pickling needed)
    if callbacks and result.get("train_losses"):
        for epoch, loss in enumerate(result["train_losses"]):
            logs = {"loss": loss, "epoch": epoch}
            if result.get("val_losses") and epoch < len(result["val_losses"]):
                logs["val_loss"] = result["val_losses"][epoch]
            for cb in callbacks:
                cb(epoch, logs, model)

    return result


__all__ = ["fit"]
