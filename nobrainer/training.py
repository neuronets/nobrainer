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
    """Run one validation pass.

    Returns dict with val_loss, val_acc (overall), and val_bal_acc
    (balanced accuracy — mean of per-class recall).
    """
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    n_batches = 0
    # Per-class correct/total for balanced accuracy
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

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
            # Handle model parallel: pred may be on different device
            if pred.device != labels.device:
                labels = labels.to(pred.device)
            total_loss += criterion(pred, labels).item()

            pred_labels = pred.argmax(1)
            correct_mask = pred_labels == labels
            n_correct += correct_mask.sum().item()
            n_total += labels.numel()
            n_batches += 1

            # Accumulate per-class stats
            for c in labels.unique().tolist():
                mask = labels == c
                cc = correct_mask[mask].sum().item()
                ct = mask.sum().item()
                class_correct[c] = class_correct.get(c, 0) + cc
                class_total[c] = class_total.get(c, 0) + ct

    val_loss = total_loss / max(n_batches, 1)
    val_acc = n_correct / max(n_total, 1)

    # Balanced accuracy: mean recall per class
    per_class_recall = []
    for c in sorted(class_total.keys()):
        if class_total[c] > 0:
            per_class_recall.append(class_correct[c] / class_total[c])
    val_bal_acc = sum(per_class_recall) / max(len(per_class_recall), 1)

    return {"val_loss": val_loss, "val_acc": val_acc, "val_bal_acc": val_bal_acc}


def _apply_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing on sequential layers to save memory.

    Wraps each layer's forward in ``torch.utils.checkpoint.checkpoint``
    so intermediate activations are recomputed during backward instead
    of stored. Roughly halves activation memory at ~30% compute cost.
    """
    from torch.utils.checkpoint import checkpoint

    for name, module in model.named_children():
        orig_forward = module.forward

        def _make_ckpt_forward(fwd):
            def _ckpt_forward(*args, **kwargs):
                # checkpoint requires at least one tensor with requires_grad
                def run(*a):
                    return fwd(*a, **kwargs)

                tensors = [a for a in args if isinstance(a, torch.Tensor)]
                if tensors and any(t.requires_grad for t in tensors):
                    return checkpoint(run, *args, use_reentrant=False)
                return fwd(*args, **kwargs)

            return _ckpt_forward

        module.forward = _make_ckpt_forward(orig_forward)
    logger.info("Gradient checkpointing enabled on %d modules", len(list(model.children())))


def _apply_model_parallel(model: nn.Module, gpus: int) -> nn.Module:
    """Distribute model layers across multiple GPUs (pipeline parallelism).

    Splits the model's children into ``gpus`` roughly equal groups and
    places each group on a different GPU. Inserts device-transfer hooks
    between groups so tensors move between GPUs automatically.

    Parameters
    ----------
    model : nn.Module
        Model with sequential children (e.g., KWYKMeshNet).
    gpus : int
        Number of GPUs to distribute across.

    Returns
    -------
    nn.Module
        The model with layers placed on different GPUs and transfer hooks.
    """
    children = list(model.named_children())
    if not children:
        logger.warning("Model has no children — placing on GPU 0")
        return model.to("cuda:0")

    # Split children into roughly equal groups
    n = len(children)
    group_size = max(1, (n + gpus - 1) // gpus)
    groups: list[list[tuple[str, nn.Module]]] = []
    for i in range(0, n, group_size):
        groups.append(children[i : i + group_size])

    # Place each group on its GPU
    device_map: dict[str, int] = {}
    for gpu_idx, group in enumerate(groups):
        device = torch.device(f"cuda:{gpu_idx}")
        for name, module in group:
            module.to(device)
            device_map[name] = gpu_idx
    logger.info(
        "Model parallel: %d layers across %d GPUs: %s",
        n,
        min(gpus, len(groups)),
        {k: f"cuda:{v}" for k, v in device_map.items()},
    )

    # Wrap forward to move tensors between devices
    orig_forward = model.forward

    def _mp_forward(*args, **kwargs):
        # Move input to first device
        first_device = torch.device(f"cuda:{groups[0][0][1].weight.device.index if hasattr(groups[0][0][1], 'weight') else 0}")
        new_args = tuple(
            a.to(first_device) if isinstance(a, torch.Tensor) else a for a in args
        )
        return orig_forward(*new_args, **kwargs)

    model.forward = _mp_forward

    # Add hooks to move activations between GPUs at group boundaries
    for gpu_idx, group in enumerate(groups):
        if gpu_idx == 0:
            continue
        target_device = torch.device(f"cuda:{gpu_idx}")
        first_module = group[0][1]

        def _make_hook(dev):
            def _hook(module, inputs):
                return tuple(
                    x.to(dev) if isinstance(x, torch.Tensor) else x for x in inputs
                )

            return _hook

        first_module.register_forward_pre_hook(_make_hook(target_device))

    return model


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
    checkpoint_freq: int = 0,
    gradient_checkpointing: bool = False,
    model_parallel: bool = False,
    resume_from: str | Path | None = None,
) -> dict:
    """Train a model with optional multi-GPU DDP or model parallelism.

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
        Number of GPUs to use (1 = single GPU/CPU, >1 = DDP or model parallel).
    checkpoint_dir : path or None
        Directory for saving checkpoints. None disables checkpointing.
    callbacks : list or None
        Optional callback functions called after each epoch with
        signature ``callback(epoch, logs, model)`` where logs is a dict
        containing at minimum ``{"loss": float}``.
    val_loader : DataLoader or None
        Validation data loader. If provided, validation loss and accuracy
        are computed each epoch and included in the logs dict.
    checkpoint_freq : int
        Save a checkpoint every N epochs (in addition to best model).
        0 = only save best model. Checkpoints are saved as
        ``epoch_NNN.pth`` in checkpoint_dir.
    gradient_checkpointing : bool
        If True, trade compute for memory by recomputing activations
        during backward. Roughly halves activation memory.
    model_parallel : bool
        If True and gpus > 1, distribute layers across GPUs (pipeline
        parallelism) instead of DDP. Useful when a single batch is too
        large for one GPU.

    Returns
    -------
    dict with keys: final_loss, best_loss, epochs_completed, checkpoint_path,
    train_losses, val_losses, checkpoint_epochs
    """
    device = get_device()

    # Apply gradient checkpointing if requested
    if gradient_checkpointing:
        _apply_gradient_checkpointing(model)

    # Multi-GPU dispatch
    if gpus > 1 and torch.cuda.device_count() >= gpus:
        if model_parallel:
            # Pipeline parallelism: split layers across GPUs
            model = _apply_model_parallel(model, gpus)
            device = torch.device("cuda:0")  # input goes to first GPU
            # Fall through to single-process training loop below
        else:
            # Data parallelism: DDP
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
                checkpoint_freq,
            )

    if not model_parallel:
        model = model.to(device)

    best_loss = float("inf")
    ckpt_path = None
    history: list[dict[str, Any]] = []  # one entry per epoch
    start_epoch = 0

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint (auto-detect or explicit path)
    resume_path = None
    if resume_from is not None:
        resume_path = Path(resume_from)
    elif checkpoint_dir is not None and (checkpoint_dir / "checkpoint.pt").exists():
        resume_path = checkpoint_dir / "checkpoint.pt"

    if resume_path is not None and resume_path.exists():
        from nobrainer.slurm import load_checkpoint as _load_ckpt

        ckpt_dir = (
            resume_path.parent
            if resume_path.name == "checkpoint.pt"
            else checkpoint_dir
        )
        start_epoch, prev_metrics = _load_ckpt(ckpt_dir, model, optimizer)
        history = prev_metrics.get("history", [])
        best_loss = min((h["loss"] for h in history), default=float("inf"))
        logger.info(
            "Resumed from epoch %d (%d history entries, best_loss=%.4f)",
            start_epoch, len(history), best_loss,
        )

    for epoch in range(start_epoch, max_epochs):
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
            if pred.device != labels.device:
                labels = labels.to(pred.device)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Epoch metrics
        logs: dict[str, Any] = {"epoch": epoch + 1, "loss": avg_loss}

        if val_loader is not None:
            logs.update(_run_validation(model, val_loader, criterion, device))
            model.train()

        history.append(logs)

        # Best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_dir is not None:
                ckpt_path = str(checkpoint_dir / "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)

        # Resumable checkpoint (every epoch)
        if checkpoint_dir is not None:
            from nobrainer.slurm import save_checkpoint as _save_ckpt

            _save_ckpt(checkpoint_dir, model, optimizer, epoch + 1,
                       {"history": history})

        # Named checkpoint (for post-hoc Dice eval)
        if (
            checkpoint_dir is not None
            and checkpoint_freq > 0
            and (epoch + 1) % checkpoint_freq == 0
        ):
            epoch_ckpt = checkpoint_dir / f"epoch_{epoch + 1:03d}.pth"
            torch.save(model.state_dict(), epoch_ckpt)

        if callbacks:
            for cb in callbacks:
                cb(epoch, logs, model)

        logger.debug("Epoch %d/%d: %s", epoch + 1, max_epochs,
                     " ".join(f"{k}={v:.4f}" for k, v in logs.items()
                              if isinstance(v, float)))

    return {"history": history, "checkpoint_path": ckpt_path}


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
    checkpoint_freq: int,
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
    ckpt_path = None
    history: list[dict[str, Any]] = []

    if checkpoint_dir is not None and rank == 0:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ExperimentTracker for live metrics (rank 0 only)
    tracker = None
    if rank == 0 and checkpoint_dir is not None:
        from nobrainer.experiment import ExperimentTracker

        tracker = ExperimentTracker(output_dir=checkpoint_dir)

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
        logs: dict[str, Any] = {"epoch": epoch + 1, "loss": avg_loss}

        if rank == 0:
            if val_loader is not None:
                logs.update(_run_validation(
                    ddp_model.module, val_loader, criterion, device
                ))
                ddp_model.train()

            history.append(logs)

            if avg_loss < best_loss:
                best_loss = avg_loss
                if checkpoint_dir is not None:
                    ckpt_path = str(Path(checkpoint_dir) / "best_model.pth")
                    torch.save(ddp_model.module.state_dict(), ckpt_path)

            # Resumable checkpoint
            if checkpoint_dir is not None:
                from nobrainer.slurm import save_checkpoint as _save_ckpt

                _save_ckpt(checkpoint_dir, ddp_model.module, optimizer,
                           epoch + 1, {"history": history})

            # Named checkpoint for post-hoc Dice eval
            if (
                checkpoint_dir is not None
                and checkpoint_freq > 0
                and (epoch + 1) % checkpoint_freq == 0
            ):
                torch.save(ddp_model.module.state_dict(),
                           Path(checkpoint_dir) / f"epoch_{epoch + 1:03d}.pth")

            if tracker is not None:
                tracker.log(logs)

            logger.info("Epoch %d/%d: %s", epoch + 1, max_epochs,
                        " ".join(f"{k}={v:.4f}" for k, v in logs.items()
                                 if isinstance(v, float)))

    if rank == 0:
        result_dict["history"] = history
        result_dict["checkpoint_path"] = ckpt_path

    if tracker is not None:
        tracker.finish()

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
    checkpoint_freq: int = 0,
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
            checkpoint_freq,
            results,
        ),
        nprocs=gpus,
        join=True,
    )

    result = dict(results)
    if "history" in result:
        result["history"] = list(result["history"])
    return result


__all__ = ["fit"]
