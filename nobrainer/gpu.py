"""GPU utilities: device detection, memory profiling, batch size optimization.

Examples
--------
Auto-select the best batch size for a model and block shape::

    from nobrainer.gpu import auto_batch_size, gpu_info

    info = gpu_info()
    print(info)
    # [{'name': 'Tesla T4', 'memory_gb': 15.1, 'id': 0}, ...]

    batch_size = auto_batch_size(
        model=my_model,
        block_shape=(32, 32, 32),
        n_classes=2,
        target_memory_fraction=0.85,
    )
    print(f"Optimal batch size: {batch_size}")

Scale batch size for multi-GPU::

    from nobrainer.gpu import scale_for_multi_gpu

    effective_batch, per_gpu_batch, n_gpus = scale_for_multi_gpu(
        base_batch_size=32,
        block_shape=(32, 32, 32),
    )
    # On 4x T4: effective=128, per_gpu=32, n_gpus=4
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_count() -> int:
    """Return the number of CUDA GPUs available (0 if none)."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def gpu_info() -> list[dict[str, Any]]:
    """Return a list of dicts with GPU name, memory, and id.

    Returns an empty list if no CUDA GPUs are available.
    """
    info = []
    if not torch.cuda.is_available():
        return info
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append(
            {
                "id": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / 1e9, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def _estimate_memory_per_sample(
    model: nn.Module,
    block_shape: tuple[int, int, int],
    n_classes: int = 2,
    in_channels: int = 1,
    dtype: torch.dtype = torch.float32,
    forward_kwargs: dict | None = None,
) -> float:
    """Estimate GPU memory (bytes) for one training sample.

    Runs a forward + backward pass with batch_size=1 and measures the
    peak allocated memory.  The model is moved to GPU temporarily.

    Parameters
    ----------
    model : nn.Module
        Model to profile.
    block_shape : tuple of int
        Spatial dimensions of one input patch.
    n_classes : int
        Number of output classes.
    in_channels : int
        Number of input channels.
    dtype : torch.dtype
        Input data type.

    Returns
    -------
    float
        Estimated bytes per sample (forward + backward + optimizer overhead).
    """
    if forward_kwargs is None:
        forward_kwargs = {}

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for memory estimation")

    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    baseline = torch.cuda.memory_allocated(device)

    x = torch.randn(1, in_channels, *block_shape, device=device, dtype=dtype)
    labels = torch.randint(0, n_classes, (1, *block_shape), device=device)

    # Pass forward_kwargs if model accepts them (e.g. mc_vwn, mc_dropout)
    try:
        out = model(x, **forward_kwargs)
    except TypeError:
        out = model(x)
    loss = nn.CrossEntropyLoss()(out, labels)
    loss.backward()

    peak = torch.cuda.max_memory_allocated(device) - baseline

    # Clean up
    model.zero_grad(set_to_none=True)
    del x, labels, out, loss
    torch.cuda.empty_cache()
    model.cpu()

    return float(peak)


def auto_batch_size(
    model: nn.Module,
    block_shape: tuple[int, int, int],
    n_classes: int = 2,
    in_channels: int = 1,
    target_memory_fraction: float = 0.85,
    gpu_id: int = 0,
    min_batch: int = 1,
    max_batch: int = 512,
    forward_kwargs: dict | None = None,
) -> int:
    """Estimate the largest batch size that fits in GPU memory.

    Profiles one sample, then scales to fill ``target_memory_fraction``
    of the GPU.

    Parameters
    ----------
    model : nn.Module
        Model to profile (will be temporarily moved to GPU).
    block_shape : tuple of int
        Spatial dimensions ``(D, H, W)`` of one input patch.
    n_classes : int
        Number of output classes.
    in_channels : int
        Number of input channels.
    target_memory_fraction : float
        Fraction of total GPU memory to target (default 0.85).
    gpu_id : int
        Which GPU to profile.
    min_batch : int
        Minimum batch size to return.
    max_batch : int
        Maximum batch size to return.

    Returns
    -------
    int
        Recommended batch size for one GPU.
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA — returning min_batch=%d", min_batch)
        return min_batch

    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    target_mem = total_mem * target_memory_fraction

    try:
        mem_per_sample = _estimate_memory_per_sample(
            model,
            block_shape,
            n_classes,
            in_channels,
            forward_kwargs=forward_kwargs,
        )
    except RuntimeError as e:
        logger.warning("Memory estimation failed: %s — returning min_batch", e)
        return min_batch

    # Account for ~20% overhead (optimizer state, fragmentation)
    effective_per_sample = mem_per_sample * 1.2
    batch = int(target_mem / effective_per_sample)
    batch = max(min_batch, min(batch, max_batch))

    logger.info(
        "auto_batch_size: %.1f GB total, %.1f MB/sample, "
        "target %.0f%% → batch_size=%d",
        total_mem / 1e9,
        mem_per_sample / 1e6,
        target_memory_fraction * 100,
        batch,
    )
    return batch


def scale_for_multi_gpu(
    base_batch_size: int,
    block_shape: tuple[int, int, int] | None = None,
    model: nn.Module | None = None,
    n_classes: int = 2,
    target_memory_fraction: float = 0.85,
) -> tuple[int, int, int]:
    """Scale batch size for multi-GPU training.

    If ``model`` is provided, uses :func:`auto_batch_size` to determine
    the per-GPU batch size based on actual memory profiling.  Otherwise,
    divides ``base_batch_size`` evenly across available GPUs.

    Parameters
    ----------
    base_batch_size : int
        Desired effective (global) batch size.
    block_shape : tuple of int, optional
        Spatial dimensions for memory profiling.
    model : nn.Module, optional
        Model for memory profiling.  If None, uses simple division.
    n_classes : int
        Number of output classes (for profiling).
    target_memory_fraction : float
        Target GPU memory fraction (for profiling).

    Returns
    -------
    effective_batch : int
        Total batch size across all GPUs.
    per_gpu_batch : int
        Batch size per GPU.
    n_gpus : int
        Number of GPUs to use.
    """
    n_gpus = gpu_count()
    if n_gpus == 0:
        return base_batch_size, base_batch_size, 0

    if model is not None and block_shape is not None:
        per_gpu = auto_batch_size(
            model,
            block_shape,
            n_classes=n_classes,
            target_memory_fraction=target_memory_fraction,
        )
    else:
        per_gpu = max(1, base_batch_size // n_gpus)

    effective = per_gpu * n_gpus

    logger.info(
        "Multi-GPU scaling: %d GPUs × %d per-GPU = %d effective batch",
        n_gpus,
        per_gpu,
        effective,
    )
    return effective, per_gpu, n_gpus
