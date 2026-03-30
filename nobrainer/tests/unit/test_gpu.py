"""Unit tests for nobrainer.gpu utilities."""

from __future__ import annotations

import torch

from nobrainer.gpu import get_device, gpu_count, gpu_info, scale_for_multi_gpu


class TestGetDevice:
    def test_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device)

    def test_device_type_known(self):
        d = get_device()
        assert d.type in ("cuda", "mps", "cpu")


class TestGpuCount:
    def test_returns_int(self):
        n = gpu_count()
        assert isinstance(n, int)
        assert n >= 0


class TestGpuInfo:
    def test_returns_list(self):
        info = gpu_info()
        assert isinstance(info, list)
        if torch.cuda.is_available():
            assert len(info) > 0
            assert "name" in info[0]
            assert "memory_gb" in info[0]


class TestScaleForMultiGpu:
    def test_no_gpu_returns_base(self):
        if torch.cuda.is_available():
            return  # skip on GPU machines
        eff, per, n = scale_for_multi_gpu(base_batch_size=32)
        assert eff == 32
        assert per == 32
        assert n == 0

    def test_simple_division(self):
        # Without model, just divides
        eff, per, n = scale_for_multi_gpu(base_batch_size=32)
        if n > 0:
            assert eff == per * n
