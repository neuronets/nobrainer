"""Unit tests for nobrainer.qc.corruption_configs."""

from __future__ import annotations

import pytest


class TestCorruptionConfig:
    def test_get_all_configs(self):
        from nobrainer.qc.corruption_configs import get_corruption_configs

        configs = get_corruption_configs()
        assert len(configs) == 8
        assert "motion" in configs
        assert "noise" in configs

    def test_all_have_five_severities(self):
        from nobrainer.qc.corruption_configs import get_corruption_configs

        for name, config in get_corruption_configs().items():
            assert set(config.severity_params.keys()) == {
                1,
                2,
                3,
                4,
                5,
            }, f"{name} missing severities"

    def test_kspace_domain_count(self):
        from nobrainer.qc.corruption_configs import get_corruption_configs

        kspace = [c for c in get_corruption_configs().values() if c.domain == "kspace"]
        assert len(kspace) == 3  # motion, ghosting, spike

    def test_image_domain_count(self):
        from nobrainer.qc.corruption_configs import get_corruption_configs

        image = [c for c in get_corruption_configs().values() if c.domain == "image"]
        assert len(image) == 5  # noise, bias_field, blur, downsample, gamma

    def test_get_transform_returns_callable(self):
        from nobrainer.qc.corruption_configs import get_corruption_configs

        config = get_corruption_configs()["noise"]
        transform = config.get_transform(severity=1)
        assert callable(transform)

    def test_invalid_severity_raises(self):
        from nobrainer.qc.corruption_configs import get_corruption_configs

        config = get_corruption_configs()["motion"]
        with pytest.raises(ValueError, match="Severity 0 not defined"):
            config.get_transform(severity=0)

    def test_frozen_dataclass(self):
        from nobrainer.qc.corruption_configs import get_corruption_configs

        config = get_corruption_configs()["motion"]
        with pytest.raises(AttributeError):
            config.name = "changed"

    def test_all_transforms_instantiate(self):
        """Every config at every severity should produce a valid transform."""
        from nobrainer.qc.corruption_configs import get_corruption_configs

        for name, config in get_corruption_configs().items():
            for sev in range(1, 6):
                transform = config.get_transform(sev)
                assert callable(transform), f"{name} sev {sev} not callable"
