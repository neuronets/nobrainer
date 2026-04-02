"""Unit tests for pyramid-level-to-GAN-stage mapping (T025)."""

from __future__ import annotations

from pathlib import Path
import sys

# Add scripts to path so we can import the mapping function
sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "scripts" / "progressive_gan")
)
from train import map_pyramid_to_gan_stages  # noqa: E402


class TestMapPyramidToGanStages:
    def test_exact_match_4_levels_4_stages(self):
        """4 pyramid levels, 4 GAN stages → each stage maps to a level."""
        mapping = map_pyramid_to_gan_stages(
            n_pyramid_levels=4,
            resolution_schedule=[4, 8, 16, 32],
            volume_shape=(32, 32, 32),
        )
        assert len(mapping) == 4
        # Stage 0 (coarsest GAN) → highest level (coarsest pyramid)
        assert mapping[0] > mapping[-1]
        # Stage -1 (finest GAN) → level 0 (finest pyramid)
        assert mapping[-1] == 0

    def test_fewer_levels_than_stages(self):
        """2 pyramid levels, 4 GAN stages → reuse finest for extra stages."""
        mapping = map_pyramid_to_gan_stages(
            n_pyramid_levels=2,
            resolution_schedule=[4, 8, 16, 32],
            volume_shape=(32, 32, 32),
        )
        assert len(mapping) == 4
        # First stages use coarsest, later stages reuse finest (level 0)
        assert mapping[-1] == 0
        assert mapping[-2] == 0  # reused

    def test_more_levels_than_stages(self):
        """4 pyramid levels, 2 GAN stages → skip coarsest levels."""
        mapping = map_pyramid_to_gan_stages(
            n_pyramid_levels=4,
            resolution_schedule=[4, 8],
            volume_shape=(64, 64, 64),
        )
        assert len(mapping) == 2
        assert mapping[-1] == 0  # finest stage → level 0

    def test_skips_tiny_levels(self):
        """Level with shape < 4³ gets skipped."""
        # volume 16³ with 4 levels → level 3 is 2³ (too small)
        mapping = map_pyramid_to_gan_stages(
            n_pyramid_levels=4,
            resolution_schedule=[4, 8, 16],
            volume_shape=(16, 16, 16),
        )
        assert len(mapping) == 3
        # Level 3 (2³) should not appear in mapping
        assert 3 not in mapping

    def test_single_level_store(self):
        """Single-level store maps all stages to level 0."""
        mapping = map_pyramid_to_gan_stages(
            n_pyramid_levels=1,
            resolution_schedule=[4, 8, 16],
            volume_shape=(64, 64, 64),
        )
        assert mapping == [0, 0, 0]
