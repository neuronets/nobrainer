"""Shared fixtures for somewhat-realistic tests."""

import pytest

from nobrainer.io import read_csv
from nobrainer.utils import get_data


@pytest.fixture(scope="session")
def sample_data():
    """Download sample brain data once per test session."""
    csv_path = get_data()
    return read_csv(csv_path)


@pytest.fixture(scope="session")
def train_eval_split(sample_data):
    """Split into 9 train + 1 eval."""
    return sample_data[:9], sample_data[9]
