"""conftest.py - common fixtures for modelling_lib tests."""

import tempfile
from pathlib import Path

import jax
import pytest

# Float64 throughout, so cross-backend comparisons can be tight
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
