"""conftest.py - common fixtures for modelling_lib tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
