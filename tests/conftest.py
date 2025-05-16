"""Common fixtures for modelling_lib tests."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest
from modelling_lib.data import SpatialData
from modelling_lib.parameter import Parameter


@pytest.fixture
def simple_parameter():
    """Basic parameter with value 1.0"""
    return Parameter(initial=1.0)


@pytest.fixture
def array_parameter():
    """Parameter with array value"""
    return Parameter(initial=jnp.array([1.0, 2.0, 3.0]))


@pytest.fixture
def fixed_parameter():
    """Fixed parameter that shouldn't change during optimization"""
    return Parameter(initial=5.0, fixed=True)


@pytest.fixture
def spatial_data():
    """Simple spatial data for testing"""
    return SpatialData(
        x=jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        y=jnp.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        indices=jnp.array([0, 1, 2, 3, 4]),
    )


@pytest.fixture
def large_spatial_data():
    """Larger spatial data for more complex tests"""
    n = 100
    return SpatialData(x=jnp.linspace(0, 1, n), y=jnp.linspace(0, 1, n), indices=jnp.arange(n))


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
