from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar, field
from jax.tree_util import register_dataclass

from .kernels import Kernel

# @partial(register_dataclass, meta_fields=[], data_fields=["x", "y"])
# @dataclass
# class SpatialData:
#     x: jax.Array
#     y: jax.Array
#     n_spaxels: Optional[int] = field(init=False)
#     indices: Optional[jax.Array] = field(init=False)

#     def __post_init__(self):
#         # Make sure x and y are 1D arrays
#         self.x = jnp.asarray(self.x).flatten()
#         self.y = jnp.asarray(self.y).flatten()
#         # Check that x and y are the same length
#         if self.x.shape != self.y.shape:
#             raise ValueError("x and y must be the same length")
#         # Set the number of spaxels
#         self.n_spaxels = len(self.x)
#         # Set the indices
#         self.indices = jnp.arange(self.n_spaxels)


class SpatialData(eqx.Module):
    x: jax.Array
    y: jax.Array
    n_spaxels: Optional[int] = field(init=False)
    indices: Optional[jax.Array] = field(init=False)

    def __post_init__(self):
        # Make sure x and y are 1D arrays
        self.x = jnp.asarray(self.x).flatten()
        self.y = jnp.asarray(self.y).flatten()
        # Check that x and y are the same length
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must be the same length")
        # Set the number of spaxels
        n_spaxels = len(self.x)
        self.n_spaxels = n_spaxels
        # Set the indices
        self.indices = jnp.arange(n_spaxels)


class SpatialModel(eqx.Module):
    @abstractmethod
    def __call__(self, data: SpatialData):
        pass


class FourierGP(SpatialModel):
    n_modes: tuple[int, int]
    coefficients: jax.Array
    kernel: Kernel

    def __init__(self, n_modes: tuple[int, int], kernel: Kernel):
        # Model specfication
        self.n_modes = n_modes
        self.kernel = kernel
        # Initialise parameters
        self.coefficients = jnp.ones(n_modes).flatten()

    def __call__(self, data: SpatialData):
        x, y = data.x, data.y
        # Leave this stupid for now
        return x * y


class PerSpaxel(SpatialModel):
    # Model parameters
    values: jax.Array

    def __init__(self, n_spaxels: int):
        self.values = jnp.zeros(n_spaxels)

    def __call__(self, data: SpatialData):
        return self.values[data.indices]
