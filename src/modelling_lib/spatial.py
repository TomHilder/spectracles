from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array

from .data import SpatialData
from .kernels import Kernel


class SpatialModel(Module):
    @abstractmethod
    def __call__(self, data: SpatialData):
        pass


class FourierGP(SpatialModel):
    n_modes: tuple[int, int]
    coefficients: Array
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
        return x + y


class PerSpaxel(SpatialModel):
    # Model parameters
    values: Array

    def __init__(self, n_spaxels: int):
        self.values = jnp.zeros(n_spaxels)

    def __call__(self, data: SpatialData):
        return self.values[data.indices]
