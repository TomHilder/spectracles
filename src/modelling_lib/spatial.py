from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field

from .kernels import Kernel


def convert_to_flat_array(array: jax.Array) -> jax.Array:
    return jnp.asarray(array).flatten()


class SpatialData(eqx.Module):
    x: jax.Array = field(converter=convert_to_flat_array)
    y: jax.Array = field(converter=convert_to_flat_array)
    indices: jax.Array = field(converter=convert_to_flat_array)


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
        return x + y


class PerSpaxel(SpatialModel):
    # Model parameters
    values: jax.Array

    def __init__(self, n_spaxels: int):
        self.values = jnp.zeros(n_spaxels)

    def __call__(self, data: SpatialData):
        return self.values[data.indices]
