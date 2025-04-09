from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module, field
from jax.scipy.stats import norm
from jaxtyping import Array

from .data import SpatialData
from .kernels import Kernel


def get_freqs_1D(n_modes):
    if n_modes % 2 == 0:
        return jnp.arange(-n_modes // 2, n_modes // 2, dtype=float)
    else:
        return jnp.arange(-(n_modes - 1) // 2, (n_modes - 1) // 2 + 1, dtype=float)


def get_freqs(n_modes, n_dim):
    if n_dim == 1:
        return get_freqs_1D(n_modes)
    else:
        assert len(n_modes) == n_dim
        modes_grid = jnp.meshgrid(*[get_freqs_1D(n_modes[i]) for i in range(n_dim)], indexing="ij")
        # Transpose because fiNUFFT treats the first dimension as the fastest changing
        return modes_grid


class SpatialModel(Module):
    @abstractmethod
    def __call__(self, data: SpatialData):
        pass


class FourierGP(SpatialModel):
    n_modes: tuple[int, int]
    coefficients: Array
    kernel: Kernel
    freqs: Array = field(static=True)

    def __init__(self, n_modes: tuple[int, int], kernel: Kernel):
        # Model specfication
        self.n_modes = n_modes
        fx, fy = get_freqs(n_modes, 2)
        self.freqs = jnp.sqrt(fx**2 + fy**2)
        self.kernel = kernel
        # Initialise parameters
        self.coefficients = jnp.zeros(n_modes)

    def __call__(self, data: SpatialData):
        x, y = data.x, data.y
        # Leave this stupid for now
        return x + y

    def prior_logpdf(self):
        prior_stddev = self.kernel.feature_weights(self.freqs)
        return norm.logpdf(
            x=self.coefficients,
            loc=jnp.zeros(self.n_modes),
            scale=prior_stddev,
        )


class PerSpaxel(SpatialModel):
    # Model parameters
    values: Array

    def __init__(self, n_spaxels: int):
        self.values = jnp.zeros(n_spaxels)

    def __call__(self, data: SpatialData):
        return self.values[data.indices]
