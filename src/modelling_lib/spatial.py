from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module
from jax.scipy.stats import norm
from jax_finufft import nufft2
from jaxtyping import Array

from .data import SpatialData
from .kernels import Kernel
from .parameter import Parameter

# NOTE: List of current obvious foot guns:
# - n_modes must always be two odd integers, but this is not enforced
#   - in theory the nifty-solve full implementation should be able to handle
#     this, but from memory it doesn't work right now. This is why p is
#     repeated in the shape info right now, since I didn't implement n_modes
#     != n_requested_modes yet.

FINUFFT_KWDS = dict(eps=1e-6)


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
    coefficients: Array
    kernel: Kernel
    n_modes: tuple[int, int]
    _freqs: Array
    _shape_info: tuple[int, int, int]

    def __init__(self, n_modes: tuple[int, int], kernel: Kernel):
        # Model specfication
        self.n_modes = n_modes
        fx, fy = get_freqs(n_modes, 2)
        self._freqs = jnp.sqrt(fx**2 + fy**2)
        self.kernel = kernel
        # Initialise parameters
        self.coefficients = Parameter(dims=n_modes)
        # Initialise the shape info
        p = int(jnp.prod(jnp.array(n_modes)))
        self._shape_info = (p, p // 2, p)

    def __call__(self, data: SpatialData):
        # Feature weighted coefficients
        scaled_coeffs = self.coefficients * self.kernel.feature_weights(self._freqs)
        # Sum basis functions with nufft after processing the coefficients to enforce conjugate symmetry
        return nufft2(
            self._conj_symmetry(scaled_coeffs.flatten()), data.x, data.y, **FINUFFT_KWDS
        ).real

    def _conj_symmetry(self, c):
        m, h, p = self._shape_info
        f = 0.5 * jnp.hstack(
            [c[: h + 1], jnp.zeros(p - h - 1)],
        ) + 0.5j * jnp.hstack(
            [jnp.zeros(p - m + h + 1), c[h + 1 :]],
        )
        f = f.reshape(self.n_modes)
        return f + jnp.conj(jnp.flip(f))

    def prior_logpdf(self):
        return norm.logpdf(x=self.coefficients)


class PerSpaxel(SpatialModel):
    # Model parameters
    values: Array

    def __init__(self, n_spaxels: int):
        self.values = Parameter(dims=n_spaxels)

    def __call__(self, data: SpatialData):
        return self.values[data.indices]
