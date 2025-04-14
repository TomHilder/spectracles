from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module, field
from jax.scipy.stats import norm
from jax_finufft import nufft2
from jaxtyping import Array
from nifty_solve.jax_operators import JaxFinufft2DRealOperator

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


def _pre_matvec(c, n_modes, _shape_half_p):
    m, h, p = _shape_half_p
    f = 0.5 * jnp.hstack([c[: h + 1], jnp.zeros(p - h - 1)]) + 0.5j * jnp.hstack(
        [jnp.zeros(p - m + h + 1), c[h + 1 :]]
    )
    f = f.reshape(n_modes)
    return f + jnp.conj(jnp.flip(f))


# def _matvec(self, c):
#     return nufft2(self._pre_matvec(c), *self.points, **self.finufft_kwds).real

# @cached_property
# def _shape_half_p(self):
# return (self.shape[1], self.shape[1] // 2, int(np.prod(self.n_modes)))


class SpatialModel(Module):
    @abstractmethod
    def __call__(self, data: SpatialData):
        pass


# class FourierGP(SpatialModel):
#     n_modes: tuple[int, int]
#     coefficients: Array
#     kernel: Kernel
#     _freqs: Array = field(static=True)

#     def __init__(self, n_modes: tuple[int, int], kernel: Kernel):
#         # Model specfication
#         self.n_modes = n_modes
#         fx, fy = get_freqs(n_modes, 2)
#         self._freqs = jnp.sqrt(fx**2 + fy**2)
#         self.kernel = kernel
#         # Initialise parameters
#         self.coefficients = jnp.zeros(n_modes)

#     def __call__(self, data: SpatialData):
#         c = self.coefficients.astype(jnp.complex128)
#         # f = 0.5 * (c + jnp.conj(jnp.flip(c))) * self.kernel.feature_weights(self.freqs)
#         f = c * self.kernel.feature_weights(self._freqs)
#         return nufft2(f, data.x, data.y).real

#     def prior_logpdf(self):
#         return norm.logpdf(x=self.coefficients)


class FourierGP(SpatialModel):
    n_modes: tuple[int, int]
    coefficients: Array
    kernel: Kernel
    _freqs: Array = field(static=True)

    def __init__(self, n_modes: tuple[int, int], kernel: Kernel):
        # Model specfication
        self.n_modes = n_modes
        fx, fy = get_freqs(n_modes, 2)
        self._freqs = jnp.sqrt(fx**2 + fy**2)
        self.kernel = kernel
        # Initialise parameters
        self.coefficients = jnp.zeros(n_modes)

    def __call__(self, data: SpatialData):
        op = JaxFinufft2DRealOperator(data.x, data.y, self.n_modes)
        scaled_coeffs = self.coefficients * self.kernel.feature_weights(self._freqs)
        return op @ scaled_coeffs.flatten()

    def prior_logpdf(self):
        return norm.logpdf(x=self.coefficients)


class PerSpaxel(SpatialModel):
    # Model parameters
    values: Array

    def __init__(self, n_spaxels: int):
        self.values = jnp.zeros(n_spaxels)

    def __call__(self, data: SpatialData):
        return self.values[data.indices]
