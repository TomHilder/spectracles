"""spatial.py - Spatial models, especially flexible models built on Fourier-accelerated Gaussian Processes."""

from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module
from jax.scipy.stats import norm
from jax_finufft import nufft2  # type: ignore
from jaxtyping import Array

from spectracles.model.data import SpatialData
from spectracles.model.kernels import Kernel
from spectracles.model.parameter import Parameter, init_parameter

# NOTE: List of current obvious foot guns:
# - n_modes must always be two odd integers, but this is not enforced
#   - in theory the nifty-solve full implementation should be able to handle
#     this, but from memory it doesn't work right now. This is why p is
#     repeated in the shape info right now, since I didn't implement n_modes
#     != n_requested_modes yet.

FINUFFT_KWDS = dict(eps=1e-10)


def get_freqs_1D(n_modes: int) -> Array:
    if n_modes % 2 == 0:
        return jnp.arange(-n_modes // 2, n_modes // 2, dtype=float)
    else:
        return jnp.arange(-(n_modes - 1) // 2, (n_modes - 1) // 2 + 1, dtype=float)


def get_freqs(n_modes: int | tuple[int, ...], n_dim: int = 2) -> Array | list[Array]:
    if n_dim == 1 and isinstance(n_modes, int):
        return get_freqs_1D(n_modes)
    elif isinstance(n_modes, tuple):
        if len(n_modes) != n_dim:
            raise ValueError(
                f"n_modes must be a tuple of length {n_dim} for {n_dim}D data, but got {len(n_modes)}",
            )
        modes_grid = jnp.meshgrid(*[get_freqs_1D(n_modes[i]) for i in range(n_dim)], indexing="ij")
        # Transpose because fiNUFFT treats the first dimension as the fastest changing
        return modes_grid
    else:
        raise ValueError(
            f"n_modes must be an int or a tuple of ints, but got {type(n_modes)}",
        )


class SpatialModel(Module):
    @abstractmethod
    def __call__(self, data: SpatialData):
        pass


class FourierGP(SpatialModel):
    coefficients: Parameter
    kernel: Kernel
    n_modes: tuple[int, int]
    _freqs: Array
    _shape_info: tuple[int, int, int]

    def __init__(
        self,
        n_modes: tuple[int, int],
        kernel: Kernel,
        coefficients: Parameter | None = None,
    ):
        # Model specfication
        self.n_modes = n_modes
        fx, fy = get_freqs(n_modes)
        self._freqs = jnp.sqrt(fx**2 + fy**2)
        self.kernel = kernel
        # Initialise parameters
        self.coefficients = init_parameter(coefficients, dims=n_modes)
        # Initialise the shape info
        p = int(jnp.prod(jnp.array(n_modes)))
        self._shape_info = (p, p // 2, p)

    def __call__(self, data: SpatialData) -> Array:
        # Feature weighted coefficients
        scaled_coeffs = self.coefficients.val * self.kernel.feature_weights(self._freqs)
        # Sum basis functions with nufft after processing the coefficients to enforce conjugate symmetry
        model_eval: Array = nufft2(
            self._conj_symmetry(scaled_coeffs.flatten()),
            data.x,
            data.y,
            **FINUFFT_KWDS,
        ).real
        return model_eval

    def _conj_symmetry(self, c: Array) -> Array:
        m, h, p = self._shape_info
        f = 0.5 * jnp.hstack(
            [c[: h + 1], jnp.zeros(p - h - 1)],
        ) + 0.5j * jnp.hstack(
            [jnp.zeros(p - m + h + 1), c[h + 1 :]],
        )
        f = f.reshape(self.n_modes)
        return f + jnp.conj(jnp.flip(f))

    def prior_logpdf(self) -> Array:
        fw = self.kernel.feature_weights(self._freqs)
        jacobian = -0.5 * jnp.log(fw).sum()
        return norm.logpdf(x=self.coefficients.val).sum() + jacobian


class PerSpaxel(SpatialModel):
    # Model parameters
    spaxel_values: Parameter

    def __init__(self, n_spaxels: int, spaxel_values: Parameter | None = None):
        self.spaxel_values = init_parameter(spaxel_values, dims=n_spaxels)

    def __call__(self, data: SpatialData) -> Array:
        return self.spaxel_values.val[data.idx]


class PerIFUAndTile(SpatialModel):
    # Model parameters
    ifu_values: Parameter

    def __init__(self, n_tiles: int, n_ifus: int, ifu_values: Parameter | None = None):
        self.ifu_values = init_parameter(ifu_values, dims=(n_tiles, n_ifus))

    def __call__(self, data: SpatialData) -> Array:
        return self.ifu_values.val[data.tile_idx, data.ifu_idx]


class PerIFU(SpatialModel):
    # Model parameters
    ifu_values: Parameter

    def __init__(self, n_ifus: int, ifu_values: Parameter | None = None):
        self.ifu_values = init_parameter(ifu_values, dims=n_ifus)

    def __call__(self, data: SpatialData) -> Array:
        return self.ifu_values.val[data.ifu_idx]


class PerTile(SpatialModel):
    # Model parameters
    tile_values: Parameter

    def __init__(self, n_tiles: int, tile_values: Parameter | None = None):
        self.tile_values = init_parameter(tile_values, dims=n_tiles)

    def __call__(self, data: SpatialData) -> Array:
        return self.tile_values.val[data.tile_idx]
