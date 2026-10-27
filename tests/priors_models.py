"""priors_models.py - Tiny models for the prior, latent-space and backend tests.

`make_line_model` covers every row of the semantics table in proposal §3: GP
coefficients (StdNormal default), kernel hyperparameters (LogNormal), a kernel
shared between two lines (one parameter used twice), a Known line centre, a
fixed width, a per-tile offset (Normal) and a per-spaxel scatter (Laplace, which
has no inverse transform).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array

from spectracles import (
    FourierGP,
    Known,
    Matern32,
    Parameter,
    PerSpaxel,
    SpatialDataGeneric,
    SpatialDataLVM,
    build_model,
)
from spectracles.model.spatial import PerTile
from spectracles.priors import Laplace, LogNormal, Normal

C_KMS = 299_792.458
N_MODES = (5, 5)
N_SPAXELS = 30
N_TILES = 2
LAM = jnp.linspace(6556.0, 6572.0, 12)


class GPModel(eqx.Module):
    gp: FourierGP

    def __call__(self, data):
        return self.gp(data)


def make_gp_model(fixed_hyperparameters: bool = False):
    """One FourierGP; all priors invertible (StdNormal coefficients, LogNormal kernel)."""
    kw = {"fixed": True} if fixed_hyperparameters else {"prior": LogNormal(0.6, 0.2)}
    kernel = Matern32(
        length_scale=Parameter(0.6, **kw),
        variance=Parameter(
            1.0, **({"fixed": True} if fixed_hyperparameters else {"prior": LogNormal(1.0, 0.5)})
        ),
    )
    return build_model(GPModel, FourierGP(n_modes=N_MODES, kernel=kernel))


def gp_data(n: int = 40, seed: int = 0) -> SpatialDataGeneric:
    xy = random.uniform(random.PRNGKey(seed), (2, n), minval=-jnp.pi, maxval=jnp.pi)
    return SpatialDataGeneric(x=xy[0], y=xy[1], idx=jnp.arange(n))


class Line(eqx.Module):
    μ: Known  # rest wavelength [Å]
    log_A: FourierGP  # log line flux
    v: FourierGP  # velocity [km/s / V_SCALE]
    σ: Parameter  # line width [Å], fixed

    V_SCALE = 20.0

    def __call__(self, λ: Array, s) -> Array:
        A = jnp.exp(self.log_A(s))
        μ_obs = self.μ.val * (1 + self.V_SCALE * self.v(s) / C_KMS)
        σ = self.σ.val
        return A / (σ * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * ((λ - μ_obs) / σ) ** 2)


class TwoLines(eqx.Module):
    line_1: Line
    line_2: Line
    offset: PerTile
    scatter: PerSpaxel

    def __call__(self, λ: Array, s) -> Array:
        return self.line_1(λ, s) + self.line_2(λ, s) + self.offset(s) + self.scatter(s)


def _line(μ: float, v_kernel: Matern32) -> Line:
    A_kernel = Matern32(
        length_scale=Parameter(0.8, prior=LogNormal(0.8, 0.3)),
        variance=Parameter(0.5, prior=LogNormal(0.5, 0.2)),
    )
    return Line(
        μ=Known(μ),
        log_A=FourierGP(n_modes=N_MODES, kernel=A_kernel),
        v=FourierGP(n_modes=N_MODES, kernel=v_kernel),
        σ=Parameter(1.2, fixed=True),
    )


def make_line_model(untie_velocity_kernel: bool = False):
    """Two lines sharing one velocity kernel (unless untied, for reference gradients)."""

    def v_kernel():
        return Matern32(
            length_scale=Parameter(1.0, prior=LogNormal(1.0, 0.3)),
            variance=Parameter(1.0, prior=LogNormal(1.0, 0.3)),
        )

    k1 = v_kernel()
    k2 = v_kernel() if untie_velocity_kernel else k1
    return build_model(
        TwoLines,
        line_1=_line(6562.8, k1),
        line_2=_line(6566.0, k2),
        offset=PerTile(N_TILES, tile_values=Parameter(dims=N_TILES, prior=Normal(0.0, 0.1))),
        scatter=PerSpaxel(N_SPAXELS, spaxel_values=Parameter(dims=N_SPAXELS, prior=Laplace(0.01))),
    )


def line_data(seed: int = 3) -> SpatialDataLVM:
    xy = random.uniform(random.PRNGKey(seed), (2, N_SPAXELS), minval=-2.5, maxval=2.5)
    return SpatialDataLVM(
        x=xy[0],
        y=xy[1],
        idx=jnp.arange(N_SPAXELS),
        tile_idx=jnp.arange(N_SPAXELS) % N_TILES,
        ifu_idx=jnp.zeros(N_SPAXELS, int),
        fib_idx=jnp.arange(N_SPAXELS),
    )


def line_forward(spatial_data):
    def forward(model):
        return jax.vmap(model, in_axes=(0, None))(LAM, spatial_data)  # (n_λ, n_spaxels)

    return forward


def noisy_data(pred: Array, noise_std: float, mask_frac: float, seed: int):
    """Data = pred + noise, with a random fraction of pixels masked and set to NaN."""
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred)
    data = pred + noise_std * rng.standard_normal(pred.shape)
    ivar = np.full(pred.shape, noise_std**-2.0)
    mask = rng.random(pred.shape) > mask_frac
    data[~mask] = np.nan
    ivar[~mask] = np.nan
    return data, ivar, mask
