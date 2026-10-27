"""likelihood.py - Likelihoods shared by every inference backend.

Each likelihood has a pure-JAX log-likelihood (used for NUTS and MAP) and maps to
its NIFTy.re twin (used for MGVI/geoVI); tests check that the two agree.
"""

import math
import operator

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from jaxtyping import Array, ArrayLike

LOG_2PI = math.log(2.0 * math.pi)


class GaussianLikelihood(eqx.Module):
    """Independent Gaussian noise with known inverse variance, and a mask.

    Masked pixels are removed when the likelihood is built, so their data and
    inverse variance (often NaN) never enter a value or a gradient. Pixels with
    non-finite data or inverse variance, or inverse variance <= 0, are masked too.

    Args:
        data: Observed data, any shape.
        ivar: Inverse variance, same shape as `data`.
        mask: Boolean, True where a pixel is used. Defaults to all pixels.
    """

    data: Array  # (n_used,) the used pixels only
    std_inv: Array  # (n_used,)
    index: Array  # (n_used,) flat indices of the used pixels
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, data: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None):
        data = np.asarray(data)
        ivar = np.asarray(ivar)
        if data.shape != ivar.shape:
            raise ValueError(f"data {data.shape} and ivar {ivar.shape} must have the same shape.")
        use = np.ones(data.shape, bool) if mask is None else np.asarray(mask, bool)
        if use.shape != data.shape:
            raise ValueError(f"mask {use.shape} must have the same shape as data {data.shape}.")
        with np.errstate(invalid="ignore"):
            use = use & np.isfinite(data) & np.isfinite(ivar) & (ivar > 0)
        index = np.flatnonzero(use)
        self.shape = data.shape
        self.index = jnp.asarray(index)
        self.data = jnp.asarray(data.ravel()[index])
        self.std_inv = jnp.sqrt(jnp.asarray(ivar.ravel()[index]))

    @property
    def n_used(self) -> int:
        """Number of pixels used."""
        return int(self.index.shape[0])

    def compress(self, pred: Array) -> Array:
        """The used pixels of a prediction with the data's shape, as a flat array."""
        if pred.shape != self.shape:
            raise ValueError(f"prediction {pred.shape} must have the data's shape {self.shape}.")
        return pred.reshape(-1)[self.index]

    def log_likelihood(self, pred: Array) -> Array:
        """Gaussian log-likelihood of a prediction, with full normalisation."""
        r = (self.data - self.compress(pred)) * self.std_inv
        return -0.5 * jnp.sum(r**2) + jnp.sum(jnp.log(self.std_inv)) - 0.5 * r.size * LOG_2PI

    def normalised_residual(self, pred: Array) -> Array:
        """(data - pred) / σ on the used pixels."""
        return (self.data - self.compress(pred)) * self.std_inv

    def to_nifty(self):
        """The NIFTy.re twin: `jft.Gaussian` on the used pixels.

        Its input is the compressed prediction (see `compress`).
        """
        from spectracles.nifty import _import_nifty

        jft = _import_nifty()
        return jft.Gaussian(
            self.data,
            noise_cov_inv=Partial(operator.mul, self.std_inv**2),
            noise_std_inv=Partial(operator.mul, self.std_inv),
        )
