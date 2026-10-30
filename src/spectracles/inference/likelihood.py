"""likelihood.py - Likelihoods shared by every inference backend.

Each likelihood has a pure-JAX log-likelihood (used for NUTS and MAP) and an
equivalent NIFTy.re likelihood, from `to_nifty` (used for MGVI/geoVI).
"""

import math
import operator
from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from jaxtyping import Array, ArrayLike

LOG_2PI = math.log(2.0 * math.pi)


class Likelihood(eqx.Module):
    """Base class for likelihoods: the observed data and how a prediction is scored.

    A subclass provides `n_used` (the number of data points used), `compress`
    (a prediction with the data's shape to the used data points, flat),
    `log_likelihood` (of a prediction, in pure JAX) and `to_nifty` (the
    equivalent NIFTy.re likelihood, taking the compressed prediction).
    """

    @property
    @abstractmethod
    def n_used(self) -> int:
        """Number of data points used."""

    @abstractmethod
    def compress(self, pred: Array) -> Array:
        """The used data points of a prediction with the data's shape, as a flat array."""

    @abstractmethod
    def log_likelihood(self, pred: Array) -> Array:
        """Log-likelihood of a prediction with the data's shape."""

    @abstractmethod
    def to_nifty(self) -> Any:
        """The equivalent NIFTy.re likelihood, taking the compressed prediction."""


class GaussianLikelihood(Likelihood):
    """Independent Gaussian noise with known inverse variance, and a mask.

    Masked data points are removed when the likelihood is built, so their values
    and inverse variances (often NaN) never enter a value or a gradient. Data
    points with a non-finite value or inverse variance, or an inverse variance
    <= 0, are masked too.

    Args:
        data: Observed data, any shape.
        ivar: Inverse variance, same shape as `data`.
        mask: Boolean, True where a data point is used. Defaults to all data points.
    """

    data: Array  # (n_used,) the used data points only
    std_inv: Array  # (n_used,)
    index: Array  # (n_used,) flat indices of the used data points
    log_norm: Array  # the constant part of the log-likelihood
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
        if index.size == 0:
            raise ValueError(
                "No data points left: every one is masked, non-finite, or has inverse variance <= 0."
            )
        self.shape = data.shape
        self.index = jnp.asarray(index)
        self.data = jnp.asarray(data.ravel()[index])
        self.std_inv = jnp.sqrt(jnp.asarray(ivar.ravel()[index]))
        self.log_norm = jnp.sum(jnp.log(self.std_inv)) - 0.5 * index.size * LOG_2PI

    @property
    def n_used(self) -> int:
        """Number of data points used."""
        return int(self.index.shape[0])

    def compress(self, pred: Array) -> Array:
        """The used data points of a prediction with the data's shape, as a flat array."""
        if pred.shape != self.shape:
            raise ValueError(f"prediction {pred.shape} must have the data's shape {self.shape}.")
        return pred.reshape(-1)[self.index]

    def log_likelihood(self, pred: Array) -> Array:
        """Gaussian log-likelihood of a prediction, with full normalisation."""
        r = self.normalised_residual(pred)
        return -0.5 * jnp.sum(r**2) + self.log_norm

    def normalised_residual(self, pred: Array) -> Array:
        """(data - pred) / σ on the used data points."""
        return (self.data - self.compress(pred)) * self.std_inv

    def to_nifty(self):
        """The equivalent NIFTy.re likelihood: `jft.Gaussian` on the used data points.

        Its input is the compressed prediction (see `compress`).
        """
        from spectracles.nifty import _import_nifty

        jft = _import_nifty()
        return jft.Gaussian(
            self.data,
            noise_cov_inv=Partial(operator.mul, self.std_inv**2),
            noise_std_inv=Partial(operator.mul, self.std_inv),
        )
