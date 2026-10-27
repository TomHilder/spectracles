"""priors.py - Priors on parameters, given as transforms of standard-normal latents.

Every prior maps a standard-normal latent ξ ~ N(0, 1) to the parameter's value
through a deterministic transform. Inference then happens in ξ, where the prior
density is always -½|ξ|², so no prior log-density is ever needed. The transforms
follow NIFTy.re's conventions exactly (see `nifty.re.num.stats_distributions`),
but nothing here imports NIFTy.

Hyperparameters are scalars, broadcast to the parameter's shape. Priors are
frozen dataclasses so they are hashable and can live in static fields.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from jaxtyping import Array, PRNGKeyArray

LOG2 = math.log(2.0)


def _scalar(cls_name: str, name: str, value) -> float:
    """Validate that a hyperparameter is a real scalar and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        # Allow 0-d numpy/jax arrays, but nothing with a shape
        shape = getattr(value, "shape", None)
        if shape == ():
            return float(value)
        raise TypeError(
            f"{cls_name}: hyperparameter '{name}' must be a scalar, got {type(value).__name__}"
            f"{f' with shape {shape}' if shape is not None else ''}. Per-element "
            f"hyperparameters are not supported; use a StdNormal prior and compute "
            f"mean + std * value from Known arrays in the model instead."
        )
    return float(value)


def _fmt(x: float) -> str:
    return f"{x:g}"


class Prior(ABC):
    """A prior on a parameter, given as a transform of a standard normal."""

    has_inverse: ClassVar[bool] = False

    @abstractmethod
    def transform(self, ξ: Array) -> Array:
        """Map standard-normal latents ξ to parameter values."""

    def inverse(self, x: Array) -> Array:
        """Map parameter values back to latents ξ (only if `has_inverse`)."""
        raise NotImplementedError(f"{type(self).__name__} has no inverse transform.")

    def sample(self, key: PRNGKeyArray, shape: tuple[int, ...] = ()) -> Array:
        """Draw values from the prior."""
        return self.transform(random.normal(key, shape))


@dataclass(frozen=True, repr=False)
class StdNormal(Prior):
    """Standard normal, N(0, 1): the value is the latent itself."""

    has_inverse: ClassVar[bool] = True

    def transform(self, ξ: Array) -> Array:
        return ξ

    def inverse(self, x: Array) -> Array:
        return x

    def __repr__(self) -> str:
        return "StdNormal"


@dataclass(frozen=True, repr=False)
class Normal(Prior):
    """Normal with mean `mean` and standard deviation `std`."""

    mean: float
    std: float
    has_inverse: ClassVar[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _scalar("Normal", "mean", self.mean))
        object.__setattr__(self, "std", _scalar("Normal", "std", self.std))
        if self.std <= 0:
            raise ValueError(f"Normal: std must be positive, got {self.std}.")

    def transform(self, ξ: Array) -> Array:
        return self.mean + self.std * ξ

    def inverse(self, x: Array) -> Array:
        return (x - self.mean) / self.std

    def __repr__(self) -> str:
        return f"Normal({_fmt(self.mean)}, {_fmt(self.std)})"


@dataclass(frozen=True, repr=False)
class LogNormal(Prior):
    """Log-normal with mean `mean` and standard deviation `std` of the value itself.

    Moment-matched as in NIFTy.re: value = exp(m + s ξ), with
    s² = log(1 + (std/mean)²) and m = log(mean) - s²/2.
    """

    mean: float
    std: float
    has_inverse: ClassVar[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _scalar("LogNormal", "mean", self.mean))
        object.__setattr__(self, "std", _scalar("LogNormal", "std", self.std))
        if self.mean <= 0:
            raise ValueError(f"LogNormal: mean must be positive, got {self.mean}.")
        if self.std <= 0:
            raise ValueError(f"LogNormal: std must be positive, got {self.std}.")

    @property
    def log_moments(self) -> tuple[float, float]:
        """The mean and standard deviation (m, s) of log(value)."""
        s = math.sqrt(math.log1p((self.std / self.mean) ** 2))
        m = math.log(self.mean) - 0.5 * s**2
        return m, s

    def transform(self, ξ: Array) -> Array:
        m, s = self.log_moments
        return jnp.exp(m + s * ξ)

    def inverse(self, x: Array) -> Array:
        m, s = self.log_moments
        return (jnp.log(x) - m) / s

    def __repr__(self) -> str:
        return f"LogNormal({_fmt(self.mean)}, {_fmt(self.std)})"


@dataclass(frozen=True, repr=False)
class Laplace(Prior):
    """Laplace with location 0 and scale `alpha`: p(x) = exp(-|x|/alpha) / (2 alpha).

    Same construction as NIFTy.re's `LaplacePrior` (the Laplace quantile function
    applied to Φ(ξ), written with `norm.logcdf` for stability in the tails).
    """

    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _scalar("Laplace", "alpha", self.alpha))
        if self.alpha <= 0:
            raise ValueError(f"Laplace: alpha must be positive, got {self.alpha}.")

    def transform(self, ξ: Array) -> Array:
        # Both branches are finite everywhere, so jnp.where is gradient-safe here
        neg = norm.logcdf(ξ) + LOG2
        pos = -(norm.logcdf(-ξ) + LOG2)
        return self.alpha * jnp.where(ξ <= 0, neg, pos)

    def __repr__(self) -> str:
        return f"Laplace({_fmt(self.alpha)})"


@dataclass(frozen=True, repr=False)
class Uniform(Prior):
    """Uniform on [low, high]: value = low + (high - low) Φ(ξ)."""

    low: float
    high: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "low", _scalar("Uniform", "low", self.low))
        object.__setattr__(self, "high", _scalar("Uniform", "high", self.high))
        if self.low >= self.high:
            raise ValueError(
                f"Uniform: low must be below high, got low={self.low}, high={self.high}."
            )

    def transform(self, ξ: Array) -> Array:
        return self.low + (self.high - self.low) * norm.cdf(ξ)

    def __repr__(self) -> str:
        return f"Uniform({_fmt(self.low)}, {_fmt(self.high)})"
