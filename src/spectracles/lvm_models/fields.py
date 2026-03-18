"""fields.py - Spatial model field components for LVM. Probably move to spectracles.model.spatial later."""

import jax.numpy as jnp
from jaxtyping import Array

from spectracles import (
    AnyParameter,
    FourierGP,
    Kernel,
    PerSpaxel,
    SpatialData,
    SpatialModel,
    l_bounded,
)

DEFAULT_N_MODES = (101, 101)


class GPField(SpatialModel):
    """A spatial field model represented by a Gaussian Process."""

    gp: FourierGP

    def __init__(
        self,
        kernel: Kernel,
        n_modes: tuple[int, int] = DEFAULT_N_MODES,
        coefficients: Array | None = None,
    ):
        self.gp = FourierGP(n_modes=n_modes, kernel=kernel, coefficients=coefficients)

    def __call__(self, data: SpatialData) -> Array:
        return self.gp(data)


class PositiveGPField(SpatialModel):
    """A spatial field model represented by an underlying Gaussian Process constrained to be positive with a zero or tiny lower bound. Uses the softplus transformation to ensure positivity."""

    gp: FourierGP
    lower: float

    def __init__(
        self,
        kernel: Kernel,
        n_modes: tuple[int, int] = DEFAULT_N_MODES,
        coefficients: Array | None = None,
        lower: float = 0.0,
    ):
        if lower < 0.0:
            raise ValueError("Lower bound for PositiveGPField must be non-negative.")

        self.gp = FourierGP(n_modes=n_modes, kernel=kernel, coefficients=coefficients)
        self.lower = lower

    def __call__(self, data: SpatialData) -> Array:
        return l_bounded(self.gp(data), lower=self.lower)


class LogGPField(SpatialModel):
    """A spatial field model represented by a Log-Gaussian Process, i.e., the exponentiation of an underlying Gaussian Process."""

    gp: FourierGP

    def __init__(
        self,
        kernel: Kernel,
        n_modes: tuple[int, int] = DEFAULT_N_MODES,
        coefficients: Array | None = None,
    ):
        self.gp = FourierGP(n_modes=n_modes, kernel=kernel, coefficients=coefficients)

    def __call__(self, data: SpatialData) -> Array:
        return jnp.exp(self.gp(data))


class FieldFromRatio(SpatialModel):
    """A spatial field model represented as the product of a base field and a ratio field. The ratio field is represented by a positively constrained field via the log10 to ensure positivity. log10 is used instead of natural log here as it's conventional to work with log10(ratio) in astronomy."""

    base_field: SpatialModel
    log10_ratio_field: SpatialModel
    log10_ratio_mean: AnyParameter

    def ratio(self, data: SpatialData) -> Array:
        return 10 ** (self.log10_ratio_field(data) + self.log10_ratio_mean.val)

    def log10_ratio(self, data: SpatialData) -> Array:
        return self.log10_ratio_field(data) + self.log10_ratio_mean.val

    def __call__(self, data: SpatialData) -> Array:
        return self.base_field(data) * self.ratio(data)


class FieldPlusScatter(SpatialModel):
    """A spatial field model represented as a base field model and a per-spaxel scatter term. Useful for having a mostly continuous component with some sparse scatter that is not well modelled by the continuous field."""

    base_field: SpatialModel
    scatter: PerSpaxel

    def __call__(self, data: SpatialData) -> Array:
        return self.base_field(data) + self.scatter(data)


class SumTwoFields(SpatialModel):
    """A spatial field model that has two sub-fields, and returns their sum everywhere."""

    field_1: SpatialModel
    field_2: SpatialModel

    def __call__(self, data: SpatialData) -> Array:
        return self.field_1(data) + self.field_2(data)


# NOTE/TODO: The two above are basically the same. We should think about this more carefully to avoid redundant code but also not get mega confusing.
# I think having a Field base class could be nice, then we could just define arithmetic operators over fields
# Needs a thinko
