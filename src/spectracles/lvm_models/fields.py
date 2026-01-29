"""fields.py - Spatial model field components for LVM. Probably move to spectracles.model.spatial later."""

import jax.numpy as jnp
from jaxtyping import Array

from spectracles import FourierGP, Kernel, PerSpaxel, SpatialData, SpatialModel, l_bounded

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

    # Brain hurt. This is represented_field = ratio_field * base_field. In galaxy studies, usually the denominator of the ratio is the stronger signal, e.g. H-alpha. So in this way we can model the ratio field directly and multiply by the base (strong line) field to ge the the weaker line field.

    base_field: SpatialModel
    log10_ratio_field: SpatialModel

    def __init__(
        self,
        base_field: SpatialModel,
        log10_ratio_field: SpatialModel,
    ):
        self.base_field = base_field
        self.log10_ratio_field = log10_ratio_field

    def ratio(self, data: SpatialData) -> Array:
        return 10 ** (self.log10_ratio_field(data))

    def __call__(self, data: SpatialData) -> Array:
        return self.base_field(data) * self.ratio(data)


class FieldPlusScatter(SpatialModel):
    """A spatial field model represented as a base field model and a per-spaxel scatter term. Useful for having a mostly continuous component with some sparse scatter that is not well modelled by the continuous field."""

    base_field: SpatialModel
    scatter: PerSpaxel

    def __init__(
        self,
        base_field: SpatialModel,
        scatter: PerSpaxel,
    ):
        self.base_field = base_field
        self.scatter = scatter

    def __call__(self, data: SpatialData) -> Array:
        return self.base_field(data) + self.scatter(data)
