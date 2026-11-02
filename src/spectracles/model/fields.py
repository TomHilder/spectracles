"""fields.py - Fields built from other fields: one line from another and a ratio, and a field plus per-spaxel scatter."""

from jaxtyping import Array

from spectracles.model.data import SpatialData
from spectracles.model.spatial import PerSpaxel, SpatialModel


class FieldFromRatio(SpatialModel):
    """A field given as a base field times a ratio: `base(x) × 10^log10_ratio(x)`.

    Useful when a line's ratio to another line is the quantity of interest,
    e.g. [N II]/Hα. The ratio is modelled through its base-10 log, so it is
    positive; give `log10_ratio` a mean to set the typical ratio:

        halpha = FourierGP(k1, domain=d, mean=Parameter(prior=Normal(0, 1)), link="exp")
        log10_ratio = FourierGP(k2, domain=d, mean=Parameter(prior=Normal(-0.5, 0.5)))
        nii = FieldFromRatio(base=halpha, log10_ratio=log10_ratio)

    Args:
        base: The field the ratio is taken relative to.
        log10_ratio: The base-10 log of the ratio, as a field.
    """

    base: SpatialModel
    log10_ratio: SpatialModel

    def __init__(self, base: SpatialModel, log10_ratio: SpatialModel):
        for name, value in (("base", base), ("log10_ratio", log10_ratio)):
            if not isinstance(value, SpatialModel):
                raise TypeError(
                    f"{name} must be a spatial model (e.g. a FourierGP), got {type(value).__name__}."
                )
        self.base = base
        self.log10_ratio = log10_ratio

    def ratio(self, data: SpatialData) -> Array:
        """The ratio, `10^log10_ratio(x)`."""
        return 10 ** self.log10_ratio(data)

    def __call__(self, data: SpatialData) -> Array:
        return self.base(data) * self.ratio(data)


class FieldPlusScatter(SpatialModel):
    """A field plus a separate value for each spaxel: `field(x) + scatter[spaxel]`.

    For a quantity that is mostly continuous across the sky, with some
    spaxel-to-spaxel scatter the field can't follow.

    Args:
        field: The continuous part.
        scatter: The per-spaxel part, e.g.
            `PerSpaxel(n, Parameter(dims=n, prior=Normal(0.0, 0.05)))`.
    """

    field: SpatialModel
    scatter: PerSpaxel

    def __init__(self, field: SpatialModel, scatter: PerSpaxel):
        if not isinstance(field, SpatialModel):
            raise TypeError(f"field must be a spatial model, got {type(field).__name__}.")
        if not isinstance(scatter, PerSpaxel):
            raise TypeError(f"scatter must be a PerSpaxel, got {type(scatter).__name__}.")
        self.field = field
        self.scatter = scatter

    def __call__(self, data: SpatialData) -> Array:
        return self.field(data) + self.scatter(data)
