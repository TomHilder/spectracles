"""domain.py - The region of the plane that a field's data cover, in the data's own units."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import ArrayLike


@dataclass(frozen=True)
class Domain:
    """The square region the data cover, in the data's own units (e.g. arcsec).

    A `FourierGP` given a domain works in those units: its length scale is in
    the same units as the coordinates, and it represents the field on a
    periodic box made of the domain plus a margin of padding on each side.
    Several GPs can share one domain and still choose their own padding.

    Args:
        centre: The centre (x, y) of the region.
        extent: The side of the square region.

    Attributes:
        centre: The centre, as a tuple of floats.
        extent: The side of the square, as a float.
    """

    centre: tuple[float, float]
    extent: float

    def __post_init__(self):
        try:
            centre = tuple(float(c) for c in self.centre)
        except TypeError:
            raise TypeError(f"centre must be two numbers (x, y), got {self.centre!r}.") from None
        if len(centre) != 2 or not all(np.isfinite(centre)):
            raise ValueError(f"centre must be two finite numbers (x, y), got {self.centre!r}.")
        extent = float(self.extent)
        if not (np.isfinite(extent) and extent > 0):
            raise ValueError(f"extent must be a positive number, got {self.extent!r}.")
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "extent", extent)

    @classmethod
    def from_points(cls, x: ArrayLike, y: ArrayLike) -> "Domain":
        """The smallest square containing the points: centred on their bounding box,
        with the longer side of the box as its extent."""
        x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
        if x.size == 0 or x.shape != y.shape:
            raise ValueError(
                f"x and y must be non-empty and the same length, got {x.size} and {y.size}."
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("x and y must be finite.")
        x_lo, x_hi, y_lo, y_hi = x.min(), x.max(), y.min(), y.max()
        extent = max(x_hi - x_lo, y_hi - y_lo)
        if extent == 0:
            raise ValueError("All the points are at one position, so they don't span a region.")
        return cls(centre=((x_lo + x_hi) / 2, (y_lo + y_hi) / 2), extent=extent)

    def box_size(self, padding: float) -> float:
        """The side of the periodic box: the extent plus `padding` × extent on each side."""
        return self.extent * (1 + 2 * padding)
