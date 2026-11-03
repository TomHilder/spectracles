"""domain.py - The region of the plane that a field's data cover, in the data's own units."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from spectracles.model.data import SpatialDataGeneric


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

    def grid(self, shape: tuple[int, int], padding: float = 0.0) -> SpatialDataGeneric:
        """A regular grid of points over the domain, e.g. to map a field after fitting.

        With `padding=0.0` (the default) the grid spans the data region, which
        is inside the periodic box of every GP on this domain. To span one GP's
        whole box, pass its padding, e.g. `domain.grid((200, 200), padding=gp.padding)`.

        Args:
            shape: `(n, m)`: n points along x and m along y, edges included.
            padding: How far past the domain to extend on each side, as a fraction
                of the extent.

        Returns:
            The points, flattened. Values evaluated on them reshape to `(n, m)`,
            as do the grid's own `x` and `y` (e.g. for `pcolormesh`):

                grid = domain.grid((200, 200))
                flux_map = field(grid).reshape(200, 200)
                x, y = grid.x.reshape(200, 200), grid.y.reshape(200, 200)
        """
        try:
            n, m = (int(v) for v in shape)
        except (TypeError, ValueError):
            raise TypeError(f"shape must be two integers (n, m), got {shape!r}.") from None
        if n < 1 or m < 1:
            raise ValueError(f"shape must be positive, got {shape!r}.")
        padding = float(padding)
        if not (np.isfinite(padding) and padding >= 0):
            raise ValueError(f"padding must be a non-negative number, got {padding!r}.")
        half = 0.5 * self.box_size(padding)
        x = jnp.linspace(self.centre[0] - half, self.centre[0] + half, n)
        y = jnp.linspace(self.centre[1] - half, self.centre[1] + half, m)
        gx, gy = jnp.meshgrid(x, y, indexing="ij")
        return SpatialDataGeneric(x=gx, y=gy, idx=jnp.arange(n * m))

    def box_size(self, padding: float) -> float:
        """The side of the periodic box: the extent plus `padding` × extent on each side."""
        return self.extent * (1 + 2 * padding)
