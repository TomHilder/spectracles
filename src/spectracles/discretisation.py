"""discretisation.py - How accurately a FourierGP represents its kernel, and the modes and padding a target accuracy needs.

A FourierGP represents a stationary GP by Fourier modes on a periodic box, so
its covariance differs from the kernel's in two ways:

- **truncation**: the mode grid stops at a highest frequency, so structure
  below the grid spacing is missing (too few modes for a short length scale);
- **aliasing**: the box is periodic, so each point is correlated with the
  periodic images of the others (too little padding for a long length scale).

The field's covariance follows from the kernel's weights alone, so both
errors are computed exactly, without data. Errors are relative to the
field's variance s².

The truncation error is a relative Frobenius error: the root-mean-square
difference between the field's covariance and the reference kernel, over
pairs of points spread uniformly over the data region, divided by the kernel's
own root-mean-square there. The reference is the continuous kernel scaled to
variance s² / (1 - τ), where τ is the fraction of the kernel's variance at
scales below the grid: the field keeps s² as its pointwise variance, so this
is the continuous GP it approximates. The aliasing error is the worst case
over pairs of points in the data region.
"""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from scipy import fft

from spectracles.model.domain import Domain
from spectracles.model.kernels import Kernel
from spectracles.model.spatial import DEFAULT_PADDING, FourierGP, _check_n_modes

DEFAULT_TOL = 1e-3
MAX_MODES = 2047  # largest number of modes per side `required_modes` will try
MAX_IMAGE_RINGS = 12  # most rings of periodic images summed (more only for ℓ ≳ the box)


@dataclass(frozen=True)
class DiscretisationError:
    """A FourierGP's errors against its kernel, relative to the field's variance.

    Attributes:
        truncation: Relative Frobenius error from the finite mode grid.
        aliasing: Worst-case error from the box's periodic images.
        truncation_max: Worst-case truncation error (≈ tau).
        tau: Fraction of the kernel's variance at scales below the grid. It turns
            negative when aliasing is large, since the grid then also picks up
            the periodic images' variance.
        length_scale: The length scale the errors are for (None if the kernel has none).
        n_modes: Modes in each direction.
        padding: Padding on each side, as a fraction of the domain's extent.

    With an array of length scales, the error fields are arrays of the same
    shape.
    """

    truncation: float | np.ndarray
    aliasing: float | np.ndarray
    truncation_max: float | np.ndarray
    tau: float | np.ndarray
    length_scale: float | np.ndarray | None
    n_modes: tuple[int, int]
    padding: float

    @property
    def worst(self) -> float | np.ndarray:
        """The larger of the truncation and aliasing errors."""
        return np.maximum(self.truncation, self.aliasing)

    def __repr__(self) -> str:
        def fmt(v):
            v = np.asarray(v)
            return f"{float(v):.2e}" if v.ndim == 0 else f"array of {v.size}"

        ls = "" if self.length_scale is None else f"length_scale={fmt(self.length_scale)}, "
        return (
            f"DiscretisationError({ls}truncation={fmt(self.truncation)}, "
            f"aliasing={fmt(self.aliasing)}, truncation_max={fmt(self.truncation_max)}, "
            f"tau={fmt(self.tau)}; n_modes={self.n_modes}, padding={self.padding:g})"
        )


@dataclass(frozen=True)
class _Geometry:
    """The periodic box (side `size`) and the square data region (side `extent`) inside it."""

    extent: float
    padding: float

    @property
    def size(self) -> float:
        return self.extent * (1 + 2 * self.padding)


def _geometry(domain: Domain | None, padding: float | None) -> _Geometry:
    if domain is None:
        # No domain: coordinates fill the [-π, π] box, with no padding
        if padding not in (None, 0, 0.0):
            raise ValueError("padding needs a domain.")
        return _Geometry(extent=2 * np.pi, padding=0.0)
    if not isinstance(domain, Domain):
        raise TypeError(f"domain must be a Domain, got {type(domain).__name__}.")
    padding = DEFAULT_PADDING if padding is None else float(padding)
    if not (np.isfinite(padding) and padding >= 0):
        raise ValueError(f"padding must be a non-negative number, got {padding!r}.")
    return _Geometry(extent=domain.extent, padding=padding)


def _with_length_scale(kernel: Kernel, length_scale: float) -> Kernel:
    if not hasattr(kernel, "length_scale"):
        raise ValueError(
            f"{type(kernel).__name__} has no length_scale, so length_scale= can't be set."
        )
    current = kernel.length_scale.val  # type: ignore[attr-defined]
    return eqx.tree_at(
        lambda k: k.length_scale.val, kernel, jnp.full_like(current, float(length_scale))
    )


def _length_scale(kernel: Kernel) -> float | None:
    if not hasattr(kernel, "length_scale"):
        return None
    return float(jnp.squeeze(kernel.length_scale.val))  # type: ignore[attr-defined]


def _mode_grid(n_modes: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    jx = np.arange(-(n_modes[0] - 1) // 2, (n_modes[0] - 1) // 2 + 1)
    jy = np.arange(-(n_modes[1] - 1) // 2, (n_modes[1] - 1) // 2 + 1)
    return jx, jy


def _weights(kernel: Kernel, n_modes: tuple[int, int], size: float):
    """The field's mode weights w_j and the kernel's trapezoid weights c_j (absolute units)."""
    jx, jy = _mode_grid(n_modes)
    dk = 2 * np.pi / size
    k = dk * np.hypot(jx[:, None], jy[None, :])
    w = np.asarray(kernel.feature_weights(jnp.asarray(k)), dtype=float) ** 2
    c = np.asarray(kernel.spectral_density(jnp.asarray(k)), dtype=float) * dk**2
    return w, c


def _check_continuous_form(kernel: Kernel) -> None:
    try:
        kernel.spectral_density(1.0)
        kernel.covariance(0.0)
    except NotImplementedError:
        raise NotImplementedError(
            f"The diagnostic needs the kernel's continuous form: {type(kernel).__name__} "
            "doesn't define spectral_density and covariance."
        ) from None


def _variance(kernel: Kernel) -> float:
    return float(jnp.squeeze(kernel.covariance(0.0)))


class _LagGrid:
    """Lags z in [0, extent]² on an FFT grid (n points per box side), with quadrature
    weights p(z) dz for iid uniform points in the data square (folded by symmetry)."""

    def __init__(self, geometry: _Geometry, n_modes: tuple[int, int]):
        self.n = int(fft.next_fast_len(4 * max(n_modes)) or 4 * max(n_modes))
        self.p_max = min(int(geometry.extent * self.n / geometry.size), self.n - 1)
        dz = geometry.size / self.n
        z = dz * np.arange(self.p_max + 1)
        fold = np.where(np.arange(z.size) == 0, 1.0, 2.0)
        q = fold * np.clip(geometry.extent - z, 0, None) / geometry.extent**2 * dz
        self.zx, self.zy = np.meshgrid(z, z, indexing="ij")
        self.q = q[:, None] * q[None, :]


def _on_lag_grid(w: np.ndarray, grid: _LagGrid) -> np.ndarray:
    """Σ_j w_j cos(k_j · z) on the lag grid, by one FFT."""
    n = grid.n
    hx, hy = (w.shape[0] - 1) // 2, (w.shape[1] - 1) // 2
    a = np.zeros((n, n))
    ix, iy = np.r_[0 : hx + 1, n - hx : n], np.r_[0 : hy + 1, n - hy : n]
    a[np.ix_(ix, iy)] = np.fft.ifftshift(w)
    # The weights are even in each axis, so the sum is too: rfft2 keeps columns
    # 0..n/2, and a column p > n/2 equals n - p
    p = np.arange(grid.p_max + 1)
    p = np.where(p <= n // 2, p, n - p)
    return np.real(np.asarray(fft.rfft2(a, workers=-1)))[np.ix_(p, p)]


def _image_rings(kernel: Kernel, geometry: _Geometry) -> int:
    """Rings of periodic images to sum: until the next ring's nearest image is below
    1e-16 of the variance, for every lag in the data region."""
    s2 = _variance(kernel)
    for rings in range(1, MAX_IMAGE_RINGS + 1):
        nearest_beyond = (rings + 1) * geometry.size - np.sqrt(2) * geometry.extent
        if float(jnp.squeeze(kernel.covariance(nearest_beyond))) < 1e-16 * s2:
            return rings
    return MAX_IMAGE_RINGS


def _periodised(kernel: Kernel, zx, zy, geometry: _Geometry, include_centre: bool) -> np.ndarray:
    size, rings = geometry.size, _image_rings(kernel, geometry)
    out = np.zeros(np.broadcast(zx, zy).shape)
    for a in range(-rings, rings + 1):
        for b in range(-rings, rings + 1):
            if a == 0 and b == 0 and not include_centre:
                continue
            r = np.hypot(zx + a * size, zy + b * size)
            out += np.asarray(kernel.covariance(jnp.asarray(r)), dtype=float)
    return out


def _aliasing(kernel: Kernel, geometry: _Geometry) -> float:
    """Worst |Σ_{n≠0} k(z + n · size)| / s² over lags z in [0, extent]²."""
    z = np.linspace(0.0, geometry.extent, 65)
    zx, zy = np.meshgrid(z, z, indexing="ij")
    images = _periodised(kernel, zx, zy, geometry, include_centre=False)
    return float(np.abs(images).max() / _variance(kernel))


def _truncation(kernel: Kernel, n_modes: tuple[int, int], geometry: _Geometry):
    """(relative Frobenius error, worst-case error / s², tau) from the finite mode grid."""
    s2 = _variance(kernel)
    w, c = _weights(kernel, n_modes, geometry.size)
    tau = 1.0 - c.sum() / s2
    grid = _LagGrid(geometry, n_modes)
    realised = s2 * _on_lag_grid(w, grid) / w.sum()
    # The continuous kernel scaled to variance s² / (1 - tau), periodised so the
    # images (the aliasing error) don't count as truncation
    reference = _periodised(kernel, grid.zx, grid.zy, geometry, include_centre=True)
    reference *= s2 / c.sum()
    err = realised - reference
    frob = np.sqrt(np.sum(grid.q * err**2) / np.sum(grid.q * reference**2))
    return float(frob), float(np.abs(err).max() / s2), float(tau)


def _resolve(gp_or_kernel, domain, n_modes, padding):
    """The kernel, mode grid and geometry: from a FourierGP, with any overrides."""
    if isinstance(gp_or_kernel, FourierGP):
        gp = gp_or_kernel
        kernel = gp.kernel
        domain = gp.domain if domain is None else domain
        n_modes = gp.n_modes if n_modes is None else n_modes
        if padding is None and domain is gp.domain:
            padding = gp.padding
    elif isinstance(gp_or_kernel, Kernel):
        kernel = gp_or_kernel
        if n_modes is None:
            raise ValueError("n_modes is needed when a kernel (not a FourierGP) is given.")
    else:
        raise TypeError(
            f"Expected a FourierGP or a Kernel, got {type(gp_or_kernel).__name__}. For a part of "
            "a built model, pass model.for_evaluation().<path>."
        )
    _check_continuous_form(kernel)
    if isinstance(n_modes, (int, np.integer)):
        n_modes = (int(n_modes), int(n_modes))
    return kernel, _check_n_modes(n_modes), _geometry(domain, padding)


def discretisation_error(
    gp_or_kernel: FourierGP | Kernel,
    *,
    domain: Domain | None = None,
    n_modes: int | tuple[int, int] | None = None,
    padding: float | None = None,
    length_scale: float | ArrayLike | None = None,
) -> DiscretisationError:
    """The truncation and aliasing errors of a FourierGP, or of a kernel on a given grid.

    Every setting can be overridden, to ask what the error would be with other
    modes, padding or length scale:

        discretisation_error(gp)                        # as it is
        discretisation_error(gp, length_scale=40.0)     # if ℓ were 40
        discretisation_error(gp, n_modes=201)           # with 201 x 201 modes
        discretisation_error(kernel, domain=d, n_modes=301, padding=0.15)

    Args:
        gp_or_kernel: A FourierGP, or a kernel (then `n_modes` is needed, and
            `domain`, unless coordinates are in [-π, π]).
        domain: The data region; the GP's own by default.
        n_modes: Modes per side (an int, or two odd ints); the GP's by default.
        padding: Padding as a fraction of the extent; the GP's by default, or 0.15.
        length_scale: The kernel's length scale; its current value by default.
            An array gives the errors at each value.

    Returns:
        A DiscretisationError.
    """
    kernel, n_modes, geometry = _resolve(gp_or_kernel, domain, n_modes, padding)
    if length_scale is None:
        kernels, scales = [kernel], None
    else:
        scales = np.asarray(length_scale, dtype=float)
        kernels = [_with_length_scale(kernel, v) for v in scales.ravel()]
    rows = np.array(
        [(*_truncation(k, n_modes, geometry), _aliasing(k, geometry)) for k in kernels]
    )
    if scales is None:
        trunc, trunc_max, tau, alias = (float(v) for v in rows[0])
        scale_out = _length_scale(kernel)
    else:
        trunc, trunc_max, tau, alias = (rows[:, i].reshape(scales.shape) for i in range(4))
        scale_out = scales if scales.ndim else float(scales)
        if scales.ndim == 0:
            trunc, trunc_max, tau, alias = (float(v) for v in (trunc, trunc_max, tau, alias))
    return DiscretisationError(
        truncation=trunc,
        aliasing=alias,
        truncation_max=trunc_max,
        tau=tau,
        length_scale=scale_out,
        n_modes=n_modes,
        padding=geometry.padding,
    )


def required_modes(
    kernel: Kernel,
    domain: Domain,
    *,
    length_scale: float | None = None,
    padding: float = DEFAULT_PADDING,
    tol: float = DEFAULT_TOL,
) -> int:
    """The fewest modes per side (odd) with a truncation error at most `tol`.

    Truncation is worst at the shortest length scale, so pass the smallest
    one you expect (the kernel's current value by default).
    """
    _check_continuous_form(kernel)
    if length_scale is not None:
        kernel = _with_length_scale(kernel, length_scale)
    geometry = _geometry(domain, padding)

    def ok(m: int) -> bool:
        return _truncation(kernel, (m, m), geometry)[0] <= tol

    lo, hi = 1, 3
    while not ok(hi):
        lo, hi = hi, 2 * hi + 1
        if hi > MAX_MODES:
            raise ValueError(
                f"More than {MAX_MODES} modes per side would be needed for tol={tol:g}: the "
                "length scale is too short for the domain."
            )
    # Smallest odd m in (lo, hi] that passes; the error falls with m
    while hi - lo > 2:
        mid = (lo + hi) // 2
        mid += 1 - mid % 2
        if mid >= hi:
            break
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def required_padding(
    kernel: Kernel,
    domain: Domain,
    *,
    length_scale: float | None = None,
    tol: float = DEFAULT_TOL,
) -> float:
    """The least padding (a fraction of the extent, to 1%) with an aliasing error at most `tol`.

    Aliasing is worst at the longest length scale, so pass the largest one
    you expect (the kernel's current value by default).
    """
    _check_continuous_form(kernel)
    if length_scale is not None:
        kernel = _with_length_scale(kernel, length_scale)

    def alias(padding: float) -> float:
        return _aliasing(kernel, _geometry(domain, padding))

    if alias(0.0) <= tol:
        return 0.0
    hi = 0.25
    while alias(hi) > tol:
        hi *= 2
        if hi > 64:
            raise ValueError(
                f"More than 64 x the extent of padding would be needed for tol={tol:g}: the "
                "length scale is too long for the domain."
            )
    lo = 0.0
    while hi - lo > 0.01:
        mid = 0.5 * (lo + hi)
        if alias(mid) <= tol:
            hi = mid
        else:
            lo = mid
    return float(np.ceil(hi * 100) / 100)


def realised_covariance(gp: FourierGP, dx: ArrayLike, dy: ArrayLike) -> np.ndarray:
    """The field's covariance at lags (dx, dy), in the coordinates' units: exact, from the
    kernel's weights (no data or NUFFT needed)."""
    if not isinstance(gp, FourierGP):
        raise TypeError(f"Expected a FourierGP, got {type(gp).__name__}.")
    geometry = _geometry(gp.domain, gp.padding)
    jx, jy = _mode_grid(gp.n_modes)
    dk = 2 * np.pi / geometry.size
    k = dk * np.hypot(jx[:, None], jy[None, :])
    w = np.asarray(gp.kernel.feature_weights(jnp.asarray(k)), dtype=float) ** 2
    s2 = _variance(gp.kernel)
    dx, dy = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    phase = dk * (dx[..., None, None] * jx[:, None] + dy[..., None, None] * jy[None, :])
    return s2 * np.sum(w * np.cos(phase), axis=(-2, -1)) / w.sum()
