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

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any

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


def _weights(kernel: Kernel, n_modes: tuple[int, int], size: float, has_mean: bool = False):
    """The field's mode weights w_j and the kernel's trapezoid weights c_j (absolute
    units), and the dropped k = 0 weight c_0 (0 unless the field has a mean)."""
    jx, jy = _mode_grid(n_modes)
    dk = 2 * np.pi / size
    k = dk * np.hypot(jx[:, None], jy[None, :])
    w = np.asarray(kernel.feature_weights(jnp.asarray(k)), dtype=float) ** 2
    c = np.asarray(kernel.spectral_density(jnp.asarray(k)), dtype=float) * dk**2
    c0 = 0.0
    if has_mean:
        w = np.where(k == 0, 0.0, w)
        c0 = float(c[k == 0].sum())
    return w, c, c0


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


def _truncation(
    kernel: Kernel, n_modes: tuple[int, int], geometry: _Geometry, has_mean: bool = False
):
    """(relative Frobenius error, worst-case error / s², tau) from the finite mode grid."""
    s2 = _variance(kernel)
    w, c, c0 = _weights(kernel, n_modes, geometry.size, has_mean)
    tau = 1.0 - c.sum() / s2
    grid = _LagGrid(geometry, n_modes)
    realised = s2 * _on_lag_grid(w, grid) / w.sum()
    # The continuous kernel, periodised so the images (the aliasing error) don't
    # count as truncation, without the k = 0 mode if the field has a mean, and
    # scaled so its weights on the grid sum to s² as the field's do
    reference = _periodised(kernel, grid.zx, grid.zy, geometry, include_centre=True) - c0
    reference *= s2 / (c.sum() - c0)
    err = realised - reference
    frob = np.sqrt(np.sum(grid.q * err**2) / np.sum(grid.q * reference**2))
    return float(frob), float(np.abs(err).max() / s2), float(tau)


def _resolve(gp_or_kernel, domain, n_modes, padding, has_mean):
    """The kernel, mode grid, geometry and mean flag: from a FourierGP, with any overrides."""
    if isinstance(gp_or_kernel, FourierGP):
        gp = gp_or_kernel
        kernel = gp.kernel
        has_mean = gp.mean is not None
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
    return kernel, _check_n_modes(n_modes), _geometry(domain, padding), has_mean


def discretisation_error(
    gp_or_kernel: FourierGP | Kernel,
    *,
    domain: Domain | None = None,
    n_modes: int | tuple[int, int] | None = None,
    padding: float | None = None,
    length_scale: float | ArrayLike | None = None,
    has_mean: bool = False,
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
        has_mean: For a kernel: whether the field has a mean (its k = 0 mode is
            then dropped). A FourierGP's own setting is used otherwise.

    Returns:
        A DiscretisationError.
    """
    kernel, n_modes, geometry, has_mean = _resolve(
        gp_or_kernel, domain, n_modes, padding, has_mean
    )
    if length_scale is None:
        kernels, scales = [kernel], None
    else:
        scales = np.asarray(length_scale, dtype=float)
        kernels = [_with_length_scale(kernel, v) for v in scales.ravel()]
    rows = np.array(
        [(*_truncation(k, n_modes, geometry, has_mean), _aliasing(k, geometry)) for k in kernels]
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
    has_mean: bool = False,
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
        return _truncation(kernel, (m, m), geometry, has_mean)[0] <= tol

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
    if gp.mean is not None:
        w = np.where(k == 0, 0.0, w)
    s2 = _variance(gp.kernel)
    dx, dy = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    phase = dk * (dx[..., None, None] * jx[:, None] + dy[..., None, None] * jy[None, :])
    return s2 * np.sum(w * np.cos(phase), axis=(-2, -1)) / w.sum()


def _find_gps(obj: Any, path: tuple[str, ...] = ()):
    """(dotted path, FourierGP) for every FourierGP in a module tree."""
    if isinstance(obj, FourierGP):
        yield ".".join(path), obj
        return
    if isinstance(obj, eqx.Module) and dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield from _find_gps(getattr(obj, f.name), (*path, f.name))


def _resolve_model(model_or_latent: Any):
    """(evaluable model, latent space or None) from a ShareModule, LatentSpace or Posterior."""
    from spectracles.inference.latent import LatentSpace
    from spectracles.inference.posterior import Posterior
    from spectracles.model.share_module import ShareModule

    if isinstance(model_or_latent, Posterior):
        model_or_latent = model_or_latent.latent
    if isinstance(model_or_latent, LatentSpace):
        return model_or_latent.model.for_evaluation(), model_or_latent
    if isinstance(model_or_latent, ShareModule):
        return model_or_latent.for_evaluation(), None
    raise TypeError(
        "report takes a built model, a LatentSpace or a Posterior, got "
        f"{type(model_or_latent).__name__}."
    )


def _get(obj: Any, path: str) -> Any:
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


def report(
    model_or_latent: Any,
    samples: Any = None,
    *,
    tol: float | None = None,
    space: str = "latent",
    print_table: bool = True,
) -> dict[str, dict[str, Any]]:
    """Check every FourierGP in a model against its tolerance, at its current
    hyperparameters or at every sample.

    Truncation is checked at the shortest length scale across the samples and
    aliasing at the longest, which is where each is worst. A GP whose modes
    were chosen for you is checked against its own `tol` (and its design
    range of length scales is shown); others against 1e-3, or `tol`.

    Args:
        model_or_latent: A built model, a LatentSpace or a Posterior.
        samples: Samples (as for `posterior.draws`); needs a LatentSpace or
            Posterior. Without them, the model's current values are checked.
        tol: The tolerance for every GP, instead of each GP's own.
        space: Whether `samples` are latents ("latent") or parameter values ("params").
        print_table: Whether to print the table.

    Returns:
        {path: {"n_modes", "padding", "length_scale" (shortest, longest),
        "design" (or None), "truncation", "aliasing", "tol", "ok"}}.
    """
    model, latent = _resolve_model(model_or_latent)
    gps = dict(_find_gps(model.model))
    if not gps:
        raise ValueError("The model has no FourierGP.")
    with_ls = [p for p, g in gps.items() if hasattr(g.kernel, "length_scale")]
    if samples is not None:
        if latent is None:
            raise ValueError("Samples need a LatentSpace or Posterior, not a model.")
        draws = latent.map_samples(
            lambda m: {p: jnp.ravel(_get(m.model, p).kernel.length_scale.val) for p in with_ls},
            samples,
            space=space,  # type: ignore[arg-type]
        )
        ls = {p: np.asarray(draws.values[p]) for p in with_ls}
    else:
        ls = {p: np.asarray(jnp.ravel(gps[p].kernel.length_scale.val)) for p in with_ls}  # type: ignore[attr-defined]

    out: dict[str, dict[str, Any]] = {}
    for path, gp in gps.items():
        gp_tol = tol if tol is not None else (gp.design[0] if gp.design else DEFAULT_TOL)
        row: dict[str, Any] = {
            "n_modes": gp.n_modes,
            "padding": gp.padding,
            "design": gp.design,
            "tol": gp_tol,
        }
        try:
            _check_continuous_form(gp.kernel)
        except NotImplementedError:
            row.update(length_scale=None, truncation=None, aliasing=None, ok=None)
            out[path] = row
            continue
        if path in ls:
            lo, hi = float(ls[path].min()), float(ls[path].max())
            trunc = discretisation_error(gp, length_scale=lo).truncation
            alias = discretisation_error(gp, length_scale=hi).aliasing
        else:
            lo = hi = None
            e = discretisation_error(gp)
            trunc, alias = e.truncation, e.aliasing
        row.update(
            length_scale=None if lo is None else (lo, hi),
            truncation=float(trunc),
            aliasing=float(alias),
            ok=bool(max(trunc, alias) <= gp_tol),
        )
        out[path] = row

    if print_table:
        _print_report(out, samples is not None)
    failed = [p for p, r in out.items() if r["ok"] is False]
    if failed:
        warnings.warn(
            f"Discretisation error above tolerance for {failed}: see the table, and choose "
            "more modes or padding (or let FourierGP choose them for the length scales found).",
            stacklevel=2,
        )
    return out


def _print_report(rows: dict[str, dict[str, Any]], with_samples: bool) -> None:
    from rich.table import Table
    from rich.text import Text

    from spectracles.model.formatting import get_console

    def rng(r):
        return (
            "-" if r is None else (f"{r[0]:.4g}" if r[0] == r[1] else f"[{r[0]:.4g}, {r[1]:.4g}]")
        )

    table = Table(
        title="Discretisation",
        show_header=True,
        header_style="bold",
        border_style="dim",
        title_style="bold",
    )
    table.add_column("Path", style="white", overflow="fold")
    table.add_column("Modes", style="dim")
    table.add_column("Padding", style="dim")
    table.add_column("ℓ (samples)" if with_samples else "ℓ")
    table.add_column("ℓ designed for", style="dim")
    table.add_column("Truncation")
    table.add_column("Aliasing")
    table.add_column("OK")
    for path, r in rows.items():
        design = None if r["design"] is None else r["design"][1:]
        if r["ok"] is None:
            cells = ["n/a", "n/a", Text("?", style="yellow")]
        else:
            mark = Text("✓", style="green") if r["ok"] else Text("✗", style="bold red")
            cells = [f"{r['truncation']:.1e}", f"{r['aliasing']:.1e}", mark]
        pad = "-" if r["padding"] is None else f"{r['padding']:g}"
        table.add_row(
            path,
            f"{r['n_modes'][0]}×{r['n_modes'][1]}",
            pad,
            rng(r["length_scale"]),
            rng(design),
            *cells,
        )
    console = get_console()
    console.print(table)
    tols = sorted({r["tol"] for r in rows.values()})
    console.print(
        Text(
            "truncation at the shortest ℓ, aliasing at the longest; relative to s²; tol "
            + ", ".join(f"{t:g}" for t in tols),
            style="dim",
        )
    )


def plot_discretisation(
    gp: FourierGP,
    length_scales: ArrayLike | None = None,
    *,
    tol: float | None = None,
    ax: Any = None,
):
    """Plot a FourierGP's truncation and aliasing errors against the length scale.

    Shows the tolerance, the range of length scales the GP was designed for
    (if its modes were chosen for you) and, optionally, the length scales
    found in a fit (e.g. posterior samples), as ticks along the bottom.

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if not isinstance(gp, FourierGP):
        raise TypeError(f"Expected a FourierGP, got {type(gp).__name__}.")
    tol = tol if tol is not None else (gp.design[0] if gp.design else DEFAULT_TOL)
    found = None if length_scales is None else np.ravel(np.asarray(length_scales, dtype=float))
    centre = [_length_scale(gp.kernel) or 1.0]
    if gp.design:
        centre += list(gp.design[1:])
    if found is not None and found.size:
        centre += [found.min(), found.max()]
    grid = np.geomspace(min(centre) / 4, max(centre) * 4, 50)
    e = discretisation_error(gp, length_scale=grid)
    if ax is None:
        _, ax = plt.subplots()
    ax.loglog(grid, e.truncation, label="truncation")
    ax.loglog(grid, e.aliasing, label="aliasing")
    ax.axhline(tol, color="0.5", lw=0.8, label=f"tol {tol:g}")
    if gp.design:
        ax.axvspan(gp.design[1], gp.design[2], color="0.9", zorder=0, label="designed for")
    if found is not None and found.size:
        ax.plot(
            found,
            np.full(found.shape, 0.02),
            "|",
            color="k",
            ms=8,
            transform=ax.get_xaxis_transform(),
            label="length scales found",
        )
    ax.set_ylim(tol * 1e-5, 2.0)  # the errors span hundreds of decades otherwise
    ax.set_xlabel("length scale")
    ax.set_ylabel("error / s²")
    ax.set_title(f"{gp.n_modes[0]}×{gp.n_modes[1]} modes, padding {gp.padding}")
    ax.legend()
    return ax
