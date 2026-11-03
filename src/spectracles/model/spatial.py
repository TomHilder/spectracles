"""spatial.py - Spatial models, especially flexible models built on Fourier-accelerated Gaussian Processes."""

import warnings
from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from equinox import Module, field
from jax.core import Tracer
from jax.nn import softplus
from jax.scipy.stats import norm
from jax_finufft import nufft2  # type: ignore
from jaxtyping import Array

from spectracles.model.data import SpatialData, SpatialDataLVM
from spectracles.model.domain import Domain
from spectracles.model.kernels import Kernel
from spectracles.model.parameter import AnyParameter, Parameter, _is_int, init_parameter
from spectracles.priors import StdNormal

# NOTE: List of current obvious foot guns:
# - n_modes must always be two odd integers, but this is not enforced
#   - in theory the nifty-solve full implementation should be able to handle
#     this, but from memory it doesn't work right now. This is why p is
#     repeated in the shape info right now, since I didn't implement n_modes
#     != n_requested_modes yet.

LINKS = {
    "identity": lambda f, lower: f,
    "exp": lambda f, lower: jnp.exp(f),
    "softplus": lambda f, lower: lower + softplus(f),
}

DEFAULT_PADDING = 0.15
DEFAULT_NUFFT_TOL = 1e-10


def _box(domain: Domain | None, padding: float | None) -> tuple[float, float, float]:
    """The periodic box (centre x, centre y, side) for a domain and padding.

    Without a domain, coordinates are already in [-π, π]: the box is that square.
    """
    if domain is None:
        if padding is not None:
            raise ValueError("padding needs a domain: FourierGP(..., domain=Domain(...)).")
        return 0.0, 0.0, 2 * np.pi
    if not isinstance(domain, Domain):
        raise TypeError(f"domain must be a Domain, got {type(domain).__name__}.")
    padding = DEFAULT_PADDING if padding is None else float(padding)
    if not (np.isfinite(padding) and padding >= 0):
        raise ValueError(f"padding must be a non-negative number, got {padding!r}.")
    return domain.centre[0], domain.centre[1], domain.box_size(padding)


def _length_scale_range(
    kernel: Kernel, length_scale, quantiles: tuple[float, float]
) -> tuple[float, float]:
    """The shortest and longest length scales to choose modes and padding for."""
    if length_scale is not None:
        lo, hi = (length_scale, length_scale) if np.ndim(length_scale) == 0 else length_scale
        lo, hi = float(lo), float(hi)
        if not (0 < lo <= hi and np.isfinite(hi)):
            raise ValueError(
                f"length_scale must be (shortest, longest) > 0, got {length_scale!r}."
            )
        return lo, hi
    param = getattr(kernel, "length_scale", None)
    if param is None:
        raise ValueError(
            f"{type(kernel).__name__} has no length_scale, so modes can't be chosen for you: "
            "give n_modes (and padding)."
        )
    if param.prior is None or param.fix:
        value = float(jnp.max(param.val))
        return value, value
    q_lo, q_hi = quantiles
    if not 0 < q_lo <= q_hi < 1:
        raise ValueError(f"quantiles must be in (0, 1), got {quantiles!r}.")
    lo = float(jnp.min(param.prior.transform(norm.ppf(jnp.asarray(q_lo)))))
    hi = float(jnp.max(param.prior.transform(norm.ppf(jnp.asarray(q_hi)))))
    return lo, hi


def _choose_modes(kernel, domain, padding, tol, length_scale, quantiles, has_mean=False):
    """Modes and padding for `tol` over the length-scale range, and the design record."""
    from spectracles.discretisation import (
        DEFAULT_TOL,
        discretisation_error,
        required_modes,
        required_padding,
    )

    if domain is None:
        raise ValueError(
            "Without a domain, n_modes is needed: FourierGP(kernel, n_modes=(m, m)). Give a "
            "domain to have the modes chosen for you."
        )
    tol = DEFAULT_TOL if tol is None else float(tol)
    lo, hi = _length_scale_range(kernel, length_scale, quantiles)
    if padding is None:
        padding = required_padding(kernel, domain, length_scale=hi, tol=tol)
    else:
        alias = discretisation_error(
            kernel, domain=domain, n_modes=3, padding=padding, length_scale=hi
        ).aliasing
        if alias > tol:
            warnings.warn(
                f"padding={padding} gives an aliasing error of {alias:.1e} at length scale "
                f"{hi:.4g}, above tol={tol:g}.",
                stacklevel=3,
            )
    try:
        m = required_modes(
            kernel, domain, length_scale=lo, padding=padding, tol=tol, has_mean=has_mean
        )
    except ValueError as e:
        raise ValueError(
            f"{e} The shortest length scale is {lo:.4g}: narrow the prior, pass "
            "length_scale=(shortest, longest), or give n_modes."
        ) from None
    return (m, m), padding, (tol, lo, hi)


def _check_in_box(x, y, box: tuple[float, float, float]) -> None:
    """Raise if any point is outside the periodic box (it would wrap to the other side).

    Runs on concrete positions only, so it costs nothing inside compiled code;
    positions that are themselves traced (e.g. jit arguments) aren't checked.
    """
    if isinstance(x, Tracer) or isinstance(y, Tracer):
        return
    cx, cy, size = box
    half = 0.5 * size * (1 + 1e-12)
    x, y = np.asarray(x), np.asarray(y)
    if x.size == 0:
        return
    dx, dy = np.abs(x - cx).max(), np.abs(y - cy).max()
    if dx > half or dy > half:
        raise ValueError(
            f"Points lie outside the GP's periodic box (centre ({cx:.6g}, {cy:.6g}), side "
            f"{size:.6g}): x spans [{x.min():.6g}, {x.max():.6g}], y spans "
            f"[{y.min():.6g}, {y.max():.6g}]. They would wrap to the opposite side. Use a "
            "Domain that covers them, or more padding."
        )


def _to_box(x, y, box: tuple[float, float, float]):
    """Map coordinates into the box's [-π, π] square."""
    _check_in_box(x, y, box)
    cx, cy, size = box
    scale = 2 * np.pi / size
    return (x - cx) * scale, (y - cy) * scale


def _check_n_modes(n_modes) -> tuple[int, int]:
    """Two positive odd integers, as a tuple."""
    if isinstance(n_modes, list):
        n_modes = tuple(n_modes)
    if not (isinstance(n_modes, tuple) and len(n_modes) == 2 and all(_is_int(m) for m in n_modes)):
        raise TypeError(f"n_modes must be two integers, e.g. (31, 31); got {n_modes!r}.")
    if any(m < 1 or m % 2 == 0 for m in n_modes):
        raise ValueError(
            f"n_modes must be two positive odd integers, e.g. (31, 31); got {n_modes}."
        )
    return (int(n_modes[0]), int(n_modes[1]))


def _check_kernel(kernel, name: str = "kernel") -> Kernel:
    if not isinstance(kernel, Kernel):
        raise TypeError(
            f"{name} must be a Kernel, e.g. Matern32(length_scale, variance); "
            f"got {type(kernel).__name__}."
        )
    return kernel


def get_freqs_1D(n_modes: int) -> Array:
    if n_modes % 2 == 0:
        return jnp.arange(-n_modes // 2, n_modes // 2, dtype=float)
    else:
        return jnp.arange(-(n_modes - 1) // 2, (n_modes - 1) // 2 + 1, dtype=float)


def get_freqs(n_modes: int | tuple[int, ...], n_dim: int = 2) -> Array | tuple[Array, ...]:
    if n_dim == 1 and isinstance(n_modes, int):
        return get_freqs_1D(n_modes)
    elif isinstance(n_modes, tuple):
        if len(n_modes) != n_dim:
            raise ValueError(
                f"n_modes must be a tuple of length {n_dim} for {n_dim}D data, but got {len(n_modes)}",
            )
        modes_grid = jnp.meshgrid(*[get_freqs_1D(n_modes[i]) for i in range(n_dim)], indexing="ij")
        # Transpose because fiNUFFT treats the first dimension as the fastest changing
        return modes_grid
    else:
        raise ValueError(
            f"n_modes must be an int or a tuple of ints, but got {type(n_modes)}",
        )


def conjugate_symmetry(
    c: Array,
    n_modes: tuple[int, int],
    shape_info: tuple[int, int, int],
) -> Array:
    m, h, p = shape_info
    # The zero mode (centre, index h for odd n_modes) is its own conjugate
    # partner, so f + conj(flip(f)) below adds it to itself, while each +-k pair
    # combines two independent coefficients. Scale it by 1/sqrt(2) so its
    # variance matches the other modes (otherwise k(0) = s^2 (1 + fw_0^2 / 2)).
    c = c.at[h].multiply(1 / jnp.sqrt(2))
    f = 0.5 * jnp.hstack(
        [c[: h + 1], jnp.zeros(p - h - 1)],
    ) + 0.5j * jnp.hstack(
        [jnp.zeros(p - m + h + 1), c[h + 1 :]],
    )
    f = f.reshape(n_modes)
    return f + jnp.conj(jnp.flip(f))


class SpatialModel(Module):
    @abstractmethod
    def __call__(self, data: SpatialData) -> Array:
        pass


class FourierBasis(SpatialModel):
    """A spatial model using a fixed Fourier basis with learnable coefficients.

    Note that this is different to a FourierGP, which uses a kernel to
    regularise the coefficients.
    """

    coefficients: Parameter
    n_modes: tuple[int, int]
    domain: Domain | None = field(static=True)
    padding: float | None = field(static=True)
    nufft_tol: float = field(static=True)
    _box: tuple[float, float, float] = field(static=True)
    _freqs: Array
    _shape_info: tuple[int, int, int]

    def __init__(
        self,
        n_modes: tuple[int, int],
        coefficients: Parameter | None = None,
        *,
        domain: Domain | None = None,
        padding: float | None = None,
        nufft_tol: float = DEFAULT_NUFFT_TOL,
    ):
        # Model specfication
        self.n_modes = n_modes = _check_n_modes(n_modes)
        self._box = _box(domain, padding)
        self.domain = domain
        self.padding = (
            None if domain is None else (DEFAULT_PADDING if padding is None else float(padding))
        )
        self.nufft_tol = float(nufft_tol)
        fx, fy = get_freqs(n_modes)
        self._freqs = (2 * np.pi / self._box[2]) * jnp.sqrt(fx**2 + fy**2)
        # Initialise parameters
        self.coefficients = init_parameter(coefficients, name="coefficients", dims=n_modes)
        # Initialise the shape info
        p = int(jnp.prod(jnp.array(n_modes)))
        self._shape_info = (p, p // 2, p)

    def __call__(self, data: SpatialData) -> Array:
        x, y = _to_box(data.x, data.y, self._box)
        # Sum basis functions with nufft
        model_eval: Array = nufft2(
            conjugate_symmetry(self.coefficients.val.flatten(), self.n_modes, self._shape_info),
            x,
            y,
            eps=self.nufft_tol,
        ).real
        return model_eval

    def prior_logpdf(self) -> Array:
        raise NotImplementedError(
            "FourierBasis does not have a prior_logpdf method for now. If you want this, message/email Tom Hilder."
        )


class FourierGP(SpatialModel):
    """A Gaussian-process field, represented by Fourier modes on a periodic box.

    With a `domain`, coordinates and length scales are in the domain's units
    (e.g. arcsec), and the field lives on a periodic box of the domain plus
    `padding` × extent on each side. The modes and padding are then chosen for
    you, so that the field matches its kernel to `tol` over the range of
    length scales it can take (by default, its prior's 1% to 99% quantiles):

        FourierGP(kernel, domain=domain)
        FourierGP(kernel, domain=domain, tol=1e-4, length_scale=(30.0, 300.0))
        FourierGP(kernel, domain=domain, n_modes=(301, 301), padding=0.15)  # by hand

    Without a `domain`, coordinates must already lie in [-π, π], the length
    scale is in those units, and `n_modes` is needed.

    A field can have a mean, a `Parameter` added to the GP, and a `link` that
    transforms the result, e.g. for a positive line flux:

        FourierGP(kernel, domain=domain, mean=Parameter(prior=Normal(0.0, 1.0)), link="exp")

    With a mean, the GP's own constant (k = 0) mode is dropped, so the mean
    alone sets the field's average over the box; the kernel's variance is
    still the field's pointwise variance.

    Args:
        kernel: The kernel, e.g. `Matern32(length_scale, variance)`.
        n_modes: Modes in each direction, two positive odd integers. Chosen
            from `tol` if not given (needs a domain).
        domain: The region the data cover, in the data's units.
        padding: The margin on each side of the domain, as a fraction of its
            extent. Chosen from `tol` if not given, or 0.15 when `n_modes` is.
        tol: The accuracy the modes and padding are chosen for (1e-3 by
            default): the truncation error at the shortest length scale and
            the aliasing error at the longest (see `spectracles.discretisation`).
        length_scale: The range (shortest, longest) of length scales to choose
            for, or a single value; the prior's `quantiles` by default.
        quantiles: The prior quantiles that give the default range.
        mean: A Parameter added to the GP before the link. None (default)
            means no mean: a zero-mean stationary GP.
        link: "identity" (default), "exp" (positive, log-normal) or
            "softplus" (`lower` + softplus, positive and close to linear when large).
        lower: The lower bound for `link="softplus"`.
        coefficients: The (whitened) Fourier coefficients; a standard-normal
            prior by default.
        zeromode: Deprecated name for `mean`.
        nufft_tol: The relative accuracy of the non-uniform FFT.

    Attributes:
        design: (tol, shortest, longest length scale) the modes and padding
            were chosen for, or None if they were given by hand.
    """

    coefficients: Parameter
    kernel: Kernel
    mean: Parameter | None
    n_modes: tuple[int, int]
    link: str = field(static=True)
    lower: float = field(static=True)
    domain: Domain | None = field(static=True)
    padding: float | None = field(static=True)
    nufft_tol: float = field(static=True)
    design: tuple[float, float, float] | None = field(static=True)
    _box: tuple[float, float, float] = field(static=True)
    _freqs: Array
    _shape_info: tuple[int, int, int]

    def __init__(
        self,
        kernel: Kernel,
        *legacy,
        n_modes: tuple[int, int] | None = None,
        domain: Domain | None = None,
        padding: float | None = None,
        tol: float | None = None,
        length_scale: float | tuple[float, float] | None = None,
        quantiles: tuple[float, float] = (0.01, 0.99),
        mean: Parameter | None = None,
        link: str = "identity",
        lower: float = 0.0,
        coefficients: Parameter | None = None,
        zeromode: Parameter | None = None,
        nufft_tol: float = DEFAULT_NUFFT_TOL,
    ):
        if legacy:
            # FourierGP(n_modes, kernel, coefficients, zeromode): the signature before 1.0
            if len(legacy) > 3 or n_modes is not None:
                raise TypeError(
                    "FourierGP takes the kernel first: FourierGP(kernel, n_modes=..., ...)."
                )
            warnings.warn(
                "FourierGP(n_modes, kernel) is deprecated: use FourierGP(kernel, n_modes=n_modes), "
                "or give a domain and leave out n_modes to have it chosen.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_modes, kernel = kernel, legacy[0]  # type: ignore[assignment]
            coefficients = legacy[1] if len(legacy) > 1 else coefficients
            zeromode = legacy[2] if len(legacy) > 2 else zeromode
        if zeromode is not None:
            if mean is not None:
                raise TypeError("Give mean or zeromode (its old name), not both.")
            warnings.warn(
                "zeromode is deprecated: use mean=. With a mean, the GP's k = 0 mode is dropped.",
                DeprecationWarning,
                stacklevel=2,
            )
            mean = zeromode
        kernel = _check_kernel(kernel)
        if mean is not None and not isinstance(mean, AnyParameter):
            raise TypeError(
                f"mean must be a Parameter, e.g. mean=Parameter(prior=Normal(0.0, 1.0)); got "
                f"{type(mean).__name__}."
            )
        if link not in LINKS:
            raise ValueError(f"link must be one of {sorted(LINKS)}, got {link!r}.")
        if link != "softplus" and lower != 0.0:
            raise ValueError('lower only applies to link="softplus".')
        self.mean = mean  # type: ignore[assignment]
        self.link = link
        self.lower = float(lower)
        if n_modes is None:
            n_modes, padding, self.design = _choose_modes(
                kernel, domain, padding, tol, length_scale, quantiles, has_mean=mean is not None
            )
        else:
            if tol is not None or length_scale is not None:
                raise ValueError(
                    "tol and length_scale choose n_modes; give them or n_modes, not both."
                )
            self.design = None
        # Model specfication
        self.n_modes = n_modes = _check_n_modes(n_modes)
        self._box = _box(domain, padding)
        self.domain = domain
        self.padding = (
            None if domain is None else (DEFAULT_PADDING if padding is None else float(padding))
        )
        self.nufft_tol = float(nufft_tol)
        # Physical frequencies, so the kernel sees length scales in the coordinates' units
        fx, fy = get_freqs(n_modes)
        self._freqs = (2 * np.pi / self._box[2]) * jnp.sqrt(fx**2 + fy**2)
        self.kernel = kernel
        # Initialise parameters. The coefficients are whitened (the kernel scales
        # them), so their prior is a standard normal by construction.
        # A user-supplied Parameter is kept as is (the same object, so it can be
        # shared between GPs), with whatever prior it has.
        self.coefficients = init_parameter(
            coefficients, name="coefficients", dims=n_modes, prior=StdNormal()
        )
        if mean is not None and jnp.size(mean.val) != 1:
            raise ValueError(f"mean must be a single value, got shape {jnp.shape(mean.val)}.")
        # Initialise the shape info
        p = int(jnp.prod(jnp.array(n_modes)))
        self._shape_info = (p, p // 2, p)

    @property
    def zeromode(self) -> Parameter | None:
        """Deprecated name for `mean`."""
        warnings.warn(
            "FourierGP.zeromode is deprecated: use .mean.", DeprecationWarning, stacklevel=2
        )
        return self.mean

    def feature_weights(self) -> Array:
        """The square root of each mode's power: the kernel's, without the k = 0
        mode when the field has a mean (rescaled so the variance is unchanged)."""
        fw = self.kernel.feature_weights(self._freqs)
        if self.mean is None:
            return fw
        kept = jnp.where(self._freqs == 0, 0.0, fw)
        return kept * jnp.sqrt(jnp.sum(fw**2) / jnp.sum(kept**2))

    def __call__(self, data: SpatialData) -> Array:
        x, y = _to_box(data.x, data.y, self._box)
        # Feature weighted coefficients
        scaled_coeffs = self.coefficients.val * self.feature_weights()
        # Sum basis functions with nufft after processing the coefficients to enforce conjugate symmetry
        model_eval: Array = nufft2(
            conjugate_symmetry(scaled_coeffs.flatten(), self.n_modes, self._shape_info),
            x,
            y,
            eps=self.nufft_tol,
        ).real
        if self.mean is not None:
            model_eval = model_eval + self.mean.val
        return LINKS[self.link](model_eval, self.lower)

    def prior_logpdf(self) -> Array:
        fw = self.feature_weights()
        # DC-safe: kernels with zero DC contribute no Jacobian term for that mode.
        safe_fw = jnp.where(fw > 0, fw, 1.0)
        log_normalisation = jnp.where(fw > 0, -jnp.log(safe_fw), 0.0).sum()
        return norm.logpdf(x=self.coefficients.val).sum() + log_normalisation


class MultiFourierGP(SpatialModel):
    """A collection of independent FourierGPs grouped to better batch through fiNUFFT calls.
    Requires that all component GPs use the same number of modes, but can use different kernels.

    Not available yet: instantiating it raises NotImplementedError. It holds its
    kernels in a list, which built models will support from 1.1. The aim is
    faster models with several fields on GPU, since such fields usually share
    one frequency grid and can go through one batched NUFFT.
    """

    coefficients: Parameter
    kernels: list[Kernel]
    n_modes: tuple[int, int]
    _freqs: Array
    _shape_info: tuple[int, int, int]
    _kernels_idx: Array

    def __init__(
        self,
        n_components: int,
        n_modes: tuple[int, int],
        kernels: list[Kernel],
        coefficients: Parameter | None = None,
    ):
        raise NotImplementedError(
            "MultiFourierGP is not available yet: it holds its kernels in a list, which "
            "built models will support from spectracles 1.1. Use one FourierGP per field."
        )
        # Model specfication
        self.n_modes = n_modes = _check_n_modes(n_modes)
        fx, fy = get_freqs(n_modes)
        self._freqs = jnp.sqrt(fx**2 + fy**2)
        if len(kernels) != n_components:
            raise ValueError(f"Got {len(kernels)} kernels for {n_components} components.")
        self.kernels = [_check_kernel(k, "each kernel") for k in kernels]
        # Initialise parameters
        self.coefficients = init_parameter(
            coefficients, name="coefficients", dims=(n_components, *n_modes), prior=StdNormal()
        )
        # Initialise the shape info
        p = int(jnp.prod(jnp.array(n_modes)))
        self._shape_info = (p, p // 2, p)
        # Kernel indices for vmap
        self._kernels_idx = jnp.arange(n_components)

    def __call__(self, data: SpatialData) -> Array:
        # Process each component separately since kernels can be different types
        results = []
        for c, kernel in zip(self.coefficients.val, self.kernels):
            scaled_coeffs = c * kernel.feature_weights(self._freqs)
            coeffs_processed = conjugate_symmetry(
                scaled_coeffs.flatten(), self.n_modes, self._shape_info
            )
            results.append(coeffs_processed)

        coeffs_for_nufft = jnp.stack(results)  # (n_components, n_modes[0]*n_modes[1])

        model_eval: Array = nufft2(
            coeffs_for_nufft,  # (n_components, n_data)
            data.x,  # (1, n_data) where 1 is implicit
            data.y,  # (1, n_data) where 1 is implicit
            eps=DEFAULT_NUFFT_TOL,
        ).real  # (n_components, n_data)
        return model_eval

    def prior_logpdf(self) -> Array:
        raise NotImplementedError(
            "MultiFourierGP prior_logpdf not implemented yet. If you want this, message/email Tom Hilder."
        )
        # def single_component_logpdf(c: Array, kernel: Kernel) -> Array:
        #     fw = kernel.feature_weights(self._freqs)
        #     jacobian = -0.5 * jnp.log(fw).sum()
        #     return norm.logpdf(x=c).sum() + jacobian

        # logpdfs = vmap(single_component_logpdf, in_axes=(0, 0))(
        #     self.coefficients.val, jnp.array(self.kernels)
        # )  # (n_components,)
        # return logpdfs.sum()


class PerSpaxel(SpatialModel):
    # Model parameters
    spaxel_values: Parameter

    def __init__(self, n_spaxels: int, spaxel_values: Parameter | None = None):
        self.spaxel_values = init_parameter(spaxel_values, name="spaxel_values", dims=n_spaxels)

    def __call__(self, data: SpatialData) -> Array:
        return self.spaxel_values.val[data.idx]


class PerIFUAndTile(SpatialModel):
    # Model parameters
    ifu_values: Parameter

    def __init__(self, n_tiles: int, n_ifus: int, ifu_values: Parameter | None = None):
        self.ifu_values = init_parameter(ifu_values, name="ifu_values", dims=(n_tiles, n_ifus))

    def __call__(self, data: SpatialDataLVM) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.ifu_values.val[data.tile_idx, data.ifu_idx]


class PerIFU(SpatialModel):
    # Model parameters
    ifu_values: Parameter

    def __init__(self, n_ifus: int, ifu_values: Parameter | None = None):
        self.ifu_values = init_parameter(ifu_values, name="ifu_values", dims=n_ifus)

    def __call__(self, data: SpatialDataLVM) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.ifu_values.val[data.ifu_idx]


class PerTile(SpatialModel):
    # Model parameters
    tile_values: Parameter

    def __init__(self, n_tiles: int, tile_values: Parameter | None = None):
        self.tile_values = init_parameter(tile_values, name="tile_values", dims=n_tiles)

    def __call__(self, data: SpatialDataLVM) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.tile_values.val[data.tile_idx]


class PerTilePinned(SpatialModel):
    # Model parameters (n_tiles - 1 since one is pinned)
    tile_values: AnyParameter
    pinned_val: float

    def __init__(
        self,
        n_tiles: int,
        tile_values: Parameter | None = None,
        pinned_val: float = 0.0,
    ):
        if n_tiles < 2:
            raise ValueError("n_tiles must be at least 2 for PerTilePinned")
        self.tile_values = init_parameter(tile_values, name="tile_values", dims=n_tiles - 1)
        self.pinned_val = pinned_val

    def __call__(self, data: SpatialDataLVM) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        tile_values_full = jnp.insert(self.tile_values.val, 0, self.pinned_val)
        return tile_values_full[data.tile_idx]
