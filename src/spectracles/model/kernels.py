"""kernels.py - Kernel classes that implement various covariance functions expressed by their power spectral density or 'feature weights' for use with Fourier-accelerated GP models."""

from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, ArrayLike

from spectracles.model.parameter import AnyParameter, Parameter


def normalise_fw(fw: ArrayLike) -> Array:
    """
    Normalise the feature weights. Includes a factor of sqrt(2) which accounts for the
    halving in total power incurred by enforcing the Fourier coefficients to be real via
    conjugate symmetry.
    """
    power = jnp.sum(jnp.abs(fw) ** 2)
    return jnp.sqrt(2) * fw / jnp.sqrt(power)


def matern_kernel_fw_nd(
    freqs: ArrayLike,
    length: ArrayLike,
    var: ArrayLike,
    nu: float,
    n: int,
) -> Array:
    """Square root of the PSD of the Matern kernel in n dimensions."""
    fw = (1 + (freqs * length) ** 2) ** (-0.5 * (nu + n / 2))
    return jnp.sqrt(var) * normalise_fw(fw)


def _hyperparameter(kernel: str, name: str, value) -> Parameter:
    if not isinstance(value, AnyParameter):
        raise TypeError(
            f"{kernel}: {name} must be a Parameter, e.g. "
            f"{name}=Parameter(prior=LogNormal(0.5, 0.2)); got {type(value).__name__}."
        )
    return value  # type: ignore[return-value]


def matern_spectral_density(k: ArrayLike, length: ArrayLike, var: ArrayLike, nu: float) -> Array:
    """The 2D Matern spectral density, normalised so that its integral over k is `var`."""
    length, var = jnp.asarray(length), jnp.asarray(var)
    return var * (nu * length**2 / jnp.pi) * (1 + (jnp.asarray(k) * length) ** 2) ** (-(nu + 1))


def matern_covariance(r: ArrayLike, length: ArrayLike, var: ArrayLike, nu: float) -> Array:
    """The Matern covariance for half-integer nu, in spectracles' convention (x = r / length)."""
    x = jnp.abs(jnp.asarray(r)) / jnp.asarray(length)
    if nu == 0.5:
        poly = 1.0
    elif nu == 1.5:
        poly = 1 + x
    elif nu == 2.5:
        poly = 1 + x + x**2 / 3
    else:
        raise ValueError(f"Only nu = 1/2, 3/2 and 5/2 have a closed form here, got {nu}.")
    return jnp.asarray(var) * poly * jnp.exp(-x)


class Kernel(Module):
    """Base class for kernels. Subclasses declare whatever hyperparameter fields
    they need. The only contract is `feature_weights(freqs) -> sqrt(P(k))` on the
    mode grid.

    Kernels can also give their continuous form, which the discretisation
    diagnostic compares the field against: `spectral_density(k)`, the 2D
    spectral density normalised so that its integral over k is the variance,
    and `covariance(r)`. Both raise `NotImplementedError` unless a kernel
    defines them.
    """

    @abstractmethod
    def feature_weights(self, freqs: Array) -> Array:
        pass

    def spectral_density(self, k: ArrayLike) -> Array:
        """The 2D spectral density S(|k|), with integral over k equal to k(0)."""
        raise NotImplementedError(f"{type(self).__name__} has no spectral_density.")

    def covariance(self, r: ArrayLike) -> Array:
        """The covariance k(r) at separation r."""
        raise NotImplementedError(f"{type(self).__name__} has no covariance.")


class Matern12(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: Parameter, variance: Parameter):
        self.length_scale = _hyperparameter("Matern12", "length_scale", length_scale)
        self.variance = _hyperparameter("Matern12", "variance", variance)

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale.val, self.variance.val, nu=0.5, n=2)

    def spectral_density(self, k: ArrayLike) -> Array:
        return matern_spectral_density(k, self.length_scale.val, self.variance.val, nu=0.5)

    def covariance(self, r: ArrayLike) -> Array:
        return matern_covariance(r, self.length_scale.val, self.variance.val, nu=0.5)


class Matern32(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: Parameter, variance: Parameter):
        self.length_scale = _hyperparameter("Matern32", "length_scale", length_scale)
        self.variance = _hyperparameter("Matern32", "variance", variance)

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale.val, self.variance.val, nu=1.5, n=2)

    def spectral_density(self, k: ArrayLike) -> Array:
        return matern_spectral_density(k, self.length_scale.val, self.variance.val, nu=1.5)

    def covariance(self, r: ArrayLike) -> Array:
        return matern_covariance(r, self.length_scale.val, self.variance.val, nu=1.5)


class Matern52(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: Parameter, variance: Parameter):
        self.length_scale = _hyperparameter("Matern52", "length_scale", length_scale)
        self.variance = _hyperparameter("Matern52", "variance", variance)

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale.val, self.variance.val, nu=2.5, n=2)

    def spectral_density(self, k: ArrayLike) -> Array:
        return matern_spectral_density(k, self.length_scale.val, self.variance.val, nu=2.5)

    def covariance(self, r: ArrayLike) -> Array:
        return matern_covariance(r, self.length_scale.val, self.variance.val, nu=2.5)


class SquaredExponential(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: Parameter, variance: Parameter):
        self.length_scale = _hyperparameter("SquaredExponential", "length_scale", length_scale)
        self.variance = _hyperparameter("SquaredExponential", "variance", variance)

    def feature_weights(self, freqs: Array) -> Array:
        fw = jnp.exp(-0.25 * freqs**2 * self.length_scale.val**2 + 1e-4)
        return jnp.sqrt(self.variance.val) * normalise_fw(fw)

    def spectral_density(self, k: ArrayLike) -> Array:
        ℓ = self.length_scale.val
        return self.variance.val * (ℓ**2 / (2 * jnp.pi)) * jnp.exp(-0.5 * (k * ℓ) ** 2)

    def covariance(self, r: ArrayLike) -> Array:
        return self.variance.val * jnp.exp(-0.5 * (jnp.asarray(r) / self.length_scale.val) ** 2)


class LogLogLinear(Kernel):
    """Log-log-linear CFM kernel: log-amplitude PSD is linear in log(k).

    `feature_weights(freqs) = exp(m_0 + m_s * log(k))` with DC (k=0) forced to zero
    so the fluctuation field has zero mean by construction. The field's spatial
    mean should be provided separately via `FourierGP.zeromode`.

    `m_s` is the spectral index; `m_0` the log-amplitude. For negative `m_s`
    (the physically common regime) the continuous PSD is IR-divergent, which is
    why the DC term must be handled as a separate Parameter outside the PSD.
    """

    m_0: Parameter
    m_s: Parameter

    def __init__(self, m_0: Parameter, m_s: Parameter):
        self.m_0 = _hyperparameter("LogLogLinear", "m_0", m_0)
        self.m_s = _hyperparameter("LogLogLinear", "m_s", m_s)

    def feature_weights(self, freqs: Array) -> Array:
        safe_k = jnp.where(freqs == 0, 1.0, freqs)
        log_k = jnp.log(safe_k)
        fw = jnp.exp(self.m_0.val + self.m_s.val * log_k)
        return jnp.where(freqs == 0, 0.0, fw)
