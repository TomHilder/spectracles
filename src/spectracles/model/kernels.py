"""kernels.py - Kernel classes that implement various covariance functions expressed by their power spectral density or 'feature weights' for use with Fourier-accelerated GP models."""

from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, ArrayLike

from spectracles.model.parameter import Parameter


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


class Kernel(Module):
    """Base class for kernels. Subclasses declare whatever hyperparameter fields
    they need. The only contract is `feature_weights(freqs) -> sqrt(P(k))` on the
    mode grid."""

    @abstractmethod
    def feature_weights(self, freqs: Array) -> Array:
        pass


class Matern12(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: Parameter, variance: Parameter):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale.val, self.variance.val, nu=0.5, n=2)


class Matern32(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: Parameter, variance: Parameter):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale.val, self.variance.val, nu=1.5, n=2)


class Matern52(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: Parameter, variance: Parameter):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale.val, self.variance.val, nu=2.5, n=2)


class SquaredExponential(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: Parameter, variance: Parameter):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        fw = jnp.exp(-0.25 * freqs**2 * self.length_scale.val**2 + 1e-4)
        return jnp.sqrt(self.variance.val) * normalise_fw(fw)


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
        self.m_0 = m_0
        self.m_s = m_s

    def feature_weights(self, freqs: Array) -> Array:
        safe_k = jnp.where(freqs == 0, 1.0, freqs)
        log_k = jnp.log(safe_k)
        fw = jnp.exp(self.m_0.val + self.m_s.val * log_k)
        return jnp.where(freqs == 0, 0.0, fw)
