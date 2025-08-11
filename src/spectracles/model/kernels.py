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
    # All kernels should have a length scale and variance
    length_scale: Parameter
    variance: Parameter

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
