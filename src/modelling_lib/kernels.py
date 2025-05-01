from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array

from .parameter import Parameter


def normalise_fw(fw: Array) -> Array:
    """
    Normalise the feature weights. Includes a factor of sqrt(2) which accounts for the
    halving in total power incurred by enforcing the Fourier coefficients to be real via
    conjugate symmetry.
    """
    power = jnp.sum(jnp.abs(fw) ** 2)
    return jnp.sqrt(2) * fw / jnp.sqrt(power)


def matern_kernel_fw_nd(freqs: Array, length: float, var: float, nu: float, n: int) -> Array:
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

    def __init__(self, length_scale: float, variance: float):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale, self.variance, nu=0.5, n=2)


class Matern32(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: float, variance: float):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale, self.variance, nu=1.5, n=2)


class Matern52(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: float, variance: float):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        return matern_kernel_fw_nd(freqs, self.length_scale, self.variance, nu=2.5, n=2)


class SquaredExponential(Kernel):
    length_scale: Parameter
    variance: Parameter

    def __init__(self, length_scale: float, variance: float):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        fw = jnp.sqrt(jnp.exp(-0.5 * freqs**2 * self.length_scale**2))
        return jnp.sqrt(self.variance) * normalise_fw(fw)
