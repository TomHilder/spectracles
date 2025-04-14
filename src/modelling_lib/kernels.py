from abc import abstractmethod

import jax.numpy as jnp
from equinox import AbstractVar, Module
from jaxtyping import Array


class Kernel(Module):
    # All kernels should have a length scale and variance
    length_scale: AbstractVar[float]
    variance: AbstractVar[float]

    @abstractmethod
    def feature_weights(self, freqs: Array) -> Array:
        pass


class Matern32(Kernel):
    length_scale: float
    variance: float

    def __init__(self, length_scale: float, variance: float):
        self.length_scale = length_scale
        self.variance = variance

    def feature_weights(self, freqs: Array) -> Array:
        # return self.variance / (1 + (freqs * self.length_scale) ** 2)
        return jnp.sqrt(self.length_scale) * self.variance / (1 + (freqs * self.length_scale) ** 2)
