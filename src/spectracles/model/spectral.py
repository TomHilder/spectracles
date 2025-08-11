"""spectral.py - Spectral models with components that are spatial models."""

from abc import abstractmethod

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array

from spectracles.model.data import SpatialData
from spectracles.model.spatial import SpatialModel


class SpectralSpatialModel(Module):
    @abstractmethod
    def __call__(self, λ: Array, data: SpatialData):
        pass


class Constant(SpectralSpatialModel):
    # Model parameters
    const: SpatialModel

    def __init__(self, const: SpatialModel):
        self.const = const

    def __call__(self, λ: Array, spatial_data: SpatialData):
        return self.const(spatial_data) * jnp.ones_like(λ)


class Gaussian(SpectralSpatialModel):
    # Model parameters
    A: SpatialModel
    λ0: SpatialModel
    σ: SpatialModel

    def __init__(self, A: SpatialModel, λ0: SpatialModel, σ: SpatialModel):
        self.A = A
        self.λ0 = λ0
        self.σ = σ

    def __call__(self, λ: Array, spatial_data: SpatialData):
        A_norm = self.A(spatial_data) / (self.σ(spatial_data) * jnp.sqrt(2 * jnp.pi))
        return A_norm * jnp.exp(-0.5 * ((λ - self.λ0(spatial_data)) / self.σ(spatial_data)) ** 2)
