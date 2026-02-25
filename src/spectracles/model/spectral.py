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


class WindowConstant(SpectralSpatialModel):
    # Model parameters
    const: SpatialModel
    λ_min: float
    λ_max: float

    def __init__(self, const: SpatialModel, λ_min: float, λ_max: float):
        self.const = const
        self.λ_min = λ_min
        self.λ_max = λ_max

    def __call__(self, λ: Array, spatial_data: SpatialData):
        window_cond = jnp.logical_and(λ >= self.λ_min, λ <= self.λ_max)
        window = jnp.where(window_cond, 1.0, 0.0)
        return self.const(spatial_data) * window


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
