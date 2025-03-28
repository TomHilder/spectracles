from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from .spatial import SpatialData, SpatialModel


class SpectralSpatialModel(eqx.Module):
    @abstractmethod
    def __call__(self, λ: jax.Array, data: SpatialData):
        pass


class Constant(SpectralSpatialModel):
    # Model parameters
    const: SpatialModel

    def __init__(self, const: SpatialModel):
        self.const = const

    def __call__(self, λ: jax.Array, spatial_data: SpatialData):
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

    def __call__(self, λ: jax.Array, spatial_data: SpatialData):
        A_norm = self.A(spatial_data) / (self.σ(spatial_data) * jnp.sqrt(2 * jnp.pi))
        return A_norm * jnp.exp(-0.5 * ((λ - self.λ0(spatial_data)) / self.σ(spatial_data)) ** 2)


# class Combined(eqx.Module):
#     """A combined model of a constant and a Gaussian line."""

#     # Model components
#     gaussian: Gaussian
#     constant: Constant

#     def __init__(self, const, A, λ0, σ):
#         self.gaussian = Gaussian(A, λ0, σ)
#         self.constant = Constant(const)

#     def __call__(self, λ, x, y):
#         return self.constant(λ, x, y) + self.gaussian(λ, x, y)
