"""data.py - Data structures used as arguments for model evaluations/predictions."""

import jax.numpy as jnp
from equinox import Module, field
from jaxtyping import Array


def convert_to_flat_array(array: Array) -> Array:
    return jnp.asarray(array).flatten()


class SpatialDataGeneric(Module):
    x: Array = field(converter=convert_to_flat_array)
    y: Array = field(converter=convert_to_flat_array)
    idx: Array = field(converter=convert_to_flat_array)


class SpatialDataLVM(Module):
    x: Array = field(converter=convert_to_flat_array)
    y: Array = field(converter=convert_to_flat_array)
    idx: Array = field(converter=convert_to_flat_array)
    tile_idx: Array = field(converter=convert_to_flat_array)
    ifu_idx: Array = field(converter=convert_to_flat_array)


# TODO: This is bad. We want users to be able to write their own SpatialDataFoo class, but with the current setup the typing doesn't strictly allow this. Really this should be a Protocol or subclassing sitation. Not worth refactoring right now.
SpatialData = SpatialDataGeneric | SpatialDataLVM
