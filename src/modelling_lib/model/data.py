"""data.py - Data structures used as arguments for model evaluations/predictions."""

import jax.numpy as jnp
from equinox import Module, field
from jaxtyping import Array


def convert_to_flat_array(array: Array) -> Array:
    return jnp.asarray(array).flatten()


class SpatialData(Module):
    x: Array = field(converter=convert_to_flat_array)
    y: Array = field(converter=convert_to_flat_array)
    indices: Array = field(converter=convert_to_flat_array)
