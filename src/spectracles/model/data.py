"""data.py - Data structures used as arguments for model evaluations/predictions."""

import jax.numpy as jnp
from equinox import AbstractVar, Module, field
from jaxtyping import Array


def convert_to_flat_array(array: Array) -> Array:
    return jnp.asarray(array).flatten()


class SpatialData(Module):
    """Base class for the spatial data that spatial models are evaluated at.

    Subclasses must provide `x`, `y` (sky coordinates in model units) and `idx`
    (each point's index, e.g. its spaxel), all flat arrays. Declare them with
    `field(converter=convert_to_flat_array)` to flatten inputs, and add any
    other per-point fields a model needs:

        class MySpatialData(SpatialData):
            x: Array = field(converter=convert_to_flat_array)
            y: Array = field(converter=convert_to_flat_array)
            idx: Array = field(converter=convert_to_flat_array)
            exposure_idx: Array = field(converter=convert_to_flat_array)
    """

    x: AbstractVar[Array]
    y: AbstractVar[Array]
    idx: AbstractVar[Array]

    def __check_init__(self):
        shapes = {name: jnp.shape(getattr(self, name)) for name in ("x", "y", "idx")}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"x, y and idx must have the same length, got shapes {shapes}.")


class SpatialDataGeneric(SpatialData):
    x: Array = field(converter=convert_to_flat_array)
    y: Array = field(converter=convert_to_flat_array)
    idx: Array = field(converter=convert_to_flat_array)


class SpatialDataLVM(SpatialData):
    x: Array = field(converter=convert_to_flat_array)
    y: Array = field(converter=convert_to_flat_array)
    idx: Array = field(converter=convert_to_flat_array)
    tile_idx: Array = field(converter=convert_to_flat_array)
    ifu_idx: Array = field(converter=convert_to_flat_array)
    fib_idx: Array = field(converter=convert_to_flat_array)
