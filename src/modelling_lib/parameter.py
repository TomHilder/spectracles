import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array


class Param(Module):
    """A varied parameter that is trainable. This is just for the purpose of internal tree surgery required when fitting the model."""

    value: Array


class Parameter:
    """Factory class for instantiating parameters and avoiding the user screwing it up."""

    def __new__(
        self, dims: int | tuple = None, initial: Array = None, fixed: bool = False
    ) -> Param | Array:
        # If user provided nothing assume scalar parameter
        if dims is None and initial is None:
            shape = (1,)
        # If user provided initial that doesn't match the shape
        elif initial is not None and dims is not None and dims != initial.shape:
            raise ValueError("The shape of initial must match dims.")
        else:
            shape = dims

        # Return either zeros
        if initial is None:
            param_vals = jnp.zeros(shape)
        # Or a copy of their initial values
        else:
            param_vals = self.to_float_array(initial).copy()

        return Param(param_vals) if not fixed else param_vals

    @staticmethod
    def to_float_array(x):
        if jnp.isscalar(x):  # catches int, float, etc.
            return jnp.array([float(x)])
        return jnp.asarray(x, dtype=float)
