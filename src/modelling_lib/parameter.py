import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array


class Parameter(Module):
    """A parameter in a statistical model, which may be fixed or free."""

    value: Array
    fixed: bool

    def __init__(self, dims: int | tuple = None, initial: Array = None, fixed: bool = False):
        if dims is None and initial is None:
            shape = (1,)
        elif initial is not None and dims is not None and dims != initial.shape:
            raise ValueError("The shape of initial must match dims.")
        else:
            shape = dims

        if initial is None:
            param_vals = jnp.zeros(shape)
        else:
            param_vals = self._to_float_array(initial).copy()

        self.value = param_vals
        self.fixed = fixed

    @staticmethod
    def _to_float_array(x) -> Array:
        if jnp.isscalar(x):
            return jnp.array([float(x)])
        return jnp.asarray(x, dtype=float)
