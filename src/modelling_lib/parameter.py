import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array


class Parameter(Module):
    """A parameter in a statistical model, which may be fixed or free."""

    val: Array
    fix: bool

    def __init__(
        self, dims: int | tuple | None = None, initial: Array | None = None, fixed: bool = False
    ):
        # If nothing provided, default to a scalar
        if dims is None and initial is None:
            shape = (1,)
        # If initial and dims are provided, check they match
        elif initial is not None and dims is not None and dims != initial.shape:
            raise ValueError("The shape of initial must match dims.")
        # If dims is provided, use it
        elif isinstance(dims, tuple):
            shape = dims
        elif isinstance(dims, int):
            shape = (dims,)
        # If initial is provided, use it, otherwise just use zeros
        if initial is None:
            param_vals = jnp.zeros(shape)
        else:
            param_vals = self._to_float_array(initial).copy()
        # Save
        self.val = param_vals
        self.fix = fixed

    @staticmethod
    def _to_float_array(x) -> Array:
        if jnp.isscalar(x):
            return jnp.array([float(x)])
        return jnp.asarray(x, dtype=float)


"""
It would be nice to do something like the below, although I can't get it to work. Potentially using paramax might be easier. Anyway, the point is that it avoids when writing the model classes having to the param.val to access the array of some Parameter object called param, when calculating things in the model with that parameter. The current way means that you definitely need to know what is and is not a Parameter when writing the model, which to be fair really isn't too bad. Although with the type of change below, it would make __call__ not work except inside jax transformations like jit, vmap, etc. which is kind of shitty.
"""
# @register_pytree_node_class
# class Parameter(Module):
#     """A parameter in a statistical model, auto-unwrapped in JAX transforms."""

#     value: Array
#     fixed: bool

#     def __init__(self, dims=None, initial=None, fixed=False):
#         if dims is None and initial is None:
#             shape = (1,)
#         elif initial is not None and dims is not None and dims != initial.shape:
#             raise ValueError("Shape mismatch")
#         else:
#             shape = dims

#         self.value = jnp.zeros(shape) if initial is None else jnp.asarray(initial, float).copy()
#         self.fixed = fixed

#     def __repr__(self):
#         return f"Parameter(value={self.value}, fixed={self.fixed})"

#     # —– JAX leaf conversion —–
#     def __jax_array__(self) -> Array:
#         # Ensures all JAX primitives see `self` as `self.value`.
#         return self.value

#     # —– PyTree registration —–
#     def tree_flatten(self):
#         # child = the array; metadata = fixed flag
#         return (self.value,), self.fixed

#     @classmethod
#     def tree_unflatten(cls, fixed, children):
#         return cls(initial=children[0], fixed=fixed)
