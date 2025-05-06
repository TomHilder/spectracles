import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array


def _to_float_array(x) -> Array:
    if jnp.isscalar(x):
        return jnp.array([float(x)])
    return jnp.asarray(x, dtype=float)


def _array_from_user(
    dims: int | tuple | None = None,
    initial: float | Array | None = None,
) -> Array:
    if initial is not None:
        param_vals = _to_float_array(initial).copy()
        if dims is not None:
            expected_shape = (dims,) if isinstance(dims, int) else dims
            if param_vals.shape != expected_shape:
                raise ValueError("The shape of initial must match dims.")
    else:
        if dims is None:
            shape = (1,)
        else:
            shape = (dims,) if isinstance(dims, int) else dims
        param_vals = jnp.zeros(shape)

    return param_vals


class Parameter(Module):
    """A parameter in a statistical model, which may be fixed or free."""

    val: Array
    fix: bool

    def __init__(
        self,
        dims: int | tuple | None = None,
        initial: Array | None = None,
        fixed: bool = False,
    ):
        self.fix = fixed
        self.val = _array_from_user(dims, initial)


class ConstrainedParameter(Module):
    """A parameter in a statistical model, which may be fixed or free, with lower and/or upper bounds. Can be an array, but all entries must share the same bounds."""

    unconstrained_val: Array
    fix: bool
    forward_transform: callable
    backward_transform: callable

    def __init__(
        self,
        dims: int | tuple | None = None,
        initial: Array | None = None,
        fixed: bool = False,
        lower: float | None = None,
        upper: float | None = None,
    ):
        self.fix = fixed

        # If only lower bound
        if lower is not None and upper is None:
            self.forward_transform = lambda x: l_bounded(x, lower)
            self.backward_transform = ...
        # If only upper bound
        elif lower is None and upper is not None:
            self.forward_transform = lambda x: u_bounded(x, upper)
            self.backward_transform = ...
        # If both bounds
        elif lower is not None and upper is not None:
            self.forward_transform = lambda x: lu_bounded(x, lower, upper)
            self.backward_transform = ...
        # If no bounds
        else:
            raise ValueError(
                "Either lower or upper bound must be provided, or both. For no bounds, use Parameter."
            )

        # Initialise the unconstrained value
        if initial is None:
            # We need to set the initial value to be within the bounds
            if lower is not None and upper is not None and (lower > 0 or upper < 0):
                initial = (lower + upper) / 2
            elif lower is not None and lower > 0:
                initial = lower + 1
            elif upper is not None and upper < 0:
                initial = upper - 1
        self.unconstrained_val = self.backward_transform(_array_from_user(dims, initial))

    @property
    def val(self) -> Array:
        return self.forward_transform(self.unconstrained_val)


def init_parameter(parameter: Parameter | None, **kwargs) -> Parameter:
    return Parameter(**kwargs) if parameter is None else parameter


# ==== Transformations for constrained parameters ====


def softplus(x: Array) -> Array:
    return jnp.maximum(x, 0) + jnp.log1p(jnp.exp(-jnp.abs(x)))


def softplus_inv(f: Array) -> Array:
    if jnp.any(f <= 0):
        raise ValueError("Specified initial value is likely outside bounds.")
    return jnp.where(f > 20, f + jnp.log1p(-jnp.exp(-f)), jnp.log(jnp.expm1(f)))


def l_bounded(x: Array, lower: float) -> Array:
    return lower + softplus(x)


def l_bounded_inv(f: Array, lower: float) -> Array:
    return softplus_inv(f - lower)


def u_bounded(x: Array, upper: float) -> Array:
    return upper - softplus(-x)


def u_bounded_inv(f: Array, upper: float) -> Array:
    return softplus_inv(upper - f)


def lu_bounded(x: Array, lower: float, upper: float) -> Array:
    s = softplus(x)
    return lower + (upper - lower) * s / (1.0 + s)
