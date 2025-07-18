"""parameter.py - Parameter objects for use with the modelling framework."""

from typing import Callable

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array


class BoundsError(Exception):
    pass


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
    forward_transform: Callable[[Array], Array]
    backward_transform: Callable[[Array], Array]

    def __init__(
        self,
        dims: int | tuple | None = None,
        initial: Array | None = None,
        fixed: bool = False,
        lower: float | None = None,
        upper: float | None = None,
        log: bool = False,
    ):
        self.fix = fixed

        # If log parameterization is requested
        if log:
            if lower is not None or upper is not None:
                raise ValueError(
                    "Cannot specify bounds (lower/upper) when log=True. "
                    "Log parameterization already ensures positivity."
                )
            self.forward_transform = lambda x: log_bounded(x)
            self.backward_transform = lambda x: log_bounded_inv(x)
        # If only lower bound
        elif lower is not None and upper is None:
            self.forward_transform = lambda x: l_bounded(x, lower)
            self.backward_transform = lambda x: l_bounded_inv(x, lower)
        # If only upper bound
        elif lower is None and upper is not None:
            self.forward_transform = lambda x: u_bounded(x, upper)
            self.backward_transform = lambda x: u_bounded_inv(x, upper)
        # If both bounds
        elif lower is not None and upper is not None:
            self.forward_transform = lambda x: lu_bounded(x, lower, upper)
            self.backward_transform = lambda x: lu_bounded_inv(x, lower, upper)
        # If no bounds
        else:
            raise ValueError(
                "Either lower or upper bound must be provided, or both, or log=True. "
                "For no bounds, use Parameter."
            )

        # Initialise the unconstrained value within bounds if not provided
        init_none_flag: bool = False
        if initial is None:
            init_none_flag = True
            # For log parameterization, default to exp(0) = 1 instead of 0
            if log:
                initial = 1.0

        # Try to initialise
        try:
            self.unconstrained_val = self.backward_transform(_array_from_user(dims, initial))
        # Initial is outside bounds either
        except BoundsError as e:
            # Because we auto-initialised with zeros which is outside the bounds
            if init_none_flag:
                if log:
                    raise BoundsError(
                        "Attempted to auto-initialise log-parameterized ConstrainedParameter "
                        "but encountered an error. Please provide a manual positive initialisation."
                    )
                else:
                    raise BoundsError(
                        "Attempted to auto-initialise ConstrainedParameter with zeros by default, "
                        "but this lies outside provided bounds. Please provide a manual "
                        "intialisation inside the bounds instead."
                    )
            # Or because the user asked for an initial value outside the bounds
            else:
                raise e

    @property
    def val(self) -> Array:
        return self.forward_transform(self.unconstrained_val)


AnyParameter = Parameter | ConstrainedParameter


def init_parameter(parameter: Parameter | None, **kwargs) -> Parameter:
    return Parameter(**kwargs) if parameter is None else parameter


def is_parameter(x):
    return isinstance(x, AnyParameter)


def is_trainable(x):
    return is_parameter(x) and not x.fix


def is_constrained(x):
    if not is_parameter(x):
        raise ValueError("is_constrained only works on AnyParameter.")
    return isinstance(x, ConstrainedParameter)


# ==== Transformations for constrained parameters ====


def softplus(x: Array) -> Array:
    return jnp.maximum(x, 0) + jnp.log1p(jnp.exp(-jnp.abs(x)))


def softplus_inv(f: Array) -> Array:
    if jnp.any(f <= 0):
        raise ValueError(
            "Specified initial value is likely outside bounds, or you've hit under/overflow. The inverse transformations used to initialise constrained parameters can be unstable for large values."
        )
    return jnp.where(f > 20, f + jnp.log1p(-jnp.exp(-f)), jnp.log(jnp.expm1(f)))


def softplus_frac_inv(x: Array, eps: float = 1e-16) -> Array:
    x_clamped = jnp.clip(x, eps, 1.0 - eps)
    t = x_clamped / (1.0 - x_clamped)
    expm1_t = jnp.expm1(t)
    return jnp.log(expm1_t)


def l_bounded(x: Array, lower: float) -> Array:
    return lower + softplus(x)


def l_bounded_inv(f: Array, lower: float) -> Array:
    if jnp.any(f < lower):
        raise BoundsError("Initial value lies below lower bound.")
    return softplus_inv(f - lower)


def u_bounded(x: Array, upper: float) -> Array:
    return upper - softplus(-x)


def u_bounded_inv(f: Array, upper: float) -> Array:
    if jnp.any(f > upper):
        raise BoundsError("Initial value lies above upper bound.")
    return -softplus_inv(upper - f)


def lu_bounded(x: Array, lower: float, upper: float) -> Array:
    s = softplus(x)
    return lower + (upper - lower) * s / (1.0 + s)


def lu_bounded_inv(f: Array, lower: float, upper: float) -> Array:
    if jnp.any(f < lower):
        raise BoundsError("Initial value lies below lower bound.")
    elif jnp.any(f > upper):
        raise BoundsError("Initial value lies above upper bound.")
    return softplus_frac_inv((f - lower) / (upper - lower))


# ==== Log parameterization functions ====


def log_bounded(x: Array) -> Array:
    """Transform unconstrained parameter to positive parameter via exp."""
    return jnp.exp(x)


def log_bounded_inv(f: Array) -> Array:
    """Transform positive parameter to unconstrained parameter via log."""
    if jnp.any(f <= 0):
        raise BoundsError("Initial value must be positive for log parameterization.")
    return jnp.log(f)
