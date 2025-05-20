"""test_models.py - example models used for testing purposes, especially for testing that sharing is handled correctly.

These need to be in a separate file from the tests beacuse the serialisation we use to save and load models relies on dill, which requires it."""

import equinox as eqx
import jax.numpy as jnp
from equinox import Module
from modelling_lib.model.data import SpatialData
from modelling_lib.model.parameter import Parameter


class SimpleModel(Module):
    """Simple model with a single parameter."""

    param: Parameter

    def __init__(self, value=1.0):
        self.param = Parameter(initial=value)

    def __call__(self, x):
        return self.param.val * x


class SharedLeafModel(Module):
    """Model with shared leaves between two parameters."""

    a: Parameter
    b: Parameter

    def __init__(self, value=1.0):
        # Create a single parameter to be shared
        shared_param = Parameter(initial=value)
        self.a = shared_param
        self.b = shared_param  # Share the same parameter

    def __call__(self, x):
        return self.a.val * x + self.b.val * x


class NestedModel(Module):
    """Model with nested parameters."""

    inner1: SimpleModel
    inner2: SimpleModel
    shared: Parameter

    def __init__(self, value=1.0):
        self.inner1 = SimpleModel(value)
        self.inner2 = SimpleModel(value)
        self.shared = Parameter(initial=2.0)

    def __call__(self, x):
        return self.inner1(x) + self.inner2(x) + self.shared.val


class ComplexSharedModel(Module):
    """Complex model with multiple types of sharing."""

    inner1: SimpleModel
    inner2: SimpleModel
    inner3: NestedModel
    param: Parameter

    def __init__(self, value=1.0):
        # Create a single parameter to be shared
        shared_param = Parameter(initial=value)

        self.inner1 = SimpleModel(value=value)  # Initialize with some value
        self.inner1 = eqx.tree_at(lambda m: m.param, self.inner1, shared_param)

        self.inner2 = SimpleModel(value=value)
        self.inner2 = eqx.tree_at(lambda m: m.param, self.inner2, shared_param)

        _inner3 = NestedModel(value=value)  # Initial placeholder
        _inner3_inner1 = eqx.tree_at(lambda m: m.param, _inner3.inner1, shared_param)
        self.inner3 = eqx.tree_at(lambda m: m.inner1, _inner3, _inner3_inner1)

        self.param = shared_param

    def __call__(self, x):
        return self.inner1(x) + self.inner2(x) + self.inner3(x) + self.param.val


class SpatialDummyModel(Module):
    """Dummy spatial model class that works with SpatialData."""

    value: Parameter

    def __init__(self, value=1.0):
        self.value = Parameter(initial=value)

    def __call__(self, data: SpatialData):
        return self.value.val * jnp.ones_like(data.x)
