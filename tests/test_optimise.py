"""test_optimise.py tests for the modelling_lib.optimise.optimiser_frame module."""

import equinox as eqx
import jax.numpy as jnp
import optax  # type: ignore[import]
import pytest
from jax.tree_util import tree_map
from modelling_lib.model.parameter import AnyParameter, Parameter
from modelling_lib.model.share_module import ShareModule, build_model
from modelling_lib.optimise.optimiser_frame import OptimiserFrame, get_opt_filter_spec

from .test_models import SharedLeafModel, SimpleModel


class TestGetFilterSpec:
    def test_basic_model(self):
        # Test with a simple model
        model = build_model(SimpleModel, value=1.0)
        filter_spec = get_opt_filter_spec(model)

        # Should be a function that returns True for trainable parameters
        assert callable(filter_spec)

    def test_model_with_fixed_param(self):
        # Create a model with one fixed parameter
        class ModelWithFixedParam(eqx.Module):
            free_param: Parameter
            fixed_param: Parameter

            def __init__(self):
                self.free_param = Parameter(initial=1.0, fixed=False)
                self.fixed_param = Parameter(initial=2.0, fixed=True)

            def __call__(self, x):
                return self.free_param.val * x + self.fixed_param.val

        model = build_model(ModelWithFixedParam)

        # Apply filter spec to model
        filter_spec = get_opt_filter_spec(model)

        # Use the filter_spec with eqx.filter to extract trainable parameters
        trainable_model, fixed_model = eqx.partition(model, filter_spec)

        # Check that free_param is included but fixed_param is not
        assert trainable_model.free_param.val is not None
        assert trainable_model.fixed_param.val is None
        # Check converse for fixed_model
        assert fixed_model.free_param.val is None
        assert fixed_model.fixed_param.val is not None


class TestOptimiserFrame:
    def test_initialization(self):
        # Create a model and loss function
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        optimiser = optax.sgd(learning_rate=0.1)

        # Create optimiser frame
        frame = OptimiserFrame(model, loss_fn, optimiser)

        # Check attributes
        assert frame.model is model
        assert frame.loss_fn is loss_fn
        assert frame.optimiser is optimiser
        assert frame.loss_history == []

    def test_initialization_with_non_sharemodule(self):
        # Create a model that's not wrapped with ShareModule
        model = SimpleModel(value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        optimiser = optax.sgd(learning_rate=0.1)

        # Should raise an error
        with pytest.raises(ValueError):
            OptimiserFrame(model, loss_fn, optimiser)

    def test_initialization_with_locked_model(self):
        # Create a locked model
        model = SimpleModel(value=1.0)
        locked_model = ShareModule(model, locked=True)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        optimiser = optax.sgd(learning_rate=0.1)

        # Should raise an error
        with pytest.raises(ValueError):
            OptimiserFrame(locked_model, loss_fn, optimiser)

    def test_run_simple_optimization(self):
        # Create a model and loss function
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        # Create data where optimal parameter is 2.0
        x = jnp.array([1.0, 2.0, 3.0])
        y = 2.0 * x

        # Create optimiser and frame
        optimiser = optax.sgd(learning_rate=0.1)
        frame = OptimiserFrame(model, loss_fn, optimiser)

        # Run optimization
        optimized_model = frame.run(n_steps=100, x=x, y=y)

        # Check that parameter has moved toward 2.0
        assert jnp.allclose(optimized_model.param.val, 2.0, rtol=1e-1)

        # Check that loss history is recorded
        assert len(frame.loss_history) == 100
        assert frame.loss_history[0] > frame.loss_history[-1]  # Loss should decrease

    def test_optimization_with_shared_params(self):
        # Create a model with shared parameters
        model = build_model(SharedLeafModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        # Create data where optimal a+b should be 1.0
        x = jnp.array([1.0, 2.0, 3.0])
        y = x  # Optimal is a=0.5, b=0.5

        # Create optimiser and frame
        optimiser = optax.adam(learning_rate=0.1)
        frame = OptimiserFrame(model, loss_fn, optimiser)

        # Run optimization
        optimized_model = frame.run(n_steps=100, x=x, y=y)

        # Check that parameters have been optimized
        # Since a and b are shared, they should be equal
        a_val = optimized_model.a.val
        assert jnp.allclose(a_val, 0.5, rtol=1e-1)

        # Verify that the prediction is close to y
        pred = optimized_model(x)
        assert jnp.allclose(pred, y, rtol=1e-1)

    def test_optimization_with_fixed_params(self):
        # Create a model with fixed parameters
        class ModelWithFixedParam(eqx.Module):
            free_param: Parameter
            fixed_param: Parameter

            def __init__(self):
                self.free_param = Parameter(initial=1.0, fixed=False)
                self.fixed_param = Parameter(initial=2.0, fixed=True)

            def __call__(self, x):
                return self.free_param.val * x + self.fixed_param.val

        model = build_model(ModelWithFixedParam)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        # Create data where optimal parameters are free_param=2, fixed_param=2
        x = jnp.array([1.0, 2.0, 3.0])
        y = 2.0 * x + 2.0

        # Create optimiser and frame
        optimiser = optax.sgd(learning_rate=0.1)
        frame = OptimiserFrame(model, loss_fn, optimiser)

        # Run optimization
        optimized_model = frame.run(n_steps=100, x=x, y=y)

        # Check that free_param has moved toward 2.0
        assert jnp.allclose(optimized_model.free_param.val, 2.0, rtol=1e-1)

        # Check that fixed_param hasn't changed
        assert jnp.allclose(optimized_model.fixed_param.val, 2.0)

    def test_verify_loss_fn(self):
        # Create a model
        model = build_model(SimpleModel, value=1.0)

        # Bad loss function that raises an exception
        def bad_loss_fn(model, x):
            raise ValueError("Bad loss function")

        optimiser = optax.sgd(learning_rate=0.1)
        frame = OptimiserFrame(model, bad_loss_fn, optimiser)

        # Should raise an error when running
        with pytest.raises(ValueError):
            frame._verify_loss_fn(jnp.array([1.0]))

        # Bad loss function that returns NaN
        def nan_loss_fn(model, x):
            return jnp.nan

        frame = OptimiserFrame(model, nan_loss_fn, optimiser)

        # Should raise an error when running
        with pytest.raises(ValueError):
            frame._verify_loss_fn(jnp.array([1.0]))

    def test_custom_filter_spec(self):
        # Create a model with multiple parameters
        class MultiParamModel(eqx.Module):
            param1: Parameter
            param2: Parameter

            def __init__(self):
                self.param1 = Parameter(initial=1.0)
                self.param2 = Parameter(initial=0.1)

            def __call__(self, x):
                return self.param1.val * x + self.param2.val

        model = build_model(MultiParamModel)

        # Custom filter spec that only trains param1
        def get_custom_filter_spec(model):
            def is_parameter(x):
                return isinstance(x, AnyParameter)

            def is_param1(x):
                return isinstance(x, Parameter) and id(x) == id(model.param1)

            return tree_map(is_param1, model, is_leaf=is_parameter)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        # Create data where optimal parameters are param1=3, param2=0
        x = jnp.linspace(0, 1, 100)
        y = 3.0 * x

        # Create optimiser and frame with custom filter spec
        optimiser = optax.adam(learning_rate=0.01)
        frame = OptimiserFrame(
            model, loss_fn, optimiser, get_filter_spec_fn=get_custom_filter_spec
        )

        # Run optimization
        optimized_model = frame.run(n_steps=1000, x=x, y=y)
        print(optimized_model.param1.val, optimized_model.param2.val)

        # Check that param1 has moved toward 3.0
        assert jnp.allclose(optimized_model.param1.val, 3.0, rtol=1e-1)

        # Check that param2 hasn't changed at all
        assert optimized_model.param2.val[0] == 0.1
