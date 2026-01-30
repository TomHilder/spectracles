"""test_opt_schedule.py - tests for the spectracles.optimise.opt_schedule module."""

import jax.numpy as jnp
import optax  # type: ignore[import]
import pytest
from spectracles.model.share_module import build_model
from spectracles.optimise.opt_schedule import (
    OptimiserSchedule,
    Phase,
    PhaseConfig,
)

from .test_models import SimpleModel, SharedLeafModel


class TestPhaseConfig:
    def test_basic_initialization(self):
        # Test basic PhaseConfig creation
        config = PhaseConfig(
            n_steps=100,
            optimiser=optax.sgd(learning_rate=0.1),
        )
        assert config.n_steps == 100
        assert config.Δloss_criterion == 1e2  # default
        assert config.fix_status_updates == {}
        assert config.param_val_updates == {}

    def test_with_fix_status_updates(self):
        # Test PhaseConfig with fix_status_updates
        config = PhaseConfig(
            n_steps=50,
            optimiser=optax.adam(learning_rate=0.01),
            fix_status_updates={"param": True},
        )
        assert config.fix_status_updates == {"param": True}

    def test_with_param_val_updates(self):
        # Test PhaseConfig with param_val_updates
        config = PhaseConfig(
            n_steps=50,
            optimiser=optax.adam(learning_rate=0.01),
            param_val_updates={"param": jnp.array([1.0])},
        )
        assert "param" in config.param_val_updates

    def test_invalid_n_steps_type(self):
        # n_steps must be int
        with pytest.raises(TypeError, match="n_steps must be int"):
            PhaseConfig(
                n_steps=100.0,  # float instead of int
                optimiser=optax.sgd(learning_rate=0.1),
            )

    def test_invalid_fix_status_type(self):
        # fix_status_updates values must be bool
        with pytest.raises(TypeError, match="fix_status_updates must be bool"):
            PhaseConfig(
                n_steps=100,
                optimiser=optax.sgd(learning_rate=0.1),
                fix_status_updates={"param": "true"},  # string instead of bool
            )

    def test_invalid_param_val_type(self):
        # param_val_updates values must be jax Arrays
        with pytest.raises(TypeError, match="param_val_updates must be jax Arrays"):
            PhaseConfig(
                n_steps=100,
                optimiser=optax.sgd(learning_rate=0.1),
                param_val_updates={"param": [1.0]},  # list instead of Array
            )

    def test_custom_delta_loss(self):
        # Test custom Δloss_criterion
        config = PhaseConfig(
            n_steps=100,
            optimiser=optax.sgd(learning_rate=0.1),
            Δloss_criterion=1e-4,
        )
        assert config.Δloss_criterion == 1e-4


class TestPhase:
    def test_phase_creation(self):
        # Create a model and phase
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        config = PhaseConfig(
            n_steps=10,
            optimiser=optax.sgd(learning_rate=0.1),
        )

        from spectracles.optimise.opt_frame import OptimiserFrame

        frame = OptimiserFrame(model, loss_fn, config.optimiser)
        phase = Phase(config=config, frame=frame)

        assert phase.config is config
        assert phase.frame is frame

    def test_phase_validation_with_valid_updates(self):
        # Phase should validate fix_status_updates
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        config = PhaseConfig(
            n_steps=10,
            optimiser=optax.sgd(learning_rate=0.1),
            fix_status_updates={"param": True},
        )

        from spectracles.optimise.opt_frame import OptimiserFrame

        frame = OptimiserFrame(model, loss_fn, config.optimiser)

        # Should not raise
        phase = Phase(config=config, frame=frame)
        assert phase is not None


class TestOptimiserSchedule:
    def test_initialization(self):
        # Test basic OptimiserSchedule creation
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = OptimiserSchedule(model, loss_fn, configs)

        assert len(schedule.phases) == 2
        assert len(schedule.model_history) == 1
        assert schedule.model_history[0] is model

    def test_run_all(self):
        # Test running all phases
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        y = 2.0 * x

        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = OptimiserSchedule(model, loss_fn, configs)
        schedule.run_all(x=x, y=y)

        # Should have model history with initial + 2 phases
        assert len(schedule.model_history) == 3

        # Loss should have decreased
        assert schedule.loss_history[0] > schedule.loss_history[-1]

    def test_run_phase(self):
        # Test running a single phase
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        y = 2.0 * x

        configs = [
            PhaseConfig(n_steps=20, optimiser=optax.sgd(learning_rate=0.1)),
        ]

        schedule = OptimiserSchedule(model, loss_fn, configs)
        schedule.run_phase(schedule.phases[0], x=x, y=y)

        # Should have 2 models in history
        assert len(schedule.model_history) == 2

    def test_loss_histories(self):
        # Test getting loss histories from phases
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        y = 2.0 * x

        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=15, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = OptimiserSchedule(model, loss_fn, configs)
        schedule.run_all(x=x, y=y)

        # Check individual histories
        histories = schedule.loss_histories
        assert len(histories) == 2
        assert len(histories[0]) == 10
        assert len(histories[1]) == 15

        # Check combined history
        assert len(schedule.loss_history) == 25

    def test_with_fix_status_updates(self):
        # Test phase that fixes a parameter
        model = build_model(SimpleModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        y = 2.0 * x

        # First phase: optimize, second phase: fix param
        configs = [
            PhaseConfig(
                n_steps=10,
                optimiser=optax.sgd(learning_rate=0.1),
            ),
            PhaseConfig(
                n_steps=10,
                optimiser=optax.sgd(learning_rate=0.1),
                fix_status_updates={"param": True},
            ),
        ]

        schedule = OptimiserSchedule(model, loss_fn, configs)
        schedule.run_all(x=x, y=y)

        # Model should have been updated
        assert len(schedule.model_history) == 3

    def test_with_shared_params(self):
        # Test schedule with shared parameters
        model = build_model(SharedLeafModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        y = x  # Optimal is a=0.5, b=0.5 (shared)

        configs = [
            PhaseConfig(n_steps=50, optimiser=optax.adam(learning_rate=0.1)),
        ]

        schedule = OptimiserSchedule(model, loss_fn, configs)
        schedule.run_all(x=x, y=y)

        # Final model should be close to optimal
        final_model = schedule.model_history[-1]
        locked = final_model.get_locked_model()
        assert jnp.allclose(locked.a.val, 0.5, rtol=1e-1)
