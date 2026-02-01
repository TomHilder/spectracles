"""test_opt_schedule.py - tests for the spectracles.optimise.opt_schedule module."""

import jax.numpy as jnp
import optax  # type: ignore[import]
import pytest
from spectracles.model.share_module import build_model
from spectracles.optimise.opt_schedule import (
    ManagedOptimiserSchedule,
    OptimiserSchedule,
    Phase,
    PhaseConfig,
    PhaseState,
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


class TestManagedOptimiserSchedule:
    """Tests for the ManagedOptimiserSchedule class with phase state management."""

    def setup_method(self):
        """Common setup for tests."""
        self.model = build_model(SimpleModel, value=1.0)
        self.x = jnp.array([1.0, 2.0, 3.0])
        self.y = 2.0 * self.x

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        self.loss_fn = loss_fn

    def test_initialization(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        assert len(schedule.phases) == 2
        assert len(schedule.model_history) == 1
        assert schedule.current_phase_index == 0
        assert all(state == PhaseState.PENDING for state in schedule.phase_states)

    def test_run_all(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        schedule.run_all(x=self.x, y=self.y)

        assert len(schedule.model_history) == 3
        assert schedule.is_complete()
        assert all(state == PhaseState.COMPLETED for state in schedule.phase_states)

    def test_run_next_phase(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        # Run first phase
        result = schedule.run_next_phase(x=self.x, y=self.y)
        assert result is True
        assert schedule.current_phase_index == 1
        assert schedule.phase_states[0] == PhaseState.COMPLETED
        assert schedule.phase_states[1] == PhaseState.PENDING

        # Run second phase
        result = schedule.run_next_phase(x=self.x, y=self.y)
        assert result is True
        assert schedule.is_complete()

        # No more phases
        result = schedule.run_next_phase(x=self.x, y=self.y)
        assert result is False

    def test_run_phase_by_index(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        # Run phase 0
        schedule.run_phase_by_index(0, x=self.x, y=self.y)
        assert schedule.phase_states[0] == PhaseState.COMPLETED

    def test_run_phase_out_of_order_raises(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        # Try to run phase 1 before phase 0
        with pytest.raises(RuntimeError, match="must complete phases in order"):
            schedule.run_phase_by_index(1, x=self.x, y=self.y)

    def test_run_completed_phase_raises(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        schedule.run_all(x=self.x, y=self.y)

        # Try to run completed phase again
        with pytest.raises(RuntimeError, match="already been completed"):
            schedule.run_phase_by_index(0, x=self.x, y=self.y)

    def test_run_phase_invalid_index_raises(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        with pytest.raises(ValueError, match="out of range"):
            schedule.run_phase_by_index(5, x=self.x, y=self.y)

        with pytest.raises(ValueError, match="out of range"):
            schedule.run_phase_by_index(-1, x=self.x, y=self.y)

    def test_skip_phase(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        # Skip first phase
        schedule.skip_phase(0)
        assert schedule.phase_states[0] == PhaseState.SKIPPED
        assert schedule.current_phase_index == 1

        # Run second phase
        schedule.run_next_phase(x=self.x, y=self.y)
        assert schedule.is_complete()

    def test_skip_phase_out_of_order_raises(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        # Try to skip phase 1 before phase 0
        with pytest.raises(RuntimeError, match="must process phases in order"):
            schedule.skip_phase(1)

    def test_skip_completed_phase_raises(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        schedule.run_all(x=self.x, y=self.y)

        with pytest.raises(RuntimeError, match="already been completed"):
            schedule.skip_phase(0)

    def test_reset(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        schedule.run_all(x=self.x, y=self.y)

        # Reset
        schedule.reset()

        assert len(schedule.model_history) == 1
        assert schedule.current_phase_index == 0
        assert all(state == PhaseState.PENDING for state in schedule.phase_states)
        assert not schedule.is_complete()

    def test_reset_from_phase(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.001)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        schedule.run_all(x=self.x, y=self.y)

        # Reset from phase 1
        schedule.reset_from_phase(1)

        assert schedule.current_phase_index == 1
        assert schedule.phase_states[0] == PhaseState.COMPLETED
        assert schedule.phase_states[1] == PhaseState.PENDING
        assert schedule.phase_states[2] == PhaseState.PENDING
        assert len(schedule.model_history) == 2  # initial + phase 0 result

    def test_get_phase_status(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        schedule.run_next_phase(x=self.x, y=self.y)

        status = schedule.get_phase_status()

        assert status["current_phase"] == 1
        assert status["total_phases"] == 2
        assert status["completed_phases"] == 1
        assert status["phase_states"] == ["completed", "pending"]
        assert status["is_complete"] is False

    def test_is_complete(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        assert not schedule.is_complete()

        schedule.run_all(x=self.x, y=self.y)
        assert schedule.is_complete()

    def test_get_next_phase_index(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        assert schedule.get_next_phase_index() == 0

        schedule.run_next_phase(x=self.x, y=self.y)
        assert schedule.get_next_phase_index() == 1

        schedule.run_next_phase(x=self.x, y=self.y)
        assert schedule.get_next_phase_index() is None

    def test_get_completed_phases(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        assert schedule.get_completed_phases() == []

        schedule.run_next_phase(x=self.x, y=self.y)
        assert schedule.get_completed_phases() == [0]

        schedule.run_next_phase(x=self.x, y=self.y)
        assert schedule.get_completed_phases() == [0, 1]

    def test_get_pending_phases(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        assert schedule.get_pending_phases() == [0, 1]

        schedule.run_next_phase(x=self.x, y=self.y)
        assert schedule.get_pending_phases() == [1]

    def test_loss_histories_only_completed(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=15, optimiser=optax.adam(learning_rate=0.01)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        # No completed phases yet
        assert schedule.loss_histories == []
        assert len(schedule.loss_history) == 0

        # Run first phase
        schedule.run_next_phase(x=self.x, y=self.y)
        assert len(schedule.loss_histories) == 1
        assert len(schedule.loss_histories[0]) == 10

        # Run second phase
        schedule.run_next_phase(x=self.x, y=self.y)
        assert len(schedule.loss_histories) == 2
        assert len(schedule.loss_history) == 25

    def test_run_phases_multiple(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.01)),
            PhaseConfig(n_steps=10, optimiser=optax.adam(learning_rate=0.001)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        # Run phases 0 and 1
        schedule.run_phases([0, 1], x=self.x, y=self.y)

        assert schedule.phase_states[0] == PhaseState.COMPLETED
        assert schedule.phase_states[1] == PhaseState.COMPLETED
        assert schedule.phase_states[2] == PhaseState.PENDING

    def test_run_phases_single_int(self):
        configs = [
            PhaseConfig(n_steps=10, optimiser=optax.sgd(learning_rate=0.1)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)

        # Run single phase with int (not list)
        schedule.run_phases(0, x=self.x, y=self.y)

        assert schedule.is_complete()

    def test_optimization_actually_works(self):
        """Verify the unsafe schedule actually optimizes the model correctly."""
        configs = [
            PhaseConfig(n_steps=50, optimiser=optax.adam(learning_rate=0.1)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        initial_loss = self.loss_fn(self.model.get_locked_model(), self.x, self.y)

        schedule.run_all(x=self.x, y=self.y)

        final_model = schedule.model_history[-1]
        final_loss = self.loss_fn(final_model.get_locked_model(), self.x, self.y)

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1

        # Parameter should be close to optimal (2.0)
        locked = final_model.get_locked_model()
        assert jnp.allclose(locked.param.val, 2.0, rtol=0.1)

    def test_reset_and_rerun(self):
        """Test that reset allows re-running optimization from scratch."""
        configs = [
            PhaseConfig(n_steps=20, optimiser=optax.adam(learning_rate=0.1)),
        ]

        schedule = ManagedOptimiserSchedule(self.model, self.loss_fn, configs)
        schedule.run_all(x=self.x, y=self.y)

        first_run_final = schedule.model_history[-1]
        first_run_losses = schedule.loss_history.copy()

        # Reset and run again
        schedule.reset()
        schedule.run_all(x=self.x, y=self.y)

        second_run_final = schedule.model_history[-1]
        second_run_losses = schedule.loss_history

        # Both runs should produce similar results (fresh optimizer state)
        assert len(first_run_losses) == len(second_run_losses)
        # Loss trajectories should be similar (starting from same initial model)
        assert jnp.allclose(first_run_losses[0], second_run_losses[0], rtol=1e-3)
