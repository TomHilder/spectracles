"""test_schedule_builder.py - tests for the spectracles.optimise.schedule_builder module."""

import jax.numpy as jnp
import jax.random as jr
import optax  # type: ignore[import]
import pytest
from spectracles.model.share_module import build_model
from spectracles.optimise.opt_schedule import OptimiserSchedule, ManagedOptimiserSchedule
from spectracles.optimise.schedule_builder import (
    ParamSpec,
    InitSpec,
    free_in,
    free_after,
    free_until,
    fixed_in,
    init_normal,
    init_value,
    init_uniform,
    expand_patterns,
    build_schedule,
    _match_pattern,
    _simple_match,
    _resolve_free_phases,
)

from .test_models import SimpleModel, NestedModel, SharedLeafModel, ComplexSharedModel


class TestParamSpec:
    def test_basic_creation(self):
        spec = ParamSpec(free_phases={0, 1, 2})
        assert spec.free_phases == {0, 1, 2}
        assert spec.init_specs == {}

    def test_combine_with_or(self):
        spec1 = ParamSpec(free_phases={0, 1})
        spec2 = ParamSpec(free_phases={2, 3})

        combined = spec1 | spec2
        assert combined.free_phases == {0, 1, 2, 3}

    def test_combine_with_init_specs(self):
        spec1 = ParamSpec(free_phases={0})
        spec2 = ParamSpec(init_specs={0: InitSpec(kind="normal")})

        combined = spec1 | spec2
        assert combined.free_phases == {0}
        assert 0 in combined.init_specs


class TestHelperFunctions:
    def test_free_in(self):
        spec = free_in(0, 2, 4)
        assert spec.free_phases == {0, 2, 4}

    def test_free_in_single(self):
        spec = free_in(1)
        assert spec.free_phases == {1}

    def test_free_after(self):
        spec = free_after(2)
        # free_after uses sentinel values
        resolved = _resolve_free_phases(spec, n_phases=5)
        assert resolved == {2, 3, 4}

    def test_free_after_zero(self):
        spec = free_after(0)
        resolved = _resolve_free_phases(spec, n_phases=3)
        assert resolved == {0, 1, 2}

    def test_free_until(self):
        spec = free_until(2)
        assert spec.free_phases == {0, 1, 2}

    def test_fixed_in(self):
        spec = fixed_in(0)
        resolved = _resolve_free_phases(spec, n_phases=3)
        assert resolved == {1, 2}  # Free in all phases except 0

    def test_fixed_in_multiple(self):
        spec = fixed_in(0, 2)
        resolved = _resolve_free_phases(spec, n_phases=4)
        assert resolved == {1, 3}

    def test_init_normal(self):
        spec = init_normal(0, mean=1.0, std=0.5)
        assert 0 in spec.init_specs
        assert spec.init_specs[0].kind == "normal"
        assert spec.init_specs[0].mean == 1.0
        assert spec.init_specs[0].std == 0.5

    def test_init_value(self):
        spec = init_value(1, 2.5)
        assert 1 in spec.init_specs
        assert spec.init_specs[1].kind == "value"
        assert spec.init_specs[1].value == 2.5

    def test_init_uniform(self):
        spec = init_uniform(0, low=-1.0, high=1.0)
        assert 0 in spec.init_specs
        assert spec.init_specs[0].kind == "uniform"
        assert spec.init_specs[0].low == -1.0
        assert spec.init_specs[0].high == 1.0

    def test_combine_free_and_init(self):
        spec = free_in(0, 1) | init_normal(0)
        assert spec.free_phases == {0, 1}
        assert 0 in spec.init_specs


class TestPatternMatching:
    def test_exact_match(self):
        assert _match_pattern("param", "param")
        assert not _match_pattern("param", "other")

    def test_exact_match_nested(self):
        assert _match_pattern("inner.param", "inner.param")
        assert not _match_pattern("inner.param", "inner.other")

    def test_wildcard_single(self):
        assert _match_pattern("*.param", "inner.param")
        assert _match_pattern("*.param", "outer.param")
        assert not _match_pattern("*.param", "param")

    def test_wildcard_middle(self):
        assert _match_pattern("inner.*.val", "inner.param.val")
        assert _match_pattern("inner.*.val", "inner.other.val")
        assert not _match_pattern("inner.*.val", "inner.val")

    def test_wildcard_end(self):
        assert _match_pattern("inner.*", "inner.param")
        assert _match_pattern("inner.*", "inner.other")
        assert not _match_pattern("inner.*", "outer.param")

    def test_simple_match_length_mismatch(self):
        assert not _simple_match("a.b", "a")
        assert not _simple_match("a", "a.b")

    def test_double_wildcard_suffix(self):
        assert _match_pattern("**.param", "inner.param")
        assert _match_pattern("**.param", "a.b.c.param")
        assert _match_pattern("**.param", "param")  # ** can match zero components

    def test_double_wildcard_prefix(self):
        assert _match_pattern("inner.**", "inner.a")
        assert _match_pattern("inner.**", "inner.a.b.c")


class TestExpandPatterns:
    def test_exact_match(self):
        parent_paths = ["param", "inner.param"]
        shared_paths = []

        result = expand_patterns(
            {"param": free_in(0)},
            parent_paths,
            shared_paths,
        )
        assert "param" in result
        assert "inner.param" not in result

    def test_wildcard_expansion(self):
        parent_paths = ["inner1.param", "inner2.param", "shared"]
        shared_paths = []

        result = expand_patterns(
            {"*.param": free_in(0)},
            parent_paths,
            shared_paths,
        )
        assert "inner1.param" in result
        assert "inner2.param" in result
        assert "shared" not in result

    def test_multiple_patterns(self):
        parent_paths = ["a.param", "b.param", "c.other"]
        shared_paths = []

        result = expand_patterns(
            {
                "a.param": free_in(0),
                "b.param": free_in(1),
            },
            parent_paths,
            shared_paths,
        )
        assert "a.param" in result
        assert "b.param" in result
        assert "c.other" not in result

    def test_pattern_no_match_raises(self):
        parent_paths = ["param"]
        shared_paths = []

        with pytest.raises(ValueError, match="did not match any parameters"):
            expand_patterns(
                {"nonexistent": free_in(0)},
                parent_paths,
                shared_paths,
            )

    def test_pattern_only_shared_raises(self):
        parent_paths = ["a.param"]
        shared_paths = ["b.param"]

        with pytest.raises(ValueError, match="only matches shared parameters"):
            expand_patterns(
                {"b.param": free_in(0)},
                parent_paths,
                shared_paths,
            )

    def test_combine_specs_same_path(self):
        parent_paths = ["param"]
        shared_paths = []

        result = expand_patterns(
            {
                "param": free_in(0),
                "*": init_normal(0),  # Matches same path
            },
            parent_paths,
            shared_paths,
        )
        assert "param" in result
        assert 0 in result["param"].free_phases
        assert 0 in result["param"].init_specs


class TestBuildSchedule:
    def setup_method(self):
        self.model = build_model(SimpleModel, value=1.0)
        self.x = jnp.array([1.0, 2.0, 3.0])
        self.y = 2.0 * self.x

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        self.loss_fn = loss_fn

    def test_basic_build(self):
        schedule = build_schedule(
            self.model,
            self.loss_fn,
            phases=[
                (10, 0.1),
                (10, 0.01),
            ],
            params={
                "param": free_in(0, 1),
            },
        )

        assert isinstance(schedule, OptimiserSchedule)
        assert len(schedule.phases) == 2

    def test_build_managed(self):
        schedule = build_schedule(
            self.model,
            self.loss_fn,
            phases=[(10, 0.1)],
            params={"param": free_in(0)},
            managed=True,
        )

        assert isinstance(schedule, ManagedOptimiserSchedule)

    def test_run_schedule(self):
        schedule = build_schedule(
            self.model,
            self.loss_fn,
            phases=[
                (50, 0.1),
            ],
            params={
                "param": free_in(0),
            },
        )

        schedule.run_all(x=self.x, y=self.y)

        final_model = schedule.model_history[-1]
        locked = final_model.get_locked_model()
        # Should converge to ~2.0
        assert jnp.allclose(locked.param.val, 2.0, rtol=0.1)

    def test_multi_phase_different_free(self):
        """Test that parameters can be free in some phases, fixed in others."""
        nested_model = build_model(NestedModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        schedule = build_schedule(
            nested_model,
            loss_fn,
            phases=[
                (10, 0.1),  # Phase 0
                (10, 0.01),  # Phase 1
            ],
            params={
                "inner1.param": free_in(0),  # Only free in phase 0
                "inner2.param": free_after(1),  # Only free in phase 1
                "shared": free_in(0, 1),  # Free in both
            },
        )

        assert len(schedule.phases) == 2
        # Check phase 0 config
        phase0_config = schedule.phases[0].config
        assert phase0_config.fix_status_updates["inner1.param"] is False
        assert phase0_config.fix_status_updates["inner2.param"] is True
        assert phase0_config.fix_status_updates["shared"] is False

        # Check phase 1 config
        phase1_config = schedule.phases[1].config
        assert phase1_config.fix_status_updates["inner1.param"] is True
        assert phase1_config.fix_status_updates["inner2.param"] is False
        assert phase1_config.fix_status_updates["shared"] is False

    def test_init_value(self):
        schedule = build_schedule(
            self.model,
            self.loss_fn,
            phases=[(10, 0.1)],
            params={
                "param": free_in(0) | init_value(0, 5.0),
            },
        )

        phase0_config = schedule.phases[0].config
        assert "param" in phase0_config.param_val_updates
        assert jnp.allclose(phase0_config.param_val_updates["param"], 5.0)

    def test_init_normal_requires_key(self):
        with pytest.raises(ValueError, match="PRNG key required"):
            build_schedule(
                self.model,
                self.loss_fn,
                phases=[(10, 0.1)],
                params={
                    "param": free_in(0) | init_normal(0),
                },
            )

    def test_init_normal_with_key(self):
        key = jr.key(42)

        schedule = build_schedule(
            self.model,
            self.loss_fn,
            phases=[(10, 0.1)],
            params={
                "param": free_in(0) | init_normal(0, mean=0.0, std=1.0),
            },
            key=key,
        )

        phase0_config = schedule.phases[0].config
        assert "param" in phase0_config.param_val_updates

    def test_init_uniform_with_key(self):
        key = jr.key(42)

        schedule = build_schedule(
            self.model,
            self.loss_fn,
            phases=[(10, 0.1)],
            params={
                "param": free_in(0) | init_uniform(0, low=-1.0, high=1.0),
            },
            key=key,
        )

        phase0_config = schedule.phases[0].config
        assert "param" in phase0_config.param_val_updates
        val = phase0_config.param_val_updates["param"]
        assert jnp.all(val >= -1.0) and jnp.all(val <= 1.0)

    def test_custom_optimizer(self):
        schedule = build_schedule(
            self.model,
            self.loss_fn,
            phases=[
                (10, 0.1, optax.sgd(0.1)),  # Explicit optimizer
            ],
            params={
                "param": free_in(0),
            },
        )

        assert len(schedule.phases) == 1

    def test_default_optimizer_sgd(self):
        schedule = build_schedule(
            self.model,
            self.loss_fn,
            phases=[(10, 0.1)],
            params={"param": free_in(0)},
            default_optimizer="sgd",
        )

        assert len(schedule.phases) == 1

    def test_wildcard_with_nested_model(self):
        nested_model = build_model(NestedModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        y = 2.0 * x

        schedule = build_schedule(
            nested_model,
            loss_fn,
            phases=[(20, 0.1)],
            params={
                "*.param": free_in(0),  # Matches inner1.param and inner2.param
                "shared": free_in(0),
            },
        )

        schedule.run_all(x=x, y=y)
        assert schedule.loss_history[0] > schedule.loss_history[-1]


class TestBuildScheduleWithSharing:
    def test_shared_model_parent_path(self):
        """Test that we can use parent path for shared parameters."""
        model = build_model(SharedLeafModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        y = x  # Optimal is a=0.5, b=0.5 (shared)

        # 'a' is the parent path for the shared parameter
        schedule = build_schedule(
            model,
            loss_fn,
            phases=[(50, 0.1)],
            params={
                "a": free_in(0),  # 'a' is the parent
            },
        )

        schedule.run_all(x=x, y=y)
        final_model = schedule.model_history[-1]
        locked = final_model.get_locked_model()
        assert jnp.allclose(locked.a.val, 0.5, rtol=0.1)

    def test_shared_model_child_path_raises(self):
        """Test that specifying a shared (child) path raises an error."""
        model = build_model(SharedLeafModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        # 'b' is shared from 'a', so specifying 'b' should error
        with pytest.raises(ValueError, match="only matches shared parameters"):
            build_schedule(
                model,
                loss_fn,
                phases=[(10, 0.1)],
                params={
                    "b": free_in(0),  # 'b' is shared, should error
                },
            )

    def test_complex_shared_model(self):
        """Test with ComplexSharedModel that has multiple levels of sharing."""
        model = build_model(ComplexSharedModel, value=1.0)

        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        y = x

        # Get the parent paths to verify
        parent_paths = model.get_parameter_paths(show_shared=False)

        # Should be able to use wildcard that only matches parent paths
        schedule = build_schedule(
            model,
            loss_fn,
            phases=[(10, 0.1)],
            params={
                p: free_in(0) for p in parent_paths
            },
        )

        assert len(schedule.phases) == 1


class TestResolveFreePhasesEdgeCases:
    def test_empty_spec(self):
        spec = ParamSpec()
        resolved = _resolve_free_phases(spec, n_phases=3)
        assert resolved == set()

    def test_single_phase(self):
        spec = free_in(0)
        resolved = _resolve_free_phases(spec, n_phases=1)
        assert resolved == {0}

    def test_free_after_last_phase(self):
        spec = free_after(5)
        resolved = _resolve_free_phases(spec, n_phases=3)
        # free_after(5) with only 3 phases means never free
        assert resolved == set()
