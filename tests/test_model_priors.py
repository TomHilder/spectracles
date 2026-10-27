"""test_model_priors.py - Priors on parameters and models (proposal_tests.md section B)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import random

from spectracles import (
    ConstrainedParameter,
    FourierGP,
    Known,
    Matern32,
    OptimiserFrame,
    Parameter,
    ShareModule,
    SpatialDataGeneric,
    build_model,
    build_schedule,
    free_after,
    free_in,
    load_model,
    save_model,
)
from spectracles.model.formatting import get_console
from spectracles.model.spatial import MultiFourierGP
from spectracles.priors import Laplace, LogNormal, Normal, Prior, StdNormal, Uniform
from spectracles.tree.path_utils import str_to_leafpath, use_path_get_leaf


class TwoParams(eqx.Module):
    a: Parameter
    b: Parameter

    def __call__(self):
        return self.a.val + self.b.val


# ---- B1: storing the prior ----


def test_prior_is_stored_and_static():
    prior = LogNormal(1.0, 0.5)
    param = Parameter(1.0, prior=prior)
    assert param.prior is prior
    assert not any(isinstance(leaf, Prior) for leaf in jax.tree.leaves(param))
    # Hyperparameters are not array leaves either
    assert len(jax.tree.leaves(param)) == len(jax.tree.leaves(Parameter(1.0)))


def test_default_is_no_prior():
    assert Parameter(1.0).prior is None
    assert Known(1.0).prior is None


def test_prior_must_be_a_prior():
    with pytest.raises(TypeError):
        Parameter(1.0, prior="LogNormal")  # type: ignore[arg-type]


# ---- B2: forbidden combinations ----


def test_known_cannot_have_prior():
    with pytest.raises(ValueError, match="Known"):
        Known(1.0, prior=Normal(0.0, 1.0))


def test_constrained_parameter_cannot_have_prior():
    with pytest.raises(ValueError, match="prior"):
        ConstrainedParameter(1.0, lower=0.0, prior=LogNormal(1.0, 0.5))


# ---- B3: sharing detection isn't fooled by a shared Prior instance ----


def test_shared_prior_instance_is_not_parameter_sharing():
    prior = Normal(0.0, 1.0)
    model = ShareModule(TwoParams(a=Parameter(1.0, prior=prior), b=Parameter(2.0, prior=prior)))
    assert model.get_sharing_summary(level="parameter") == {}
    assert model.get_parameter_paths() == ["a", "b"]
    assert model() == 3.0


def test_shared_parameter_still_detected_with_prior():
    shared = Parameter(1.0, prior=Normal(0.0, 1.0))
    model = ShareModule(TwoParams(a=shared, b=shared))
    assert model.get_sharing_summary(level="parameter") == {"a": ["b"]}


# ---- repr (part of C3) ----


def test_repr_includes_prior():
    assert "~LogNormal(0.3, 0.15)" in repr(Parameter(0.3, prior=LogNormal(0.3, 0.15)))
    assert "~" not in repr(Parameter(0.3))


# ---- fixtures and models for B4-B10, C ----


class Inner(eqx.Module):
    p: Parameter
    k: Known


class Outer(eqx.Module):
    a: Parameter
    inner: Inner
    b: Parameter


def make_outer(shared_prior=None):
    shared = Parameter(2.0, prior=shared_prior)
    return ShareModule(
        Outer(a=Parameter([1.0, 2.0, 3.0]), inner=Inner(p=shared, k=Known(5.0)), b=shared)
    )


class GPModel(eqx.Module):
    gp: FourierGP

    def __call__(self, data):
        return self.gp(data)


def make_gp_model(with_priors: bool):
    kw = {"prior": LogNormal(0.6, 0.2)} if with_priors else {}
    kernel = Matern32(length_scale=Parameter(0.6, **kw), variance=Parameter(1.0, **kw))
    coeffs = Parameter(dims=(5, 5), prior=StdNormal() if with_priors else None)
    return build_model(GPModel, FourierGP(n_modes=(5, 5), kernel=kernel, coefficients=coeffs))


def gp_data(n=40, seed=0):
    xy = random.uniform(random.PRNGKey(seed), (2, n), minval=-jnp.pi, maxval=jnp.pi)
    return SpatialDataGeneric(x=xy[0], y=xy[1], idx=jnp.arange(n))


def capture(fn):
    console = get_console()
    with console.capture() as cap:
        fn()
    return cap.get()


# ---- B4 / B5: one shared parameter vs two parameters with equal priors ----


def test_shared_parameter_has_one_prior_entry():
    model = make_outer(shared_prior=Normal(0.0, 1.0))
    assert model.get_priors() == {"inner.p": Normal(0.0, 1.0)}


def test_equal_priors_do_not_merge_parameters():
    prior = Normal(0.0, 1.0)
    model = ShareModule(TwoParams(a=Parameter(1.0, prior=prior), b=Parameter(2.0, prior=prior)))
    assert model.get_priors() == {"a": prior, "b": prior}
    model = model.set(["a", "b"], [jnp.array([5.0]), jnp.array([-3.0])])
    assert model() == 2.0


# ---- B6: with_priors ----


def test_with_priors_exact_path():
    model = make_outer()
    new = model.with_priors({"a": Normal(0.0, 2.0)})
    assert new.get_priors() == {"a": Normal(0.0, 2.0)}
    assert model.get_priors() == {}  # original unchanged


def test_with_priors_globs():
    model = make_outer()
    assert set(model.with_priors({"*": Laplace(1.0)}).get_priors()) == {"a", "inner.p"}
    assert set(model.with_priors({"inner.*": Laplace(1.0)}).get_priors()) == {"inner.p"}
    assert set(model.with_priors({"**.p": Laplace(1.0)}).get_priors()) == {"inner.p"}
    assert set(model.with_priors({"**": Laplace(1.0)}).get_priors()) == {"a", "inner.p"}


def test_with_priors_fixed():
    model = make_outer()
    new = model.with_priors({"a": "fixed", "inner.p": Normal(0.0, 1.0)})
    locked = new.get_locked_model()
    assert locked.model.a.fix
    assert not locked.model.inner.p.fix and not locked.model.b.fix
    assert new.get_priors() == {"inner.p": Normal(0.0, 1.0)}


def test_with_priors_no_match_raises():
    with pytest.raises(ValueError, match="did not match"):
        make_outer().with_priors({"nope": Normal(0.0, 1.0)})
    # Known values are excluded from matching
    with pytest.raises(ValueError, match="did not match"):
        make_outer().with_priors({"inner.k": Normal(0.0, 1.0)})


def test_with_priors_bad_value_raises():
    with pytest.raises(TypeError):
        make_outer().with_priors({"a": "LogNormal"})


def test_with_priors_child_path_sets_the_one_parameter():
    model = make_outer()
    new = model.with_priors({"b": Uniform(0.0, 5.0)})  # b is shared from inner.p
    assert new.get_priors() == {"inner.p": Uniform(0.0, 5.0)}
    locked = new.get_locked_model()
    assert locked.model.b.prior == locked.model.inner.p.prior == Uniform(0.0, 5.0)
    # Still one parameter
    assert new.get_sharing_summary(level="parameter") == {"inner.p": ["b"]}
    new = new.set(["inner.p"], [jnp.array([4.0])])
    locked = new.get_locked_model()
    assert locked.model.b.val == locked.model.inner.p.val == 4.0


def test_with_priors_conflict_raises():
    with pytest.raises(ValueError, match="conflicting"):
        make_outer().with_priors({"inner.p": Normal(0.0, 1.0), "b": Normal(0.0, 2.0)})
    # The same prior through both paths is fine
    new = make_outer().with_priors({"inner.p": Normal(0.0, 1.0), "b": Normal(0.0, 1.0)})
    assert new.get_priors() == {"inner.p": Normal(0.0, 1.0)}


def test_with_priors_keeps_values_and_status():
    model = make_outer().set_fixed_status(["a"], [True])
    new = model.with_priors({"a": Normal(0.0, 1.0)})
    old_l, new_l = model.get_locked_model(), new.get_locked_model()
    np.testing.assert_array_equal(new_l.model.a.val, old_l.model.a.val)
    np.testing.assert_array_equal(new_l.model.inner.p.val, old_l.model.inner.p.val)
    assert new_l.model.a.fix  # fixed status kept


def test_with_priors_rejects_constrained_parameter():
    model = ShareModule(TwoParams(a=ConstrainedParameter(1.0, lower=0.0), b=Parameter(1.0)))
    with pytest.raises(ValueError, match="ConstrainedParameter"):
        model.with_priors({"a": LogNormal(1.0, 0.5)})
    # Fixing it is fine
    assert model.with_priors({"a": "fixed"}).get_locked_model().model.a.fix


def test_with_priors_on_locked_model_raises():
    with pytest.raises(ValueError, match="locked"):
        make_outer().get_locked_model().with_priors({"a": Normal(0.0, 1.0)})


# ---- B7: FourierGP defaults ----


def test_fourier_gp_coefficients_default_to_std_normal():
    kernel = Matern32(length_scale=Parameter(1.0), variance=Parameter(1.0))
    assert FourierGP(n_modes=(5, 5), kernel=kernel).coefficients.prior == StdNormal()
    multi = MultiFourierGP(n_components=2, n_modes=(5, 5), kernels=[kernel, kernel])
    assert multi.coefficients.prior == StdNormal()


def test_fourier_gp_user_coefficients_keep_their_prior():
    kernel = Matern32(length_scale=Parameter(1.0), variance=Parameter(1.0))
    gp = FourierGP(n_modes=(5, 5), kernel=kernel, coefficients=Parameter(dims=(5, 5)))
    assert gp.coefficients.prior is None
    gp = FourierGP(
        n_modes=(5, 5), kernel=kernel, coefficients=Parameter(dims=(5, 5), prior=Normal(0.0, 2.0))
    )
    assert gp.coefficients.prior == Normal(0.0, 2.0)


# ---- B8: JIT ----


def test_jit_compiles_once_for_new_values():
    model = make_gp_model(with_priors=True)
    data = gp_data()
    n_traces = [0]

    @eqx.filter_jit
    def f(m):
        n_traces[0] += 1
        return m(data)

    f(model)
    f(model.set(["gp.kernel.length_scale"], [jnp.array([0.9])]))
    f(model.set(["gp.coefficients"], [jnp.ones((5, 5))]))
    assert n_traces[0] == 1


# ---- B9: I/O ----


def test_save_load_preserves_priors_and_sharing(temp_dir):
    model = make_outer(shared_prior=LogNormal(2.0, 1.0)).with_priors({"a": Laplace(0.5)})
    save_model(model, temp_dir / "m")
    loaded = load_model(temp_dir / "m")
    assert loaded.get_priors() == model.get_priors()
    assert loaded.get_sharing_summary(level="parameter") == {"inner.p": ["b"]}


# ---- B10: the old optimisation path is unaffected ----


def _three_sgd_steps(model, data, target):
    def loss_fn(m, data, target):
        return jnp.mean((m(data) - target) ** 2)

    frame = OptimiserFrame(model, loss_fn, optax.sgd(1e-2))
    out = frame.run(3, data, target)
    return np.asarray(frame.loss_history), out


def test_optimiser_frame_identical_with_and_without_priors():
    data = gp_data()
    target = jnp.sin(data.x) * jnp.cos(data.y)
    loss_a, out_a = _three_sgd_steps(make_gp_model(with_priors=False), data, target)
    loss_b, out_b = _three_sgd_steps(make_gp_model(with_priors=True), data, target)
    np.testing.assert_array_equal(loss_a, loss_b)
    # Compare parameter values (ShareModule's own leaves include id-keyed metadata)
    la, lb = out_a.get_locked_model().model, out_b.get_locked_model().model
    for path in out_a.get_parameter_paths(show_knowns=True):
        leaf_path = str_to_leafpath(path)
        np.testing.assert_array_equal(
            use_path_get_leaf(la, leaf_path).val, use_path_get_leaf(lb, leaf_path).val
        )


def test_schedule_builder_and_fix_all_with_priors():
    model = make_gp_model(with_priors=True)

    def loss_fn(m, data):
        return jnp.mean(m(data) ** 2)

    schedule = build_schedule(
        model,
        loss_fn,
        phases=[(2, 1e-2), (2, 1e-2)],
        params={"gp.coefficients": free_in(0, 1), "gp.kernel.*": free_after(1)},
    )
    assert schedule is not None
    assert all(p.fix for p in [model.fix_all().get_locked_model().model.gp.coefficients])
    assert model.fix_all().free_all().get_priors() == model.get_priors()


# ---- C1-C3: inspection ----

# Captured from spectracles main (b7b6df1) for the same model, no priors
GOLDEN_SUMMARY = (
    "                 Parameter Summary                  \n"
    "┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓\n"
    "┃ Path    ┃ Shape ┃ Bounds  ┃ Status ┃ Shared from ┃\n"
    "┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩\n"
    "│ a       │ (3,)  │ (-∞, ∞) │  free  │ -           │\n"
    "│ inner.p │ (1,)  │ (-∞, ∞) │  free  │ -           │\n"
    "│ b       │ (1,)  │ (-∞, ∞) │  free  │ inner.p     │\n"
    "└─────────┴───────┴─────────┴────────┴─────────────┘\n"
    "3 parameters (2 unique, 1 shared) · 3 free, 0 fixed\n"
)


def test_summary_without_priors_unchanged_from_main():
    assert capture(make_outer().get_parameter_summary) == GOLDEN_SUMMARY


def test_summary_prior_column():
    model = make_outer(shared_prior=LogNormal(2.0, 1.0)).with_priors({"a": "fixed"})
    out = capture(model.get_parameter_summary)
    header = out.splitlines()[2]
    assert [h.strip() for h in header.strip("┃").split("┃")] == [
        "Path",
        "Shape",
        "Bounds",
        "Prior",
        "Status",
        "Shared from",
    ]
    rows = {line.split("│")[1].strip(): line for line in out.splitlines() if line.startswith("│")}
    assert "LogNormal(2, 1)" in rows["inner.p"]
    assert "LogNormal(2, 1)" in rows["b"]  # the shared row shows the one parameter's prior
    assert rows["a"].split("│")[4].strip() == "-"


def test_repr_and_tree_show_priors():
    model = make_outer(shared_prior=LogNormal(2.0, 1.0))
    assert "~LogNormal(2, 1)" in repr(model)
    assert "1 with priors" in repr(model)
    assert "~LogNormal(2, 1)" in capture(model.print_model_tree)
    assert "with priors" not in repr(make_outer())
