"""test_latent.py - The latent space (proposal_tests.md sections D and C4)."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from spectracles import ConstrainedParameter, Parameter, ShareModule
from spectracles.inference import LatentSpace
from spectracles.model.formatting import get_console
from spectracles.priors import Laplace, LogNormal, Normal, StdNormal
from spectracles.tree.path_utils import str_to_leafpath, use_path_get_leaf

from .conftest import make_xi_points
from .priors_models import gp_data, line_data, line_forward, make_line_model
from .test_model_priors import TwoParams, make_outer

# Tolerances: one decade looser than what these checks achieve on CPU float64
# (rtol 1e-14 with atol 1e-12 for near-zero components; probed in P1.1 stage 7)
RTOL = 1e-13
ATOL = 1e-11

LINE_PATHS = [
    "line_1.log_A.coefficients",
    "line_1.log_A.kernel.length_scale",
    "line_1.log_A.kernel.variance",
    "line_1.v.coefficients",
    "line_1.v.kernel.length_scale",
    "line_1.v.kernel.variance",
    "line_2.log_A.coefficients",
    "line_2.log_A.kernel.length_scale",
    "line_2.log_A.kernel.variance",
    "line_2.v.coefficients",
    "offset.tile_values",
    "scatter.spaxel_values",
]


def leaf(model, path):
    return use_path_get_leaf(model.for_evaluation().model, str_to_leafpath(path))


# ---- D1: keys ----


def test_paths_are_the_free_parameters_with_priors(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    # No fixed σ, no Known μ, and the shared velocity kernel only once (line_1)
    assert latent.paths == LINE_PATHS
    assert not any(p.startswith("line_2.v.kernel") for p in latent.paths)


def test_path_order_is_deterministic(tiny_line_model):
    assert LatentSpace(tiny_line_model).paths == LatentSpace(make_line_model()).paths
    assert LatentSpace(tiny_line_model).paths == [
        p for p in tiny_line_model.get_parameter_paths() if p in LINE_PATHS
    ]


def test_shared_parameter_is_one_key():
    model = make_outer(shared_prior=Normal(0.0, 1.0)).with_priors({"a": Normal(0.0, 1.0)})
    assert LatentSpace(model).paths == ["a", "inner.p"]  # b is the same parameter


def test_equal_priors_are_separate_keys():
    prior = Normal(0.0, 1.0)
    model = ShareModule(TwoParams(a=Parameter(1.0, prior=prior), b=Parameter(2.0, prior=prior)))
    latent = LatentSpace(model)
    assert latent.paths == ["a", "b"]
    m = latent.to_model({"a": jnp.array([0.5]), "b": jnp.array([-1.5])})
    assert leaf(m, "a").val == 0.5 and leaf(m, "b").val == -1.5


# ---- D2: shapes ----


def test_shapes_match_values(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    for p in latent.paths:
        assert latent.shapes[p].shape == leaf(tiny_line_model, p).val.shape
    assert latent.shapes["line_1.log_A.coefficients"].shape == (5, 5)
    assert latent.size == 4 * 25 + 6 * 1 + 2 + 30  # coefficients, kernel scalars, tiles, spaxels


# ---- D3: validation errors list every path ----


def test_free_parameters_without_prior_listed():
    model = make_outer()  # a and inner.p are free without priors
    with pytest.raises(ValueError) as err:
        LatentSpace(model)
    assert "'a'" in str(err.value) and "'inner.p'" in str(err.value)


def test_free_constrained_parameter_rejected():
    model = ShareModule(
        TwoParams(a=ConstrainedParameter(1.0, lower=0.0), b=Parameter(1.0, prior=StdNormal()))
    )
    with pytest.raises(ValueError, match="ConstrainedParameter"):
        LatentSpace(model)
    # Fixed, it's fine
    assert LatentSpace(model.set_fixed_status(["a"], [True])).paths == ["b"]


def test_requires_unlocked_share_module(tiny_gp_model):
    with pytest.raises(ValueError):
        LatentSpace(tiny_gp_model.for_evaluation())
    with pytest.raises(TypeError):
        LatentSpace(tiny_gp_model.model)


# ---- D4: to_model ----


def test_to_model_sets_transformed_values(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    ξ = latent.sample_prior_latent(random.PRNGKey(0))
    m = latent.to_model(ξ)
    for p in latent.paths:
        np.testing.assert_allclose(leaf(m, p).val, latent.priors[p].transform(ξ[p]), rtol=RTOL)


def test_to_model_leaves_fixed_and_known_untouched(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    m = latent.to_model(latent.sample_prior_latent(random.PRNGKey(0)))
    for p in ["line_1.σ", "line_2.σ", "line_1.μ", "line_2.μ"]:
        np.testing.assert_array_equal(leaf(m, p).val, leaf(tiny_line_model, p).val)


def test_shared_parameter_written_once(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    ξ = latent.sample_prior_latent(random.PRNGKey(0))
    m = latent.to_model(ξ)
    for name in ("length_scale", "variance"):
        parent = leaf(m, f"line_1.v.kernel.{name}").val
        np.testing.assert_array_equal(leaf(m, f"line_2.v.kernel.{name}").val, parent)
    # Changing the parent latent changes every location; nothing else does
    ξ2 = dict(ξ, **{"line_1.v.kernel.length_scale": ξ["line_1.v.kernel.length_scale"] + 1.0})
    m2 = latent.to_model(ξ2)
    new = leaf(m2, "line_1.v.kernel.length_scale").val
    assert not jnp.allclose(new, leaf(m, "line_1.v.kernel.length_scale").val)
    np.testing.assert_array_equal(leaf(m2, "line_2.v.kernel.length_scale").val, new)
    ξ3 = dict(ξ, **{"line_2.log_A.kernel.length_scale": jnp.array([3.0])})
    m3 = latent.to_model(ξ3)
    np.testing.assert_array_equal(
        leaf(m3, "line_2.v.kernel.length_scale").val, leaf(m, "line_2.v.kernel.length_scale").val
    )


def test_to_model_accepts_dict_like_with_tree(tiny_gp_model):
    latent = LatentSpace(tiny_gp_model)
    ξ = latent.sample_prior_latent(random.PRNGKey(0))

    class Wrapper:
        tree = ξ

    m1, m2 = latent.to_model(ξ), latent.to_model(Wrapper())
    for p in latent.paths:
        np.testing.assert_array_equal(leaf(m1, p).val, leaf(m2, p).val)


# ---- D5: round trip ----


def test_round_trip_all_invertible(tiny_gp_model):
    latent = LatentSpace(tiny_gp_model)
    for ξ in make_xi_points(latent):
        back = latent.from_model(latent.to_model(ξ))
        for p in latent.paths:
            np.testing.assert_allclose(back[p], ξ[p], rtol=RTOL, atol=ATOL)
    m = latent.to_model(make_xi_points(latent)[0])
    m2 = latent.to_model(latent.from_model(m))
    for p in latent.paths:
        np.testing.assert_allclose(leaf(m2, p).val, leaf(m, p).val, rtol=RTOL)


def test_from_model_no_warning_when_all_invertible(tiny_gp_model):
    latent = LatentSpace(tiny_gp_model)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        latent.from_model(tiny_gp_model)


# ---- D6: fallback for priors without an inverse ----


def test_from_model_fallback_zero_warns(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    ξ = latent.sample_prior_latent(random.PRNGKey(0))
    m = latent.to_model(ξ)
    with pytest.warns(UserWarning) as record:
        back = latent.from_model(m)
    assert len(record) == 1
    message = str(record[0].message)
    assert "'scatter.spaxel_values'" in message
    assert all(f"'{p}'" not in message for p in latent.paths if p != "scatter.spaxel_values")
    np.testing.assert_array_equal(back["scatter.spaxel_values"], 0.0)
    for p in latent.paths:
        if p != "scatter.spaxel_values":
            np.testing.assert_allclose(back[p], ξ[p], rtol=RTOL, atol=ATOL)


def test_from_model_fallback_raise(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    with pytest.raises(ValueError, match="scatter.spaxel_values"):
        latent.from_model(tiny_line_model, fallback="raise")
    with pytest.raises(ValueError):
        latent.from_model(tiny_line_model, fallback="nope")  # type: ignore[arg-type]


# ---- D7: prior draws land the right prior on the right parameter ----


def test_sample_prior_model_matches_to_model_of_latent_draw(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    key = random.PRNGKey(7)
    a, b = latent.sample_prior_model(key), latent.to_model(latent.sample_prior_latent(key))
    for p in latent.paths:
        np.testing.assert_array_equal(leaf(a, p).val, leaf(b, p).val)


def test_to_params_single_and_stacked(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    ξ = latent.sample_prior_latent(random.PRNGKey(3))
    params = latent.to_params(ξ)
    assert list(params) == latent.paths
    for p in latent.paths:
        np.testing.assert_array_equal(params[p], latent.priors[p].transform(ξ[p]))
        assert params[p].shape == latent.shapes[p].shape
    stacked = latent.sample_prior_latent(random.PRNGKey(4), n=5)
    params = latent.to_params(stacked)
    for p in latent.paths:
        assert params[p].shape == (5, *latent.shapes[p].shape)
        np.testing.assert_array_equal(
            params[p][2], latent.to_params({q: v[2] for q, v in stacked.items()})[p]
        )


def test_to_params_keeps_chain_and_draw_axes(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    flat = latent.sample_prior_latent(random.PRNGKey(8), n=6)
    chains = {p: v.reshape(2, 3, *v.shape[1:]) for p, v in flat.items()}
    params, flat_params = latent.to_params(chains), latent.to_params(flat)
    for p in latent.paths:
        assert params[p].shape == (2, 3, *latent.shapes[p].shape)
        np.testing.assert_array_equal(params[p].reshape(flat_params[p].shape), flat_params[p])


def test_posterior_summary_of_latents_or_params(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    stacked = latent.sample_prior_latent(random.PRNGKey(9), n=20)
    from_latents = latent.posterior_summary(stacked, print_table=False)
    from_params = latent.posterior_summary(
        latent.to_params(stacked), samples_type="params", print_table=False
    )
    for p in latent.paths:
        np.testing.assert_allclose(from_params[p]["mean"], from_latents[p]["mean"])
        np.testing.assert_allclose(from_params[p]["std"], from_latents[p]["std"])
    with pytest.raises(ValueError, match="samples_type"):
        latent.posterior_summary(stacked, samples_type="values", print_table=False)  # type: ignore[arg-type]


def test_posterior_summary_with_truth(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    stacked = latent.sample_prior_latent(random.PRNGKey(10), n=20)
    truth = latent.sample_prior_latent(random.PRNGKey(11))
    by_latents = latent.posterior_summary(stacked, true_latents=truth, print_table=False)
    by_params = latent.posterior_summary(
        stacked, true_params=latent.to_params(truth), print_table=False
    )
    for p in latent.paths:
        s = by_latents[p]
        np.testing.assert_allclose(s["truth"], latent.to_params(truth)[p])
        np.testing.assert_allclose(s["z"], (s["mean"] - s["truth"]) / s["std"])
        np.testing.assert_allclose(by_params[p]["z"], s["z"])
    with pytest.raises(ValueError, match="not both"):
        latent.posterior_summary(stacked, true_latents=truth, true_params=truth, print_table=False)
    with pytest.raises(ValueError, match="one draw"):
        latent.posterior_summary(stacked, true_latents=stacked, print_table=False)


def test_map_samples_matches_a_loop_over_to_model(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    stacked = latent.sample_prior_latent(random.PRNGKey(13), n=6)

    def fn(m):
        return {
            "ls": leaf(m, "line_1.log_A.kernel.length_scale").val,
            "two": 2.0 * leaf(m, "offset.tile_values").val,
        }

    out = latent.map_samples(fn, stacked)
    for i in range(6):
        one = fn(latent.to_model({p: v[i] for p, v in stacked.items()}))
        np.testing.assert_allclose(out["ls"][i], one["ls"])
        np.testing.assert_allclose(out["two"][i], one["two"])
    # Parameter values give the same result
    by_params = latent.map_samples(fn, latent.to_params(stacked), samples_type="params")
    np.testing.assert_allclose(by_params["ls"], out["ls"])
    # (chain, draw) axes are kept
    chains = {p: v.reshape(2, 3, *v.shape[1:]) for p, v in stacked.items()}
    np.testing.assert_allclose(latent.map_samples(fn, chains)["ls"], out["ls"].reshape(2, 3, 1))
    with pytest.raises(ValueError, match="leading axis"):
        latent.map_samples(fn, {p: v[0] for p, v in stacked.items()})


def test_to_model_is_set_of_to_params(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    ξ = latent.sample_prior_latent(random.PRNGKey(5))
    a, b = latent.to_model(ξ), latent.model.set(latent.to_params(ξ))
    for p in latent.paths:
        np.testing.assert_array_equal(leaf(a, p).val, leaf(b, p).val)


def test_sample_prior_with_n(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    key = random.PRNGKey(6)
    ξ = latent.sample_prior_latent(key, n=4)
    assert all(ξ[p].shape == (4, *latent.shapes[p].shape) for p in latent.paths)
    params = latent.sample_prior_params(key, n=4)
    for p in latent.paths:
        np.testing.assert_array_equal(params[p], latent.to_params(ξ)[p])
    models = latent.sample_prior_model(key, n=4)
    assert isinstance(models, list) and len(models) == 4
    for i, m in enumerate(models):
        for p in latent.paths:
            np.testing.assert_array_equal(leaf(m, p).val, params[p][i])
    single = latent.sample_prior_params(key)
    assert all(single[p].shape == latent.shapes[p].shape for p in latent.paths)


def test_prior_draw_moments(tiny_line_model):
    latent = LatentSpace(tiny_line_model)
    n = 2000

    def values(key):
        m = latent.sample_prior_model(key)
        return {p: leaf(m, p).val for p in latent.paths}

    draws = jax.vmap(values)(random.split(random.PRNGKey(1), n))
    checks = {
        "line_1.log_A.kernel.length_scale": (0.8, 0.3),
        "line_1.v.kernel.variance": (1.0, 0.3),
        "offset.tile_values": (0.0, 0.1),
        "line_1.log_A.coefficients": (0.0, 1.0),
    }
    for p, (mean, std) in checks.items():
        x = np.asarray(draws[p]).ravel()
        assert abs(x.mean() - mean) < 5 * x.std() / np.sqrt(x.size), p
        assert x.std() == pytest.approx(std, rel=0.1), p
    # Laplace(0.01): E|x| = 0.01
    s = np.abs(np.asarray(draws["scatter.spaxel_values"]))
    assert s.mean() == pytest.approx(0.01, rel=0.05)


# ---- D8: pure JAX ----


def test_to_model_under_jit_grad_vmap(tiny_gp_model):
    latent = LatentSpace(tiny_gp_model)
    data = gp_data()

    def f(ξ):
        return jnp.sum(latent.to_model(ξ)(data) ** 2)

    ξs = make_xi_points(latent)
    np.testing.assert_allclose(jax.jit(f)(ξs[0]), f(ξs[0]), rtol=RTOL)
    g = jax.grad(f)(ξs[0])
    assert all(jnp.all(jnp.isfinite(g[p])) for p in latent.paths)
    batch = jax.tree.map(lambda *x: jnp.stack(x), *ξs)
    np.testing.assert_allclose(jax.vmap(f)(batch), jnp.array([f(ξ) for ξ in ξs]), rtol=RTOL)


# ---- D9: gradients through a shared parameter add up ----


def test_gradient_through_shared_parameter_is_sum_of_uses():
    spatial = line_data()
    forward = line_forward(spatial)
    tied = LatentSpace(make_line_model())
    untied = LatentSpace(make_line_model(untie_velocity_kernel=True))
    ξ = tied.sample_prior_latent(random.PRNGKey(0))
    # The untied model gets the same latent for both copies of the velocity kernel
    ξ_untied = dict(ξ)
    for name in ("length_scale", "variance"):
        ξ_untied[f"line_2.v.kernel.{name}"] = ξ[f"line_1.v.kernel.{name}"]
    weights = random.normal(random.PRNGKey(9), (12, 30))

    def f_tied(ξ):
        return jnp.sum(weights * forward(tied.to_model(ξ)))

    def f_untied(ξ):
        return jnp.sum(weights * forward(untied.to_model(ξ)))

    np.testing.assert_allclose(f_tied(ξ), f_untied(ξ_untied), rtol=1e-12)
    g_tied, g_untied = jax.grad(f_tied)(ξ), jax.grad(f_untied)(ξ_untied)
    for name in ("length_scale", "variance"):
        p1, p2 = f"line_1.v.kernel.{name}", f"line_2.v.kernel.{name}"
        np.testing.assert_allclose(g_tied[p1], g_untied[p1] + g_untied[p2], rtol=RTOL)
        assert not jnp.allclose(g_untied[p2], 0.0)  # both uses contribute


# ---- C4: posterior_summary ----


def capture(fn):
    console = get_console()
    with console.capture() as cap:
        fn()
    return cap.get()


def test_posterior_summary_from_stacked_dict():
    prior = LogNormal(2.0, 0.5)
    model = ShareModule(
        TwoParams(a=Parameter(1.0, prior=prior), b=Parameter([0.0, 0.0], prior=Laplace(1.0)))
    )
    latent = LatentSpace(model)
    stacked = {"a": jnp.array([[-1.0], [0.0], [1.0], [2.0]]), "b": jnp.zeros((4, 2))}
    out = {}
    text = capture(lambda: out.update(latent.posterior_summary(stacked)))
    vals = prior.transform(stacked["a"])
    np.testing.assert_allclose(out["a"]["mean"], vals.mean(0), rtol=1e-12)
    np.testing.assert_allclose(out["a"]["std"], vals.std(0), rtol=1e-12)
    np.testing.assert_array_equal(out["b"]["mean"], 0.0)
    assert "Posterior Summary" in text and "LogNormal(2, 0.5)" in text and "4 samples" in text


def test_posterior_summary_accepts_nifty_samples(tiny_gp_model):
    jft = pytest.importorskip("nifty.re")
    latent = LatentSpace(tiny_gp_model)
    pos = latent.sample_prior_latent(random.PRNGKey(0))
    residuals = jax.tree.map(lambda *x: jnp.stack(x), *make_xi_points(latent, n=4))
    samples = jft.Samples(pos=jft.Vector(pos), samples=jft.Vector(residuals))
    out = {}
    capture(lambda: out.update(latent.posterior_summary(samples)))
    stacked = {p: pos[p][None] + residuals[p] for p in latent.paths}
    for p in latent.paths:
        vals = jax.vmap(latent.priors[p].transform)(stacked[p])
        np.testing.assert_allclose(out[p]["mean"], vals.mean(0), rtol=1e-12)


# ---- Regression from the stage 7 code review: values outside the support ----


def test_from_model_value_outside_support_falls_back_with_warning():
    model = ShareModule(
        TwoParams(
            a=Parameter(0.0, prior=LogNormal(1.0, 0.5)),  # outside LogNormal's support
            b=Parameter(0.3, prior=Normal(0.0, 1.0)),
        )
    )
    latent = LatentSpace(model)
    with pytest.warns(UserWarning, match="outside the prior's support: \\['a'\\]"):
        ξ = latent.from_model(model)
    np.testing.assert_array_equal(ξ["a"], 0.0)
    np.testing.assert_allclose(ξ["b"], 0.3, rtol=RTOL)
    assert list(ξ) == latent.paths
    with pytest.raises(ValueError, match="outside"):
        latent.from_model(model, fallback="raise")
