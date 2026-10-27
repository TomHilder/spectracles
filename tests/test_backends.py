"""test_backends.py - The same problem across backends (proposal_tests.md section F).

No inference is run. Each test fixes one Posterior, builds what each backend
(NIFTy MGVI/geoVI, NIFTy's NUTS, BlackJAX NUTS, optax MAP) would see, and checks
that it's the same coordinates, log-density, gradients, forward model and metric.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from spectracles import Parameter, ShareModule, fixed_in, free_after, free_in, free_until
from spectracles.inference import LatentSpace
from spectracles.nifty import stages, to_nifty
from spectracles.optimise.schedule_builder import _resolve_free_phases, init_normal
from spectracles.priors import Normal

from .conftest import make_xi_points

# Tolerances: one decade looser than what these checks achieve on CPU float64
# (rtol 1e-14 with atol 1e-12 for near-zero components; probed in P1.1 stage 7)
RTOL = 1e-13
ATOL = 1e-11


def jft():
    return pytest.importorskip("nifty.re")


def vec(ξ):
    return jft().Vector(ξ)


def tree(x):
    return x.tree if hasattr(x, "tree") else x


def masked_jvp(post, ξ, t):
    """J t of the compressed forward model (what NIFTy's likelihood sees)."""

    def f(x):
        return post.likelihood.compress(post.predict(x))

    return jax.jvp(f, (ξ,), (t,))


def masked_vjp(post, ξ, u):
    def f(x):
        return post.likelihood.compress(post.predict(x))

    _, pullback = jax.vjp(f, ξ)
    return pullback(u)[0]


# ---- F1: forward model ----


def test_nifty_forward_equals_spectracles_forward(line_posterior):
    post = line_posterior
    lh = to_nifty(post)
    assert set(tree(lh.domain)) == set(post.latent.paths)
    for p in post.latent.paths:
        assert tree(lh.domain)[p].shape == post.latent.shapes[p].shape
    for ξ in make_xi_points(post.latent):
        np.testing.assert_array_equal(
            lh.forward(vec(ξ)), post.likelihood.compress(post.predict(ξ))
        )


# ---- F2, F3: NIFTy energy vs our log-density ----


def test_nifty_energy_matches_log_density_up_to_constant(line_posterior):
    post = line_posterior
    lh = to_nifty(post)
    ξs = make_xi_points(post.latent)
    # -E = -(energy + ½|ξ|²) should equal log_density + const
    neg_E = [-(lh(vec(ξ)) + 0.5 * sum(jnp.sum(v**2) for v in ξ.values())) for ξ in ξs]
    ours = [post.log_density(ξ) for ξ in ξs]
    for i in range(len(ξs)):
        for j in range(i + 1, len(ξs)):
            np.testing.assert_allclose(neg_E[i] - neg_E[j], ours[i] - ours[j], rtol=RTOL)


def test_nifty_gradient_matches(line_posterior):
    post = line_posterior
    lh = to_nifty(post)
    for ξ in make_xi_points(post.latent):
        g_nifty = tree(jax.grad(lambda x: lh(x))(vec(ξ)))
        g_ours = jax.grad(post.log_likelihood)(ξ)
        for p in post.latent.paths:
            np.testing.assert_allclose(-g_nifty[p], g_ours[p], rtol=RTOL, atol=ATOL)


# ---- F4: NIFTy's own NUTS target ----


def test_nifty_blackjax_target_equals_ours(line_posterior):
    jft()
    from nifty.re.blackjax import LogDensity

    post = line_posterior
    target = LogDensity(to_nifty(post))
    ξs = make_xi_points(post.latent)
    theirs = [target(vec(ξ)) for ξ in ξs]
    ours = [post.log_density(ξ) for ξ in ξs]
    for i in range(1, len(ξs)):
        np.testing.assert_allclose(theirs[i] - theirs[0], ours[i] - ours[0], rtol=RTOL)
    for ξ in ξs:
        g_theirs = tree(jax.grad(target)(vec(ξ)))
        g_ours = jax.grad(post.log_density)(ξ)
        for p in post.latent.paths:
            np.testing.assert_allclose(g_theirs[p], g_ours[p], rtol=RTOL, atol=ATOL)


# ---- F5: BlackJAX accepts our target (no steps) ----


def test_blackjax_nuts_init_on_our_target(line_posterior):
    blackjax = pytest.importorskip("blackjax")
    post = line_posterior
    ξ = make_xi_points(post.latent, n=1)[0]
    nuts = blackjax.nuts(
        post.log_density, step_size=0.1, inverse_mass_matrix=jnp.ones(post.latent.size)
    )
    state = nuts.init(ξ)
    np.testing.assert_allclose(state.logdensity, post.log_density(ξ), rtol=1e-12)
    g = jax.grad(post.log_density)(ξ)
    for p in post.latent.paths:
        np.testing.assert_allclose(state.logdensity_grad[p], g[p], rtol=1e-12)


# ---- F6: the MAP objective is the same function ----


def test_map_objective_is_negative_log_density(line_posterior):
    post = line_posterior
    ξ = make_xi_points(post.latent, n=1)[0]

    def loss(x):  # what an optax MAP minimises
        return -post.log_density(x)

    v, g = jax.value_and_grad(loss)(ξ)
    v_ref, g_ref = post.value_and_grad(ξ)
    assert float(v) == -float(v_ref)
    for p in post.latent.paths:
        np.testing.assert_array_equal(g[p], -g_ref[p])


# ---- F7, F8: the metric MGVI and geoVI see ----


def test_nifty_metric_is_likelihood_fisher(line_posterior):
    post = line_posterior
    lh = to_nifty(post)
    std_inv = post.likelihood.std_inv
    for k, ξ in enumerate(make_xi_points(post.latent, n=3)):
        t = post.latent.sample(random.PRNGKey(100 + k))
        _, Jt = masked_jvp(post, ξ, t)
        ours = masked_vjp(post, ξ, std_inv**2 * Jt)
        theirs = tree(lh.metric(vec(ξ), vec(t)))
        for p in post.latent.paths:
            np.testing.assert_allclose(theirs[p], ours[p], rtol=RTOL, atol=ATOL)


def test_nifty_left_sqrt_metric(line_posterior):
    post = line_posterior
    lh = to_nifty(post)
    std_inv = post.likelihood.std_inv
    ξ = make_xi_points(post.latent, n=1)[0]
    u = random.normal(random.PRNGKey(5), (post.likelihood.n_used,))
    # The left square root maps a data-space vector u to Jᵀ σ⁻¹ u
    theirs = tree(lh.left_sqrt_metric(vec(ξ), u))
    ours = masked_vjp(post, ξ, std_inv * u)
    for p in post.latent.paths:
        np.testing.assert_allclose(theirs[p], ours[p], rtol=RTOL, atol=ATOL)
    # ...and composing it with σ⁻¹ J gives the metric
    t = post.latent.sample(random.PRNGKey(6))
    _, Jt = masked_jvp(post, ξ, t)
    composed = tree(lh.left_sqrt_metric(vec(ξ), std_inv * Jt))
    metric = tree(lh.metric(vec(ξ), vec(t)))
    for p in post.latent.paths:
        np.testing.assert_allclose(composed[p], metric[p], rtol=RTOL, atol=ATOL)


# ---- F9: linear-Gaussian identity ----


def test_linear_gaussian_hessian_equals_metric_plus_identity(linear_gaussian):
    post = linear_gaussian
    lh = to_nifty(post)
    (path,) = post.latent.paths
    shape = post.latent.shapes[path].shape
    n = int(np.prod(shape))
    ξ = make_xi_points(post.latent, n=1)[0]

    def f(flat):
        return post.log_density({path: flat.reshape(shape)})

    neg_hessian = -jax.hessian(f)(ξ[path].ravel())
    basis = jnp.eye(n)
    metric = jnp.stack(
        [tree(lh.metric(vec(ξ), vec({path: e.reshape(shape)})))[path].ravel() for e in basis]
    )
    np.testing.assert_allclose(neg_hessian, metric + jnp.eye(n), rtol=RTOL, atol=ATOL)
    # Exactly Gaussian: the Hessian doesn't depend on ξ
    other = make_xi_points(post.latent, n=2, seed=7)[1][path].ravel()
    np.testing.assert_allclose(-jax.hessian(f)(other), neg_hessian, rtol=RTOL, atol=ATOL)


# ---- F10, F11: path keys as NIFTy staging selectors ----


class Unicode(eqx.Module):
    vσ: Parameter
    plain: Parameter

    def __call__(self):
        return self.vσ.val + self.plain.val


def test_path_keys_are_valid_nifty_selectors(line_posterior):
    jft()
    from nifty.re.likelihood import _parse_point_estimates

    post = line_posterior
    lh = to_nifty(post)
    ξ = make_xi_points(post.latent, n=1)[0]
    keys = ("line_1.v.kernel.length_scale", "offset.tile_values")
    pe, liquid, frozen = _parse_point_estimates(keys, vec(ξ))
    assert {p for p, flag in tree(pe).items() if flag} == set(keys)
    assert len(frozen) == len(keys)
    # Freezing those keys gives a likelihood with the same energy
    frozen_lh, liquid = lh.freeze(primals=vec(ξ), point_estimates=keys)
    np.testing.assert_allclose(frozen_lh(liquid), lh(vec(ξ)), rtol=1e-12)


def test_unicode_path_keys_work_as_selectors():
    jft()
    from nifty.re.likelihood import _parse_point_estimates

    model = ShareModule(
        Unicode(
            vσ=Parameter(1.0, prior=Normal(0.0, 1.0)), plain=Parameter(1.0, prior=Normal(0.0, 1.0))
        )
    )
    latent = LatentSpace(model)
    assert latent.paths == ["vσ", "plain"]
    pe, _, _ = _parse_point_estimates(("vσ",), vec(latent.zeros()))
    assert tree(pe) == {"vσ": True, "plain": False}


def test_per_iteration_callable_is_resolved():
    jft()
    from nifty.re.optimize_kl import _getitem_at_nit

    config = {"constants": lambda i: ("a",) if i < 2 else ()}
    assert [_getitem_at_nit(config, "constants", i) for i in range(4)] == [("a",), ("a",), (), ()]


# ---- F12, F13: the stages helper ----

STAGE_SPEC = {
    "line_1.log_A.kernel.*": free_in(0, 2, 4),
    "line_1.v.kernel.length_scale": free_after(3),
    "offset.tile_values": free_until(1),
    "scatter.spaxel_values": fixed_in(0, 5),
}


def test_stages_follow_schedule_vocabulary(line_posterior):
    latent = line_posterior.latent
    st = stages(latent, STAGE_SPEC, n_iterations=6)
    expected_free = {
        "line_1.log_A.kernel.length_scale": {0, 2, 4},
        "line_1.log_A.kernel.variance": {0, 2, 4},
        "line_1.v.kernel.length_scale": {3, 4, 5},
        "offset.tile_values": {0, 1},
        "scatter.spaxel_values": {1, 2, 3, 4},
    }
    for i in range(6):
        c, pe = st["constants"](i), st["point_estimates"](i)
        assert c == pe
        assert set(c) == {p for p, free in expected_free.items() if i not in free}
        # Unmentioned parameters are never frozen
        assert "line_1.log_A.coefficients" not in c
    # The same vocabulary means the same thing as in build_schedule
    for pattern, spec in STAGE_SPEC.items():
        free = _resolve_free_phases(spec, 6)
        for p, exp in expected_free.items():
            if pattern == p or (pattern.endswith("*") and p.startswith(pattern[:-1])):
                assert free == exp


def test_stages_order_and_shared_paths(line_posterior):
    latent = line_posterior.latent
    # line_2.v.kernel.* is the same parameter as line_1.v.kernel.*
    st = stages(latent, {"line_2.v.kernel.variance": fixed_in(1)}, n_iterations=3)
    assert st["constants"](1) == ("line_1.v.kernel.variance",)
    assert st["constants"](0) == ()
    # Globs expand in latent order
    st = stages(latent, {"**.kernel.length_scale": fixed_in(0)}, n_iterations=2)
    assert st["constants"](0) == (
        "line_1.log_A.kernel.length_scale",
        "line_1.v.kernel.length_scale",
        "line_2.log_A.kernel.length_scale",
    )


def test_stages_continue_past_the_end(line_posterior):
    st = stages(line_posterior.latent, {"offset.tile_values": free_after(2)}, n_iterations=3)
    assert st["constants"](1) == ("offset.tile_values",)
    assert st["constants"](10) == ()


def test_stages_errors(line_posterior):
    latent = line_posterior.latent
    with pytest.raises(ValueError, match="did not match"):
        stages(latent, {"nope": free_in(0)}, n_iterations=2)
    with pytest.raises(ValueError, match="init"):
        stages(latent, {"offset.tile_values": free_in(0) | init_normal(0)}, n_iterations=2)
    with pytest.raises(ValueError, match="iteration"):
        stages(latent, {"offset.tile_values": free_in(5)}, n_iterations=3)
    with pytest.raises(TypeError):
        stages(latent, {"offset.tile_values": "fixed"}, n_iterations=3)  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        stages(latent, {}, n_iterations=0)


def test_stages_callables_work_with_nifty(line_posterior):
    jft()
    from nifty.re.likelihood import _parse_point_estimates
    from nifty.re.optimize_kl import _getitem_at_nit

    post = line_posterior
    st = stages(post.latent, STAGE_SPEC, n_iterations=6)
    lh = to_nifty(post)
    ξ = vec(make_xi_points(post.latent, n=1)[0])
    for i in range(6):
        keys = _getitem_at_nit(st, "constants", i)
        assert keys == st["constants"](i)  # NIFTy treats it as per-iteration
        if keys:
            _parse_point_estimates(keys, ξ)
            frozen_lh, liquid = lh.freeze(primals=ξ, point_estimates=keys)
            np.testing.assert_allclose(frozen_lh(liquid), lh(ξ), rtol=1e-12)
