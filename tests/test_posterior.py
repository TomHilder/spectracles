"""test_posterior.py - Posterior and likelihood (proposal_tests.md section E)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior

from .conftest import make_xi_points

RTOL = 1e-6


# ---- E1: prior only ----


def test_log_prior_is_minus_half_xi_squared(line_posterior):
    for ξ in make_xi_points(line_posterior.latent):
        expected = -0.5 * sum(float(jnp.sum(v**2)) for v in ξ.values())
        assert float(line_posterior.log_prior(ξ)) == pytest.approx(expected, rel=1e-12)


def test_log_density_is_likelihood_plus_prior(line_posterior):
    for ξ in make_xi_points(line_posterior.latent):
        assert float(line_posterior.log_density(ξ)) == pytest.approx(
            float(line_posterior.log_likelihood(ξ) + line_posterior.log_prior(ξ)), rel=1e-12
        )


# ---- E2: Gaussian log-likelihood ----


def test_gaussian_log_likelihood_matches_scipy():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 5))
    ivar = rng.uniform(0.5, 2.0, size=(4, 5))
    pred = rng.normal(size=(4, 5))
    ref = stats.norm.logpdf(data, loc=pred, scale=1 / np.sqrt(ivar)).sum()
    lh = GaussianLikelihood(data, ivar)
    assert float(lh.log_likelihood(jnp.asarray(pred))) == pytest.approx(ref, rel=1e-12)
    assert lh.n_used == 20


def test_prediction_shape_checked():
    lh = GaussianLikelihood(np.zeros((3, 4)), np.ones((3, 4)))
    with pytest.raises(ValueError, match="shape"):
        lh.log_likelihood(jnp.zeros((4, 3)))


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        GaussianLikelihood(np.zeros(3), np.ones(4))
    with pytest.raises(ValueError):
        GaussianLikelihood(np.zeros(3), np.ones(3), mask=np.ones(4, bool))


# ---- E3: the mask is honoured ----


def test_masked_pixels_do_not_change_the_likelihood():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(6, 7))
    ivar = rng.uniform(0.5, 2.0, size=(6, 7))
    mask = rng.random((6, 7)) > 0.3
    pred = jnp.asarray(rng.normal(size=(6, 7)))
    ref = GaussianLikelihood(data, ivar, mask).log_likelihood(pred)
    for fill in (np.nan, 0.0, 1e30):
        d, iv = data.copy(), ivar.copy()
        d[~mask] = fill
        iv[~mask] = fill
        assert float(GaussianLikelihood(d, iv, mask).log_likelihood(pred)) == float(ref)
    # Also equal to the likelihood of only the unmasked pixels
    only = stats.norm.logpdf(data[mask], loc=np.asarray(pred)[mask], scale=1 / np.sqrt(ivar[mask]))
    assert float(ref) == pytest.approx(only.sum(), rel=1e-12)


def test_non_finite_or_nonpositive_ivar_pixels_are_masked():
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    ivar = np.array([1.0, 1.0, np.inf, 0.0, 2.0])
    lh = GaussianLikelihood(data, ivar)
    np.testing.assert_array_equal(lh.index, [0, 4])


def test_value_and_gradient_finite_with_nan_at_masked_pixels(line_posterior):
    # The fixture really has masked (NaN) pixels
    assert line_posterior.likelihood.n_used < 12 * 30
    for ξ in make_xi_points(line_posterior.latent):
        value, grad = line_posterior.value_and_grad(ξ)
        assert jnp.isfinite(value)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad.values())


# ---- E4: the forward model is used as given ----


def test_predict_is_forward_of_to_model(line_posterior):
    for ξ in make_xi_points(line_posterior.latent):
        np.testing.assert_array_equal(
            line_posterior.predict(ξ), line_posterior.forward(line_posterior.latent.to_model(ξ))
        )


# ---- E5: gradients vs finite differences ----


def test_gradient_matches_finite_differences(line_posterior):
    post = line_posterior
    ξ = make_xi_points(post.latent, n=1)[0]
    grad = jax.grad(post.log_density)(ξ)
    keys = random.split(random.PRNGKey(4), 10)
    h = 1e-5
    for k in keys:
        d = post.latent.sample(k)
        plus = {p: ξ[p] + h * d[p] for p in ξ}
        minus = {p: ξ[p] - h * d[p] for p in ξ}
        fd = (post.log_density(plus) - post.log_density(minus)) / (2 * h)
        directional = sum(jnp.sum(grad[p] * d[p]) for p in ξ)
        assert float(directional) == pytest.approx(float(fd), rel=RTOL, abs=1e-6)


# ---- Posterior construction ----


def test_posterior_requires_latent_space(tiny_gp_model):
    with pytest.raises(TypeError):
        Posterior(tiny_gp_model, lambda m: m, GaussianLikelihood(np.zeros(2), np.ones(2)))


def test_posterior_jit_and_vmap(linear_gaussian):
    post = linear_gaussian
    ξs = make_xi_points(post.latent)
    np.testing.assert_allclose(
        jax.jit(post.log_density)(ξs[0]), post.log_density(ξs[0]), rtol=RTOL
    )
    batch = jax.tree.map(lambda *x: jnp.stack(x), *ξs)
    np.testing.assert_allclose(
        jax.vmap(post.log_density)(batch), jnp.array([post.log_density(ξ) for ξ in ξs]), rtol=RTOL
    )
    assert isinstance(post.latent, LatentSpace)
    assert "Posterior(" in repr(post)
