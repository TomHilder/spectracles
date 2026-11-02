"""test_posterior.py - Posterior and likelihood (proposal_tests.md section E)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior

from .conftest import make_xi_points

# Tolerances: one decade looser than what these checks achieve on CPU float64
# (rtol 1e-14 with atol 1e-12 for near-zero components; probed in P1.1 stage 7)
RTOL = 1e-13
ATOL = 1e-11


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


def test_masked_data_points_do_not_change_the_likelihood():
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
    # Also equal to the likelihood of only the unmasked data points
    only = stats.norm.logpdf(data[mask], loc=np.asarray(pred)[mask], scale=1 / np.sqrt(ivar[mask]))
    assert float(ref) == pytest.approx(only.sum(), rel=1e-12)


def test_non_finite_or_nonpositive_ivar_data_points_are_masked():
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    ivar = np.array([1.0, 1.0, np.inf, 0.0, 2.0])
    lh = GaussianLikelihood(data, ivar)
    np.testing.assert_array_equal(lh.index, [0, 4])


def test_value_and_gradient_finite_with_nan_at_masked_data_points(line_posterior):
    # The fixture really has masked (NaN) data points
    assert line_posterior.likelihood.n_used < 12 * 30
    for ξ in make_xi_points(line_posterior.latent):
        value, grad = line_posterior.value_and_grad(ξ)
        assert jnp.isfinite(value)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad.values())


# ---- E4: the forward model is used as given ----


def test_likelihood_must_be_a_likelihood(line_posterior):
    from spectracles.inference import GaussianLikelihood, Likelihood, Posterior

    assert isinstance(line_posterior.likelihood, Likelihood)
    assert issubclass(GaussianLikelihood, Likelihood)
    data = line_posterior.likelihood.data
    with pytest.raises(TypeError, match="likelihood must be a Likelihood"):
        Posterior(line_posterior.latent, line_posterior.forward, data)  # type: ignore[arg-type]


def test_likelihood_subclass_must_implement_the_interface():
    from spectracles.inference import Likelihood

    class Partial(Likelihood):
        def compress(self, pred):
            return pred

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_posterior_summary_calls_through(line_posterior):
    post = line_posterior
    stacked = post.sample_prior_latent(random.PRNGKey(12), n=10)
    a = post.summary(stacked, print_table=False)
    b = post.latent.posterior_summary(stacked, print_table=False)
    for p in post.latent.paths:
        np.testing.assert_array_equal(a[p]["mean"], b[p]["mean"])


def test_map_samples_of_forward_gives_predictions(line_posterior):
    post = line_posterior
    stacked = post.sample_prior_latent(random.PRNGKey(14), n=3)
    preds = post.map_samples(post.forward, stacked).values
    for i in range(3):
        np.testing.assert_allclose(preds[i], post.predict({p: v[i] for p, v in stacked.items()}))


def test_forward_must_be_a_function_not_the_model(line_posterior):
    from spectracles.inference import Posterior

    latent, likelihood = line_posterior.latent, line_posterior.likelihood
    with pytest.raises(TypeError, match="not the model itself"):
        Posterior(latent, latent.model, likelihood)
    with pytest.raises(TypeError, match="function from a model"):
        Posterior(latent, "forward", likelihood)  # type: ignore[arg-type]


def test_predict_is_forward_of_to_model(line_posterior):
    for ξ in make_xi_points(line_posterior.latent):
        np.testing.assert_array_equal(
            line_posterior.predict(ξ), line_posterior.forward(line_posterior.latent.to_model(ξ))
        )


def test_prior_draws_forwarded_and_no_bare_sample(line_posterior):
    post, key = line_posterior, random.PRNGKey(9)
    assert not hasattr(post, "sample") and not hasattr(post, "prior_sample")
    for p, v in post.sample_prior_params(key, n=3).items():
        np.testing.assert_array_equal(v, post.latent.sample_prior_params(key, n=3)[p])
    ξ = post.sample_prior_latent(key)
    np.testing.assert_array_equal(
        post.to_params(ξ)[post.latent.paths[0]], post.latent.to_params(ξ)[post.latent.paths[0]]
    )
    assert len(post.sample_prior_model(key, n=2)) == 2


# ---- E5: gradients vs finite differences ----


def test_gradient_matches_finite_differences(line_posterior):
    post = line_posterior
    ξ = make_xi_points(post.latent, n=1)[0]
    grad = jax.grad(post.log_density)(ξ)
    keys = random.split(random.PRNGKey(4), 10)
    h = 1e-5
    for k in keys:
        d = post.latent.sample_prior_latent(k)
        plus = {p: ξ[p] + h * d[p] for p in ξ}
        minus = {p: ξ[p] - h * d[p] for p in ξ}
        fd = (post.log_density(plus) - post.log_density(minus)) / (2 * h)
        directional = sum(jnp.sum(grad[p] * d[p]) for p in ξ)
        # Central differences with h = 1e-5: truncation error limits this to ~1e-6
        assert float(directional) == pytest.approx(float(fd), rel=1e-6, abs=1e-6)


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


# ---- ModelDraws ----


def test_model_draws_call_any_part(line_posterior):
    from spectracles.inference import Draws, ModelDraws

    from .priors_models import LAM, line_data

    post, data = line_posterior, line_data()
    stacked = post.sample_prior_latent(random.PRNGKey(17), n=4)
    models = post.draws(stacked, to="model")
    assert isinstance(models, ModelDraws) and isinstance(models.line_2.v, ModelDraws)

    def one(i):
        return post.latent.to_model({p: v[i] for p, v in stacked.items()}).for_evaluation()

    # The whole model, a component, and a component holding a shared kernel
    whole = models(LAM[0], data)
    v2 = models.line_2.v(data)  # line_2.v's kernel is shared with line_1.v
    assert isinstance(whole, Draws) and whole.sample_shape == (4,)
    for i in range(4):
        np.testing.assert_allclose(whole.values[i], one(i)(LAM[0], data))
        np.testing.assert_allclose(v2.values[i], one(i).line_2.v(data))
    np.testing.assert_allclose(v2.mean(), v2.values.mean(axis=0))
    # Methods work too
    lp = models.line_1.log_A.prior_logpdf()
    np.testing.assert_allclose(lp.values[0], one(0).line_1.log_A.prior_logpdf())


def test_model_draws_errors(line_posterior):
    post = line_posterior
    stacked = post.sample_prior_latent(random.PRNGKey(18), n=2)
    models = post.draws(stacked, to="model")
    with pytest.raises(AttributeError, match="line_1 has no attribute 'nope'"):
        _ = models.line_1.nope
    with pytest.raises(TypeError, match="not callable"):
        models.line_1.σ()
    with pytest.raises(AttributeError, match="read-only"):
        models.line_1 = None


def test_posterior_fix(line_posterior):
    path = "line_1.log_A.kernel.variance"
    fixed = line_posterior.fix({path: 0.7})
    assert path not in fixed.latent.paths and path in line_posterior.latent.paths
    assert fixed.likelihood is line_posterior.likelihood
    ξ = fixed.latent.sample_prior_latent(random.PRNGKey(3))
    assert jnp.isfinite(fixed.log_density(ξ))
    full = {**ξ, path: line_posterior.latent.priors[path].inverse(jnp.array([0.7]))}
    np.testing.assert_allclose(
        fixed.log_likelihood(ξ), line_posterior.log_likelihood(full), rtol=1e-5
    )
