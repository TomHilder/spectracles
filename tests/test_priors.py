"""test_priors.py - Prior transforms (proposal_tests.md section A)."""

import math
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from spectracles.priors import Laplace, LogNormal, Normal, Prior, StdNormal, Uniform

# Tolerances: one decade looser than what these checks achieve on CPU float64
# (rtol 1e-14 with atol 1e-12 for near-zero components; probed in P1.1 stage 7)
RTOL = 1e-13
ATOL = 1e-11

XI_GRID = jnp.linspace(-5.0, 5.0, 101)

ALL_PRIORS = [
    StdNormal(),
    Normal(0.0, 1.0),
    Normal(-2.5, 0.3),
    LogNormal(1.0, 0.5),
    LogNormal(0.3, 0.15),
    Laplace(1e-3),
    Laplace(2.0),
    Uniform(0.0, 1.0),
    Uniform(-3.0, 7.0),
]
INVERTIBLE = [p for p in ALL_PRIORS if p.has_inverse]


# ---- A1: reference formulas ----


@pytest.mark.parametrize("mean, std", [(0.0, 1.0), (-2.5, 0.3)])
def test_normal_formula(mean, std):
    np.testing.assert_allclose(
        Normal(mean, std).transform(XI_GRID), mean + std * XI_GRID, rtol=RTOL
    )


def test_std_normal_is_identity():
    np.testing.assert_array_equal(StdNormal().transform(XI_GRID), XI_GRID)


def test_lognormal_moment_matching_hand_values():
    # mean = std = 1: s² = log 2, m = -log(2)/2
    m, s = LogNormal(1.0, 1.0).log_moments
    assert s == pytest.approx(math.sqrt(math.log(2.0)), rel=1e-12)
    assert m == pytest.approx(-0.5 * math.log(2.0), rel=1e-12)


@pytest.mark.parametrize("mean, std", [(1.0, 0.5), (0.3, 0.15)])
def test_lognormal_formula(mean, std):
    s = math.sqrt(math.log1p((std / mean) ** 2))
    m = math.log(mean) - 0.5 * s**2
    np.testing.assert_allclose(
        LogNormal(mean, std).transform(XI_GRID), np.exp(m + s * np.asarray(XI_GRID)), rtol=RTOL
    )


@pytest.mark.parametrize("alpha", [1e-3, 2.0])
def test_laplace_is_quantile_of_normal_cdf(alpha):
    # Independent reference: the Laplace quantile function applied to Φ(ξ)
    ref = stats.laplace.ppf(stats.norm.cdf(np.asarray(XI_GRID)), loc=0.0, scale=alpha)
    # scipy's reference loses precision in the tails (achieved rtol 1e-10)
    np.testing.assert_allclose(
        Laplace(alpha).transform(XI_GRID), ref, rtol=1e-9, atol=1e-12 * alpha
    )


@pytest.mark.parametrize("low, high", [(0.0, 1.0), (-3.0, 7.0)])
def test_uniform_formula(low, high):
    ref = low + (high - low) * stats.norm.cdf(np.asarray(XI_GRID))
    np.testing.assert_allclose(Uniform(low, high).transform(XI_GRID), ref, rtol=RTOL)


# ---- A2: matches NIFTy ----


def _nifty_equivalent(prior):
    jft = pytest.importorskip("nifty.re")
    if isinstance(prior, StdNormal):
        return jft.NormalPrior(0.0, 1.0, shape=())
    if isinstance(prior, Normal):
        return jft.NormalPrior(prior.mean, prior.std, shape=())
    if isinstance(prior, LogNormal):
        return jft.LogNormalPrior(prior.mean, prior.std, shape=())
    if isinstance(prior, Laplace):
        return jft.LaplacePrior(prior.alpha, shape=())
    if isinstance(prior, Uniform):
        return jft.UniformPrior(prior.low, prior.high, shape=())
    raise AssertionError(prior)


@pytest.mark.parametrize("prior", ALL_PRIORS, ids=repr)
@pytest.mark.parametrize("shape", [(), (7,), (3, 4)])
def test_matches_nifty(prior, shape):
    nifty_prior = _nifty_equivalent(prior)
    ξ = random.normal(random.PRNGKey(0), shape) * 2.0
    np.testing.assert_allclose(prior.transform(ξ), nifty_prior(ξ), rtol=RTOL)


# ---- A3: inverses ----


def test_inverse_availability():
    assert {type(p) for p in INVERTIBLE} == {StdNormal, Normal, LogNormal}
    for prior in (Laplace(1.0), Uniform(0.0, 1.0)):
        assert not prior.has_inverse
        with pytest.raises(NotImplementedError):
            prior.inverse(jnp.zeros(3))


@pytest.mark.parametrize("prior", INVERTIBLE, ids=repr)
def test_inverse_round_trip(prior):
    ξ = jnp.linspace(-6.0, 6.0, 61)
    np.testing.assert_allclose(prior.inverse(prior.transform(ξ)), ξ, rtol=RTOL, atol=ATOL)
    x = prior.transform(ξ)
    np.testing.assert_allclose(prior.transform(prior.inverse(x)), x, rtol=RTOL)


@pytest.mark.parametrize(
    "prior", [p for p in INVERTIBLE if not isinstance(p, StdNormal)], ids=repr
)
def test_inverse_matches_nifty(prior):
    pytest.importorskip("nifty.re")
    from nifty.re.num import lognormal_invprior, normal_invprior

    x = prior.transform(jnp.linspace(-4.0, 4.0, 41))
    if isinstance(prior, Normal):
        ref = normal_invprior(prior.mean, prior.std)(x)
    else:
        ref = lognormal_invprior(prior.mean, prior.std)(x)
    np.testing.assert_allclose(prior.inverse(x), ref, rtol=RTOL, atol=ATOL)


# ---- A4: moments ----

N_DRAWS = 200_000


def _draws(prior, seed=1):
    return np.asarray(prior.sample(random.PRNGKey(seed), (N_DRAWS,)))


def _assert_mean_std(x, mean, std, n_sigma=5.0):
    se_mean = x.std() / np.sqrt(x.size)
    assert abs(x.mean() - mean) < n_sigma * se_mean
    # Standard error of the sample variance from the fourth central moment
    m4 = np.mean((x - x.mean()) ** 4)
    se_var = np.sqrt((m4 - x.var() ** 2) / x.size)
    assert abs(x.var() - std**2) < n_sigma * se_var


@pytest.mark.parametrize(
    "prior", [Normal(-2.5, 0.3), LogNormal(1.0, 0.5), LogNormal(0.3, 0.15)], ids=repr
)
def test_mean_and_std(prior):
    _assert_mean_std(_draws(prior), prior.mean, prior.std)


@pytest.mark.parametrize("low, high", [(0.0, 1.0), (-3.0, 7.0)])
def test_uniform_moments(low, high):
    x = _draws(Uniform(low, high))
    assert x.min() >= low and x.max() <= high
    _assert_mean_std(x, 0.5 * (low + high), (high - low) / np.sqrt(12.0))


@pytest.mark.parametrize("alpha", [1e-3, 2.0])
def test_laplace_scale(alpha):
    x = _draws(Laplace(alpha))
    # |x| is exponential with mean and std alpha
    assert abs(np.abs(x).mean() - alpha) < 5.0 * alpha / np.sqrt(N_DRAWS)
    assert abs(x.mean()) < 5.0 * np.sqrt(2.0) * alpha / np.sqrt(N_DRAWS)


# ---- A5: broadcasting and scalar-only hyperparameters ----


@pytest.mark.parametrize("prior", ALL_PRIORS, ids=repr)
def test_broadcasts_to_latent_shape(prior):
    assert prior.transform(jnp.zeros((3, 4))).shape == (3, 4)
    assert prior.sample(random.PRNGKey(0), (5,)).shape == (5,)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Normal(jnp.zeros(3), 1.0),
        lambda: Normal(0.0, [1.0, 2.0]),
        lambda: LogNormal(np.ones(2), 1.0),
        lambda: Laplace(jnp.ones((2, 2))),
        lambda: Uniform(0.0, np.array([1.0, 2.0])),
    ],
)
def test_array_hyperparameters_rejected(make):
    with pytest.raises(TypeError, match="scalar"):
        make()


def test_zero_d_arrays_and_ints_accepted():
    assert Normal(jnp.array(1.0), np.float32(2.0)) == Normal(1.0, 2.0)
    assert Uniform(0, 1) == Uniform(0.0, 1.0)


def test_bool_hyperparameter_rejected():
    with pytest.raises(TypeError):
        Normal(True, 1.0)


# ---- A6: bad arguments ----


@pytest.mark.parametrize(
    "make",
    [
        lambda: Normal(0.0, 0.0),
        lambda: Normal(0.0, -1.0),
        lambda: LogNormal(0.0, 1.0),
        lambda: LogNormal(-1.0, 1.0),
        lambda: LogNormal(1.0, 0.0),
        lambda: Laplace(0.0),
        lambda: Laplace(-1.0),
        lambda: Uniform(1.0, 1.0),
        lambda: Uniform(2.0, 1.0),
    ],
)
def test_bad_arguments_raise(make):
    with pytest.raises(ValueError):
        make()


# ---- A7: hashable, equal by value ----


def test_hashable_and_equal_by_value():
    assert LogNormal(1, 2) == LogNormal(1.0, 2.0)
    assert hash(LogNormal(1, 2)) == hash(LogNormal(1.0, 2.0))
    assert LogNormal(1.0, 2.0) != LogNormal(1.0, 2.5)
    assert Normal(0.0, 1.0) != StdNormal()
    assert StdNormal() == StdNormal() and hash(StdNormal()) == hash(StdNormal())
    assert len({Normal(0, 1), Normal(0.0, 1.0), Laplace(1.0)}) == 2
    with pytest.raises(FrozenInstanceError):
        Normal(0.0, 1.0).mean = 2.0  # type: ignore[misc]


def test_is_prior_and_repr():
    for prior in ALL_PRIORS:
        assert isinstance(prior, Prior)
    assert repr(LogNormal(0.3, 0.15)) == "LogNormal(0.3, 0.15)"
    assert repr(Normal(0, 0.01)) == "Normal(0, 0.01)"
    assert repr(StdNormal()) == "StdNormal"
    assert repr(Laplace(1e-3)) == "Laplace(0.001)"
    assert repr(Uniform(-3, 7)) == "Uniform(-3, 7)"


# ---- A8: differentiable ----


@pytest.mark.parametrize("prior", ALL_PRIORS, ids=repr)
def test_gradient_finite(prior):
    grad = jax.grad(lambda ξ: jnp.sum(prior.transform(ξ)))(XI_GRID)
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.parametrize("alpha", [1e-3, 2.0])
def test_laplace_gradient_continuous_at_zero(alpha):
    # d/dξ [alpha log(2 Φ(ξ))] at 0 = alpha φ(0)/Φ(0) = alpha sqrt(2/π), same from both sides
    expected = alpha * math.sqrt(2.0 / math.pi)
    g = jax.grad(lambda ξ: Laplace(alpha).transform(ξ))
    for ξ0 in (0.0, -1e-8, 1e-8):
        assert float(g(ξ0)) == pytest.approx(expected, rel=1e-6)
