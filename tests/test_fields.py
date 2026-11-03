"""test_fields.py - FourierGP's mean and link, and the field combinators."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from spectracles import (
    Domain,
    FieldFromRatio,
    FieldPlusScatter,
    FourierGP,
    Matern32,
    Parameter,
    PerSpaxel,
    SpatialDataGeneric,
    build_model,
)
from spectracles.discretisation import discretisation_error, realised_covariance
from spectracles.inference import LatentSpace
from spectracles.priors import LogNormal, Normal

N = 60


def data(seed=0):
    xy = random.uniform(random.PRNGKey(seed), (2, N), minval=-3.0, maxval=3.0)
    return SpatialDataGeneric(x=xy[0], y=xy[1], idx=jnp.arange(N))


def kernel(ls=0.6, var=1.3):
    return Matern32(length_scale=Parameter(ls), variance=Parameter(var))


def gp(seed=1, **kwargs):
    coeffs = Parameter(random.normal(random.PRNGKey(seed), (21, 21)))
    return FourierGP(kernel(), n_modes=(21, 21), coefficients=coeffs, **kwargs)


class TestMean:
    def test_no_mean_by_default(self):
        assert gp().mean is None

    def test_mean_is_added(self):
        d = data()
        flat = FourierGP(
            kernel(),
            n_modes=(21, 21),
            coefficients=Parameter(jnp.zeros((21, 21))),
            mean=Parameter(2.5),
        )
        assert jnp.allclose(flat(d), 2.5)

    def test_the_constant_mode_is_dropped(self):
        # With a mean, the k = 0 coefficient has no effect: the mean alone sets the constant
        d, centre = data(), (10, 10)
        a = gp(mean=Parameter(0.0))
        b = eqx.tree_at(lambda g: g.coefficients.val, a, a.coefficients.val.at[centre].add(5.0))
        assert jnp.allclose(a(d), b(d), atol=1e-12)
        # ...but without a mean it does
        c = gp()
        c2 = eqx.tree_at(lambda g: g.coefficients.val, c, c.coefficients.val.at[centre].add(5.0))
        assert not jnp.allclose(c(d), c2(d))

    def test_variance_is_still_the_pointwise_variance(self):
        for g in (gp(), gp(mean=Parameter(0.0))):
            assert np.isclose(realised_covariance(g, 0.0, 0.0), 1.3)

    def test_realised_covariance_with_a_mean_matches_the_model(self):
        g, d = gp(mean=Parameter(0.0), domain=Domain((0.0, 0.0), 6.0), padding=0.2), data()

        def field(c):
            return eqx.tree_at(lambda m: m.coefficients.val, g, c)(d)

        J = jax.jacfwd(field)(g.coefficients.val).reshape(N, -1)
        realised = realised_covariance(g, d.x[:, None] - d.x[None, :], d.y[:, None] - d.y[None, :])
        assert np.allclose(J @ J.T, realised, atol=1e-9)

    def test_a_mean_with_a_prior_is_one_more_latent(self):
        class Field(eqx.Module):
            f: FourierGP

            def __call__(self, d):
                return self.f(d)

        k = Matern32(Parameter(prior=LogNormal(0.6, 0.1)), Parameter(prior=LogNormal(1.0, 0.3)))
        model = build_model(
            Field, f=FourierGP(k, n_modes=(11, 11), mean=Parameter(prior=Normal(-1.0, 0.5)))
        )
        assert "f.mean" in LatentSpace(model).paths

    def test_diagnostic_knows_about_the_mean(self):
        with_mean = gp(mean=Parameter(0.0), domain=Domain((0.0, 0.0), 6.0))
        e = discretisation_error(with_mean)
        assert e == discretisation_error(
            with_mean.kernel,
            domain=Domain((0.0, 0.0), 6.0),
            n_modes=21,
            padding=0.15,
            has_mean=True,
        )
        assert np.isfinite(e.truncation) and e.truncation < 1.0

    def test_mean_must_be_a_single_parameter(self):
        with pytest.raises(TypeError, match="mean must be a Parameter"):
            gp(mean=1.0)
        with pytest.raises(ValueError, match="single value"):
            gp(mean=Parameter(dims=3))

    def test_zeromode_is_the_old_name(self):
        with pytest.warns(DeprecationWarning, match="zeromode is deprecated"):
            g = gp(zeromode=Parameter(1.0))
        assert g.mean is not None
        with pytest.warns(DeprecationWarning):
            assert g.zeromode is g.mean
        with pytest.raises(TypeError, match="not both"):
            gp(mean=Parameter(1.0), zeromode=Parameter(1.0))


class TestLink:
    def test_exp(self):
        d = data()
        assert jnp.allclose(gp(link="exp")(d), jnp.exp(gp()(d)))

    def test_softplus(self):
        d = data()
        assert jnp.allclose(gp(link="softplus", lower=0.1)(d), 0.1 + jax.nn.softplus(gp()(d)))

    def test_link_after_the_mean(self):
        d = data()
        assert jnp.allclose(
            gp(link="exp", mean=Parameter(0.7))(d), jnp.exp(gp(mean=Parameter(0.7))(d))
        )

    def test_invalid(self):
        with pytest.raises(ValueError, match="link must be one of"):
            gp(link="log")
        with pytest.raises(ValueError, match="lower only applies"):
            gp(link="exp", lower=0.1)


class TestCombinators:
    def test_field_from_ratio(self):
        d = data()
        base, log_r = gp(seed=1, link="exp"), gp(seed=2, mean=Parameter(-0.5))
        f = FieldFromRatio(base=base, log10_ratio=log_r)
        assert jnp.allclose(f.ratio(d), 10 ** log_r(d))
        assert jnp.allclose(f(d), base(d) * 10 ** log_r(d))

    def test_ratio_shares_its_base(self):
        class Lines(eqx.Module):
            halpha: FourierGP
            nii: FieldFromRatio

            def __call__(self, d):
                return self.halpha(d) + self.nii(d)

        halpha = gp(seed=1, link="exp")
        model = build_model(
            Lines, halpha=halpha, nii=FieldFromRatio(base=halpha, log10_ratio=gp(seed=2))
        )
        assert "halpha" in model.get_sharing_summary(level="component")
        model(data())

    def test_field_plus_scatter(self):
        d = data()
        scatter = PerSpaxel(N, Parameter(jnp.linspace(0, 1, N)))
        f = FieldPlusScatter(field=gp(), scatter=scatter)
        assert jnp.allclose(f(d), gp()(d) + jnp.linspace(0, 1, N))

    def test_types(self):
        with pytest.raises(TypeError, match="base must be a spatial model"):
            FieldFromRatio(base=1.0, log10_ratio=gp())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="scatter must be a PerSpaxel"):
            FieldPlusScatter(field=gp(), scatter=gp())  # type: ignore[arg-type]
