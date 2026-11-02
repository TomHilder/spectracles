"""test_discretisation.py - the discretisation diagnostic and the mode/padding search."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from spectracles import Domain, FourierGP, Matern32, Parameter, SpatialDataGeneric
from spectracles.discretisation import (
    DiscretisationError,
    discretisation_error,
    realised_covariance,
    required_modes,
    required_padding,
)
from spectracles.model.kernels import Kernel, LogLogLinear

# The 19-tile Rosette footprint at padding 0.15 (P0.2 and P1.10's reference case)
ROSETTE = Domain((0.0, 0.0), 9963.2 / 1.3)


def matern(length_scale=1.0, variance=1.0):
    return Matern32(length_scale=Parameter(length_scale), variance=Parameter(variance))


class TestRealisedCovariance:
    @pytest.mark.parametrize(
        "domain, padding, ls, half, centre",
        [
            (None, None, 0.4, 3.0, (0.0, 0.0)),
            (Domain((100.0, -40.0), 600.0), 0.3, 40.0, 300.0, (100.0, -40.0)),
        ],
    )
    def test_matches_the_models_exact_covariance(self, domain, padding, ls, half, centre):
        # The field is linear in its coefficients, so its covariance is J Jᵀ exactly
        gp = FourierGP((31, 25), matern(ls, 1.7), domain=domain, padding=padding)
        xy = random.uniform(random.PRNGKey(0), (2, 40), minval=-half, maxval=half)
        x, y = xy[0] + centre[0], xy[1] + centre[1]
        data = SpatialDataGeneric(x=x, y=y, idx=jnp.arange(40))

        def field(c):
            return eqx.tree_at(lambda g: g.coefficients.val, gp, c)(data)

        J = jax.jacfwd(field)(gp.coefficients.val).reshape(40, -1)
        realised = realised_covariance(gp, x[:, None] - x[None, :], y[:, None] - y[None, :])
        assert np.allclose(J @ J.T, realised, atol=1e-9 * 1.7)


class TestErrors:
    def test_reproduces_the_rosette_mode_counts(self):
        # tolerance_modes.py (P1.10): ℓ_min = 37″ needs 237 modes at 1%, 439 at 0.1%
        k = matern(37.0)
        assert required_modes(k, ROSETTE, padding=0.15, tol=1e-2) == 237
        assert required_modes(k, ROSETTE, padding=0.15, tol=1e-3) == 439

    def test_search_gives_the_fewest_modes(self):
        k, tol = matern(40.0), 1e-2
        m = required_modes(k, ROSETTE, tol=tol)
        assert discretisation_error(k, domain=ROSETTE, n_modes=m, padding=0.15).truncation <= tol
        assert (
            discretisation_error(k, domain=ROSETTE, n_modes=m - 2, padding=0.15).truncation > tol
        )

    def test_truncation_falls_with_modes(self):
        k = matern(100.0)
        errs = [
            discretisation_error(k, domain=ROSETTE, n_modes=m, padding=0.15).truncation
            for m in (51, 101, 201)
        ]
        assert errs[0] > errs[1] > errs[2]

    def test_errors_depend_on_length_scale_over_the_domain_only(self):
        a, b = matern(0.3), matern(0.3 * 3600)
        ea = discretisation_error(a, domain=Domain((0, 0), 6.0), n_modes=61, padding=0.1)
        eb = discretisation_error(
            b, domain=Domain((5.0, 7.0), 6.0 * 3600), n_modes=61, padding=0.1
        )
        for f in ("truncation", "aliasing", "truncation_max", "tau"):
            assert np.isclose(getattr(ea, f), getattr(eb, f), rtol=1e-8, atol=1e-15), f

    def test_truncation_max_is_about_tau(self):
        e = discretisation_error(matern(37.0), domain=ROSETTE, n_modes=201, padding=0.15)
        assert np.isclose(e.truncation_max, e.tau, rtol=0.05)

    def test_padding_search(self):
        k, tol = matern(250.0), 1e-3
        p = required_padding(k, ROSETTE, tol=tol)
        assert discretisation_error(k, domain=ROSETTE, n_modes=51, padding=p).aliasing <= tol
        assert discretisation_error(k, domain=ROSETTE, n_modes=51, padding=p - 0.01).aliasing > tol

    def test_aliasing_falls_with_padding(self):
        k = matern(250.0)
        errs = [
            discretisation_error(k, domain=ROSETTE, n_modes=51, padding=p).aliasing
            for p in (0.0, 0.1, 0.3)
        ]
        assert errs[0] > errs[1] > errs[2]


class TestWhatIf:
    def gp(self):
        return FourierGP((101, 101), matern(120.0), domain=ROSETTE, padding=0.15)

    def test_a_gp_reads_its_own_settings(self):
        gp = self.gp()
        e = discretisation_error(gp)
        assert isinstance(e, DiscretisationError)
        assert e.n_modes == (101, 101) and e.padding == 0.15 and e.length_scale == 120.0
        assert e == discretisation_error(gp.kernel, domain=ROSETTE, n_modes=101, padding=0.15)

    def test_overrides(self):
        gp = self.gp()
        base = discretisation_error(gp)
        assert discretisation_error(gp, n_modes=51).truncation > base.truncation
        assert discretisation_error(gp, padding=0.05).aliasing > base.aliasing
        assert discretisation_error(gp, length_scale=60.0).truncation > base.truncation
        assert discretisation_error(gp, length_scale=60.0).length_scale == 60.0

    def test_a_curve_over_length_scales(self):
        ls = np.geomspace(60.0, 600.0, 5)
        e = discretisation_error(self.gp(), length_scale=ls)
        assert e.truncation.shape == ls.shape and e.aliasing.shape == ls.shape
        assert np.all(np.diff(e.truncation) < 0) and np.all(np.diff(e.aliasing) > 0)
        single = discretisation_error(self.gp(), length_scale=ls[2])
        assert np.isclose(e.truncation[2], single.truncation)

    def test_without_a_domain(self):
        gp = FourierGP((31, 31), matern(0.5))
        e = discretisation_error(gp)
        assert e.padding == 0.0 and np.isfinite(e.truncation)


class TestInputs:
    def test_a_kernel_needs_n_modes(self):
        with pytest.raises(ValueError, match="n_modes"):
            discretisation_error(matern(), domain=ROSETTE)

    def test_not_a_gp_or_kernel(self):
        with pytest.raises(TypeError, match="FourierGP or a Kernel"):
            discretisation_error(1.0)  # type: ignore[arg-type]

    def test_kernel_without_a_continuous_form(self):
        k = LogLogLinear(m_0=Parameter(0.0), m_s=Parameter(-2.0))
        with pytest.raises(NotImplementedError, match="continuous form"):
            discretisation_error(k, domain=ROSETTE, n_modes=31)

    def test_length_scale_on_a_kernel_without_one(self):
        class UnitGaussian(Kernel):
            # A Gaussian kernel with a fixed length scale of 1: no length_scale field
            variance: Parameter

            def feature_weights(self, freqs):
                return jnp.exp(-0.25 * freqs**2)

            def spectral_density(self, k):
                return self.variance.val * jnp.exp(-0.5 * jnp.asarray(k) ** 2) / (2 * jnp.pi)

            def covariance(self, r):
                return self.variance.val * jnp.exp(-0.5 * jnp.asarray(r) ** 2)

        k = UnitGaussian(variance=Parameter(1.0))
        with pytest.raises(ValueError, match="no length_scale"):
            discretisation_error(k, domain=ROSETTE, n_modes=31, length_scale=1.0)

    def test_padding_without_a_domain(self):
        with pytest.raises(ValueError, match="needs a domain"):
            discretisation_error(matern(), n_modes=31, padding=0.2)

    def test_too_short_a_length_scale(self):
        with pytest.raises(ValueError, match="too short"):
            required_modes(matern(1.0), ROSETTE, tol=1e-3)
