"""test_domain.py - Domain, and FourierGP / FourierBasis in the data's own units."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from spectracles import (
    Domain,
    FourierGP,
    Matern32,
    Parameter,
    SpatialDataGeneric,
    build_model,
)
from spectracles.model.spatial import DEFAULT_PADDING, FourierBasis


def points(n=200, half=3.0, scale=1.0, centre=(0.0, 0.0), seed=0):
    xy = random.uniform(random.PRNGKey(seed), (2, n), minval=-half, maxval=half)
    return SpatialDataGeneric(
        x=centre[0] + scale * xy[0], y=centre[1] + scale * xy[1], idx=jnp.arange(n)
    )


def gp(length_scale=0.5, seed=1, **kwargs):
    kernel = Matern32(length_scale=Parameter(length_scale), variance=Parameter(1.0))
    coeffs = Parameter(random.normal(random.PRNGKey(seed), (21, 21)))
    return build_model(FourierGP, n_modes=(21, 21), kernel=kernel, coefficients=coeffs, **kwargs)


class TestDomain:
    def test_values_are_plain_floats(self):
        d = Domain(centre=(1, 2), extent=3)
        assert d.centre == (1.0, 2.0) and isinstance(d.extent, float)

    def test_hashable_and_equal_by_value(self):
        assert Domain((0, 0), 1.0) == Domain((0.0, 0.0), 1)
        assert hash(Domain((0, 0), 1.0)) == hash(Domain((0.0, 0.0), 1))

    @pytest.mark.parametrize(
        "centre, extent",
        [((0, 0), 0.0), ((0, 0), -1.0), ((0, 0), np.inf), ((0, np.nan), 1.0), ((0, 0, 0), 1.0)],
    )
    def test_invalid(self, centre, extent):
        with pytest.raises(ValueError):
            Domain(centre, extent)

    def test_centre_must_be_numbers(self):
        with pytest.raises(TypeError):
            Domain(1.0, 1.0)  # type: ignore[arg-type]

    def test_from_points(self):
        d = Domain.from_points(np.array([0.0, 4.0, 1.0]), np.array([10.0, 11.0, 12.0]))
        assert d.centre == (2.0, 11.0) and d.extent == 4.0

    def test_from_points_contains_every_point(self):
        data = points(scale=7.0, centre=(100.0, -50.0))
        d = Domain.from_points(data.x, data.y)
        half = d.extent / 2 * (1 + 1e-12)
        assert np.all(np.abs(np.asarray(data.x) - d.centre[0]) <= half)
        assert np.all(np.abs(np.asarray(data.y) - d.centre[1]) <= half)

    @pytest.mark.parametrize("x, y", [([], []), ([1.0, 2.0], [1.0]), ([1.0, 1.0], [2.0, 2.0])])
    def test_from_points_invalid(self, x, y):
        with pytest.raises(ValueError):
            Domain.from_points(np.array(x), np.array(y))

    def test_box_size(self):
        assert Domain((0, 0), 10.0).box_size(0.15) == pytest.approx(13.0)


class TestFourierGPInDataUnits:
    def test_without_a_domain_nothing_changes(self):
        # A domain whose box is exactly [-π, π] gives the same field as no domain
        data = points()
        plain = gp()
        boxed = gp(domain=Domain((0.0, 0.0), 2 * np.pi), padding=0.0)
        assert plain.padding is None and plain.domain is None
        assert jnp.array_equal(plain(data), boxed(data))

    def test_length_scale_is_in_the_coordinates_units(self):
        # Coordinates and ℓ scaled by the same factor, and shifted: the same field
        data_unit, a, c = points(), 3600.0, (500.0, -200.0)
        data_arcsec = points(scale=a, centre=c)
        unit = gp(domain=Domain((0.0, 0.0), 6.0), padding=0.2)
        arcsec = gp(length_scale=0.5 * a, domain=Domain(c, 6.0 * a), padding=0.2)
        assert jnp.allclose(unit(data_unit), arcsec(data_arcsec), rtol=1e-9, atol=1e-12)

    def test_default_padding(self):
        assert gp(domain=Domain((0, 0), 6.0)).padding == DEFAULT_PADDING

    def test_padding_needs_a_domain(self):
        with pytest.raises(ValueError, match="needs a domain"):
            gp(padding=0.2)

    def test_domain_must_be_a_domain(self):
        with pytest.raises(TypeError):
            gp(domain=(0.0, 0.0, 6.0))

    def test_negative_padding(self):
        with pytest.raises(ValueError):
            gp(domain=Domain((0, 0), 6.0), padding=-0.1)

    def test_points_in_the_padding_are_fine(self):
        model = gp(domain=Domain((0.0, 0.0), 6.0), padding=0.2)  # box side 8.4
        model(points(half=4.1))

    def test_points_outside_the_box_raise(self):
        model = gp(domain=Domain((0.0, 0.0), 6.0), padding=0.2)
        with pytest.raises(ValueError, match="outside the GP's periodic box"):
            model(points(half=4.5))

    def test_points_outside_without_a_domain_raise(self):
        with pytest.raises(ValueError, match="outside the GP's periodic box"):
            gp()(points(half=3.3))

    def test_checked_inside_jit_when_the_data_are_closed_over(self):
        model, far = gp(domain=Domain((0.0, 0.0), 6.0)), points(half=10.0)
        with pytest.raises(ValueError, match="outside"):
            eqx.filter_jit(lambda m: m(far))(model)

    def test_not_checked_when_the_data_are_traced(self):
        model, far = gp(domain=Domain((0.0, 0.0), 6.0)), points(half=10.0)
        jax.jit(lambda d: model(d))(far)  # no error: positions are traced here

    def test_nufft_tol(self):
        data = points()
        loose = gp(nufft_tol=1e-4)
        assert loose.nufft_tol == 1e-4
        assert jnp.allclose(gp()(data), loose(data), atol=1e-3)
        assert not jnp.array_equal(gp()(data), loose(data))

    def test_settings_are_static(self):
        a, b = gp(domain=Domain((0, 0), 6.0)), gp(domain=Domain((0, 0), 7.0))
        assert jax.tree.structure(a) != jax.tree.structure(b)

    def test_one_domain_shared_by_two_gps(self):
        class Two(eqx.Module):
            a: FourierGP
            b: FourierGP

            def __call__(self, d):
                return self.a(d) + self.b(d)

        domain = Domain((0.0, 0.0), 6.0)
        k = lambda: Matern32(Parameter(0.5), Parameter(1.0))
        model = build_model(
            Two,
            a=FourierGP((11, 11), k(), domain=domain, padding=0.1),
            b=FourierGP((11, 11), k(), domain=domain, padding=0.5),
        )
        assert model.a.domain is model.b.domain
        assert model.a.padding != model.b.padding
        model(points())


class TestFourierBasisInDataUnits:
    def test_same_mapping_as_the_gp(self):
        data, a = points(), 60.0
        unit = build_model(FourierBasis, n_modes=(11, 11))
        scaled = build_model(
            FourierBasis, n_modes=(11, 11), domain=Domain((0.0, 0.0), 2 * np.pi * a), padding=0.0
        )
        scaled_data = points(scale=a)
        assert jnp.allclose(unit(data), scaled(scaled_data), atol=1e-12)


class TestModesChosenForYou:
    DOMAIN = Domain((0.0, 0.0), 6.0)

    def kernel(self, prior=True):
        from spectracles.priors import LogNormal

        ls = Parameter(prior=LogNormal(0.5, 0.1)) if prior else Parameter(0.5)
        return Matern32(length_scale=ls, variance=Parameter(1.0))

    def test_chosen_for_the_tolerance(self):
        from spectracles.discretisation import discretisation_error

        gp = FourierGP(self.kernel(), domain=self.DOMAIN, tol=1e-3)
        tol, lo, hi = gp.design
        assert tol == 1e-3 and lo < 0.5 < hi
        m = gp.n_modes[0]
        assert gp.n_modes == (m, m)
        assert discretisation_error(gp, length_scale=lo).truncation <= tol
        assert discretisation_error(gp, length_scale=lo, n_modes=m - 2).truncation > tol
        assert discretisation_error(gp, length_scale=hi).aliasing <= tol

    def test_stricter_tolerance_needs_more(self):
        loose = FourierGP(self.kernel(), domain=self.DOMAIN, tol=1e-2)
        strict = FourierGP(self.kernel(), domain=self.DOMAIN, tol=1e-4)
        assert strict.n_modes[0] > loose.n_modes[0] and strict.padding > loose.padding

    def test_length_scale_range(self):
        gp = FourierGP(self.kernel(), domain=self.DOMAIN, length_scale=(0.3, 1.0))
        assert gp.design[1:] == (0.3, 1.0)

    def test_fixed_length_scale(self):
        gp = FourierGP(self.kernel(prior=False), domain=self.DOMAIN)
        assert gp.design[1:] == (0.5, 0.5)

    def test_given_padding_is_kept(self):
        gp = FourierGP(self.kernel(), domain=self.DOMAIN, padding=0.5)
        assert gp.padding == 0.5

    def test_too_little_padding_warns(self):
        with pytest.warns(UserWarning, match="aliasing error"):
            FourierGP(self.kernel(), domain=self.DOMAIN, padding=0.0)

    def test_by_hand(self):
        gp = FourierGP(self.kernel(), domain=self.DOMAIN, n_modes=(21, 21))
        assert gp.n_modes == (21, 21) and gp.padding == DEFAULT_PADDING and gp.design is None

    def test_n_modes_needed_without_a_domain(self):
        with pytest.raises(ValueError, match="Without a domain, n_modes is needed"):
            FourierGP(self.kernel())

    def test_tol_or_n_modes_not_both(self):
        with pytest.raises(ValueError, match="not both"):
            FourierGP(self.kernel(), domain=self.DOMAIN, n_modes=(21, 21), tol=1e-3)

    def test_too_short_a_length_scale(self):
        with pytest.raises(ValueError, match="narrow the prior"):
            FourierGP(self.kernel(), domain=Domain((0.0, 0.0), 6000.0))

    def test_old_signature_still_works(self):
        with pytest.warns(DeprecationWarning, match="FourierGP\\(n_modes, kernel\\)"):
            old = FourierGP((21, 21), self.kernel(prior=False))
        new = FourierGP(self.kernel(prior=False), n_modes=(21, 21))
        data = points()
        assert old.n_modes == new.n_modes
        assert jnp.array_equal(old(data), new(data))
