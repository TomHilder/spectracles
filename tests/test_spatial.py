"""test_spatial.py - tests for the spectracles.model.spatial module."""

import jax.numpy as jnp
import pytest
from spectracles.model.data import SpatialDataGeneric
from spectracles.model.kernels import Matern12, Matern32, Matern52, SquaredExponential
from spectracles.model.parameter import Parameter
from spectracles.model.spatial import FourierGP, PerSpaxel, get_freqs, get_freqs_1D


class TestGetFreqs:
    def test_get_freqs_1d_odd(self):
        # Test with odd number of modes
        freqs = get_freqs_1D(5)
        assert freqs.shape == (5,)
        assert jnp.allclose(freqs, jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]))

    def test_get_freqs_1d_even(self):
        # Test with even number of modes
        freqs = get_freqs_1D(4)
        assert freqs.shape == (4,)
        assert jnp.allclose(freqs, jnp.array([-2.0, -1.0, 0.0, 1.0]))

    def test_get_freqs_1d_single(self):
        # Test with single mode
        freqs = get_freqs_1D(1)
        assert freqs.shape == (1,)
        assert jnp.allclose(freqs, jnp.array([0.0]))

    def test_get_freqs_2d(self):
        # Test 2D freqs with tuple input
        freqs = get_freqs((3, 3), n_dim=2)
        assert len(freqs) == 2  # Should return a list of two arrays
        assert freqs[0].shape == (3, 3)
        assert freqs[1].shape == (3, 3)

        # Check specific values for 3x3 grid
        expected_x = jnp.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        expected_y = jnp.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        assert jnp.allclose(freqs[0], expected_x)
        assert jnp.allclose(freqs[1], expected_y)

    def test_get_freqs_different_dims(self):
        # Test with different dimensions in each direction
        freqs = get_freqs((3, 5), n_dim=2)
        assert len(freqs) == 2
        assert freqs[0].shape == (3, 5)
        assert freqs[1].shape == (3, 5)

    def test_get_freqs_errors(self):
        # Test error when dimensions don't match
        with pytest.raises(ValueError):
            get_freqs((3,), n_dim=2)

        # Test error with invalid input type
        with pytest.raises(ValueError):
            get_freqs("invalid")

    def test_get_freqs_1d_wrapper(self):
        # Test 1D wrapper with int input
        freqs = get_freqs(5, n_dim=1)
        assert freqs.shape == (5,)
        assert jnp.allclose(freqs, jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]))


class TestFourierGP:
    def setup_method(self):
        # Common setup for FourierGP tests
        self.n_modes = (5, 7)
        self.kernel = Matern32(
            length_scale=Parameter(initial=1.0), variance=Parameter(initial=1.0)
        )
        self.data = SpatialDataGeneric(
            x=jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            y=jnp.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            idx=jnp.array([0, 1, 2, 3, 4]),
        )

    def test_initialization(self):
        # Test basic initialization
        gp = FourierGP(n_modes=self.n_modes, kernel=self.kernel)

        # Check attributes
        assert gp.n_modes == self.n_modes
        assert gp._freqs.shape == self.n_modes
        assert gp.kernel is self.kernel
        assert gp.coefficients.val.shape == self.n_modes

        # Check shape info
        p = self.n_modes[0] * self.n_modes[1]
        assert gp._shape_info == (p, p // 2, p)

    def test_initialization_with_coefficients(self):
        # Test initialization with provided coefficients
        coeffs = Parameter(initial=jnp.ones(self.n_modes))
        gp = FourierGP(n_modes=self.n_modes, kernel=self.kernel, coefficients=coeffs)

        assert gp.coefficients is coeffs
        assert jnp.allclose(gp.coefficients.val, jnp.ones(self.n_modes))

    def test_call(self):
        # Test function evaluation
        gp = FourierGP(n_modes=self.n_modes, kernel=self.kernel)
        out = gp(self.data)

        # Check output shape
        assert out.shape == (self.data.x.shape[0],)

        # Calling twice should give same result (no randomness)
        out2 = gp(self.data)
        assert jnp.allclose(out, out2)

    def test_different_kernels(self):
        # Test with different kernel types
        for KernelClass in [Matern12, Matern32, Matern52, SquaredExponential]:
            kernel = KernelClass(
                length_scale=Parameter(initial=1.0), variance=Parameter(initial=1.0)
            )
            gp = FourierGP(n_modes=self.n_modes, kernel=kernel)
            out = gp(self.data)
            assert out.shape == (self.data.x.shape[0],)

    def test_larger_dataset(self):
        # Test with larger dataset
        n = 100
        data = SpatialDataGeneric(
            x=jnp.linspace(0, 1, n), y=jnp.linspace(0, 1, n), idx=jnp.arange(n)
        )
        gp = FourierGP(n_modes=self.n_modes, kernel=self.kernel)
        out = gp(data)
        assert out.shape == (n,)


class TestFourierBasis:
    def setup_method(self):
        from spectracles.model.spatial import FourierBasis

        self.FourierBasis = FourierBasis
        self.n_modes = (5, 5)
        self.data = SpatialDataGeneric(
            x=jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            y=jnp.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            idx=jnp.arange(5),
        )

    def test_initialization(self):
        # Test basic initialization
        fb = self.FourierBasis(n_modes=self.n_modes)

        # Check attributes
        assert fb.n_modes == self.n_modes
        assert fb.coefficients.val.shape == self.n_modes
        assert fb._freqs.shape == self.n_modes

    def test_initialization_with_coefficients(self):
        # Test initialization with provided coefficients
        coeffs = Parameter(initial=jnp.ones(self.n_modes))
        fb = self.FourierBasis(n_modes=self.n_modes, coefficients=coeffs)

        assert jnp.allclose(fb.coefficients.val, jnp.ones(self.n_modes))

    def test_call(self):
        # Test function evaluation
        fb = self.FourierBasis(n_modes=self.n_modes)
        out = fb(self.data)

        # Check output shape
        assert out.shape == (self.data.x.shape[0],)

        # Calling twice should give same result (deterministic)
        out2 = fb(self.data)
        assert jnp.allclose(out, out2)

    def test_larger_dataset(self):
        # Test with larger dataset
        n = 100
        data = SpatialDataGeneric(
            x=jnp.linspace(0, 1, n), y=jnp.linspace(0, 1, n), idx=jnp.arange(n)
        )
        fb = self.FourierBasis(n_modes=self.n_modes)
        out = fb(data)
        assert out.shape == (n,)

    def test_different_n_modes(self):
        # Test with different number of modes
        for n_modes in [(3, 3), (7, 5), (5, 7)]:
            fb = self.FourierBasis(n_modes=n_modes)
            assert fb.n_modes == n_modes
            assert fb.coefficients.val.shape == n_modes

    def test_prior_logpdf_not_implemented(self):
        # Test that prior_logpdf raises NotImplementedError
        fb = self.FourierBasis(n_modes=self.n_modes)

        with pytest.raises(NotImplementedError):
            fb.prior_logpdf()


class TestPerSpaxel:
    def setup_method(self):
        # Common setup for PerSpaxel tests
        self.n_spaxels = 10
        self.data = SpatialDataGeneric(
            x=jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            y=jnp.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            idx=jnp.array([0, 1, 2, 3, 4]),
        )

    def test_initialization(self):
        # Test basic initialization
        ps = PerSpaxel(n_spaxels=self.n_spaxels)

        # Check attributes
        assert ps.spaxel_values.val.shape == (self.n_spaxels,)

    def test_initialization_with_values(self):
        # Test initialization with provided values
        values = Parameter(initial=jnp.ones(self.n_spaxels))
        ps = PerSpaxel(n_spaxels=self.n_spaxels, spaxel_values=values)

        assert ps.spaxel_values is values
        assert jnp.allclose(ps.spaxel_values.val, jnp.ones(self.n_spaxels))

    def test_call(self):
        # Test function evaluation
        values = jnp.arange(self.n_spaxels, dtype=float)
        ps = PerSpaxel(n_spaxels=self.n_spaxels, spaxel_values=Parameter(initial=values))
        out = ps(self.data)

        # Check output shape
        assert out.shape == (self.data.x.shape[0],)

        # Check output values (should match values at corresponding indices)
        expected = values[self.data.idx.astype(int)]
        assert jnp.allclose(out, expected)

    def test_out_of_bounds_indices(self):
        # Test behavior with out-of-bounds indices
        data_bad = SpatialDataGeneric(
            x=jnp.array([0.0]),
            y=jnp.array([0.0]),
            idx=jnp.array([self.n_spaxels + 5]),  # Out of bounds
        )
        ps = PerSpaxel(n_spaxels=self.n_spaxels)

        # In JAX's default mode, this doesn't raise an error but returns an undefined value
        # Instead, verify that we get a value (which may be arbitrary due to JAX's behavior)
        result = ps(data_bad)
        assert result.shape == (1,)  # Should still return the expected shape
