"""test_kernels.py - tests for the modelling_lib.model.kernels module."""

import jax.numpy as jnp
from modelling_lib.model.kernels import (
    Matern12,
    Matern32,
    Matern52,
    SquaredExponential,
    matern_kernel_fw_nd,
    normalise_fw,
)
from modelling_lib.model.parameter import Parameter


class TestNormaliseFW:
    def test_normalisation_scalar(self):
        # Test with scalar
        fw = jnp.array([1.0])
        normalised = normalise_fw(fw)
        assert jnp.isclose(jnp.sum(jnp.abs(normalised) ** 2), 2.0)

    def test_normalisation_vector(self):
        # Test with vector
        fw = jnp.array([1.0, 2.0, 3.0])
        normalised = normalise_fw(fw)
        assert jnp.isclose(jnp.sum(jnp.abs(normalised) ** 2), 2.0)

        # Check relative proportions are preserved
        assert jnp.allclose(normalised[1] / normalised[0], 2.0)
        assert jnp.allclose(normalised[2] / normalised[0], 3.0)

    def test_normalisation_2d(self):
        # Test with 2D array
        fw = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        normalised = normalise_fw(fw)
        assert jnp.isclose(jnp.sum(jnp.abs(normalised) ** 2), 2.0)

    def test_normalisation_zero(self):
        # Test with zero array
        fw = jnp.zeros((3,))
        # This should handle division by zero gracefully
        normalised = normalise_fw(fw)
        assert jnp.all(jnp.isnan(normalised))


class TestMaternKernelFwNd:
    def test_output_shape(self):
        # Test output shape matches input
        freqs = jnp.array([0.0, 1.0, 2.0])
        out = matern_kernel_fw_nd(freqs, 1.0, 2.0, nu=1.5, n=2)
        assert out.shape == freqs.shape

        freqs_2d = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        out_2d = matern_kernel_fw_nd(freqs_2d, 1.0, 2.0, nu=1.5, n=2)
        assert out_2d.shape == freqs_2d.shape

    def test_nonnegativity(self):
        # Test output is non-negative
        freqs = jnp.array([0.0, 1.0, 2.0])
        out = matern_kernel_fw_nd(freqs, 1.0, 2.0, nu=1.5, n=2)
        assert jnp.all(out >= 0)

    def test_zero_frequency(self):
        # Test zero frequency gives maximum value
        freqs = jnp.array([0.0, 1.0, 2.0])
        out = matern_kernel_fw_nd(freqs, 1.0, 2.0, nu=1.5, n=2)
        assert jnp.argmax(out) == 0

    def test_different_nus(self):
        # Test different nu values affect decay rate
        freqs = jnp.linspace(0, 5, 100)
        out_nu05 = matern_kernel_fw_nd(freqs, 1.0, 1.0, nu=0.5, n=2)
        out_nu15 = matern_kernel_fw_nd(freqs, 1.0, 1.0, nu=1.5, n=2)
        out_nu25 = matern_kernel_fw_nd(freqs, 1.0, 1.0, nu=2.5, n=2)

        # Higher nu means faster decay, so values should be lower at high frequencies
        assert jnp.all(out_nu05[50:] >= out_nu15[50:])
        assert jnp.all(out_nu15[50:] >= out_nu25[50:])

    def test_length_scale(self):
        # Test length scale affects decay rate
        freqs = jnp.linspace(0, 5, 100)
        out_l1 = matern_kernel_fw_nd(freqs, 1.0, 1.0, nu=1.5, n=2)
        out_l2 = matern_kernel_fw_nd(freqs, 2.0, 1.0, nu=1.5, n=2)

        # Larger length scale means faster decay in frequency domain
        assert jnp.all(out_l2[50:] <= out_l1[50:])

    def test_variance(self):
        # Test variance scales output appropriately
        freqs = jnp.linspace(0, 5, 100)
        out_v1 = matern_kernel_fw_nd(freqs, 1.0, 1.0, nu=1.5, n=2)
        out_v2 = matern_kernel_fw_nd(freqs, 1.0, 2.0, nu=1.5, n=2)

        # Double variance means sqrt(2) times output
        assert jnp.allclose(out_v2, out_v1 * jnp.sqrt(2))

    def test_dimension(self):
        # Test dimension affects decay rate
        freqs = jnp.linspace(0, 5, 100)
        out_n1 = matern_kernel_fw_nd(freqs, 1.0, 1.0, nu=1.5, n=1)
        out_n2 = matern_kernel_fw_nd(freqs, 1.0, 1.0, nu=1.5, n=2)
        out_n3 = matern_kernel_fw_nd(freqs, 1.0, 1.0, nu=1.5, n=3)

        # Higher dimension means faster decay
        assert jnp.all(out_n1[50:] >= out_n2[50:])
        assert jnp.all(out_n2[50:] >= out_n3[50:])


class TestKernelClasses:
    def setup_method(self):
        # Common setup for all kernel tests
        self.length_scale = Parameter(initial=1.0)
        self.variance = Parameter(initial=2.0)
        self.freqs = jnp.array([0.0, 1.0, 2.0])

    def test_matern12(self):
        kernel = Matern12(length_scale=self.length_scale, variance=self.variance)
        fw = kernel.feature_weights(self.freqs)

        # Compare with direct calculation
        expected = matern_kernel_fw_nd(self.freqs, 1.0, 2.0, nu=0.5, n=2)
        assert jnp.allclose(fw, expected)

    def test_matern32(self):
        kernel = Matern32(length_scale=self.length_scale, variance=self.variance)
        fw = kernel.feature_weights(self.freqs)

        # Compare with direct calculation
        expected = matern_kernel_fw_nd(self.freqs, 1.0, 2.0, nu=1.5, n=2)
        assert jnp.allclose(fw, expected)

    def test_matern52(self):
        kernel = Matern52(length_scale=self.length_scale, variance=self.variance)
        fw = kernel.feature_weights(self.freqs)

        # Compare with direct calculation
        expected = matern_kernel_fw_nd(self.freqs, 1.0, 2.0, nu=2.5, n=2)
        assert jnp.allclose(fw, expected)

    def test_squared_exponential(self):
        kernel = SquaredExponential(length_scale=self.length_scale, variance=self.variance)
        fw = kernel.feature_weights(self.freqs)

        # Verify shape and positivity
        assert fw.shape == self.freqs.shape
        assert jnp.all(fw >= 0)

        # Verify it decays faster than Matern52
        # (at high frequencies, squared exponential should be smaller)
        kernel_m52 = Matern52(length_scale=self.length_scale, variance=self.variance)
        fw_m52 = kernel_m52.feature_weights(jnp.array([10.0]))
        fw_se = kernel.feature_weights(jnp.array([10.0]))
        # The test might fail if values are exactly equal due to numerical precision
        assert fw_se[0] <= fw_m52[0]

    def test_parameter_updates(self):
        # Test that different parameter values affect output
        kernel1 = Matern32(length_scale=self.length_scale, variance=self.variance)
        fw1 = kernel1.feature_weights(self.freqs)

        # Create new parameters with different values
        new_length_scale = Parameter(initial=2.0)
        new_variance = Parameter(initial=3.0)

        # Create a new kernel with the new parameters
        kernel2 = Matern32(length_scale=new_length_scale, variance=new_variance)
        fw2 = kernel2.feature_weights(self.freqs)

        # Outputs should be different
        assert not jnp.allclose(fw1, fw2)
