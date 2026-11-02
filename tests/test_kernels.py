"""test_kernels.py - tests for the spectracles.model.kernels module."""

import jax.numpy as jnp

from spectracles.model.kernels import (
    Matern12,
    Matern32,
    Matern52,
    SquaredExponential,
    matern_kernel_fw_nd,
    normalise_fw,
)
from spectracles.model.parameter import Parameter


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


class TestContinuousForm:
    """spectral_density and covariance: normalised, a Fourier pair, and the same
    shape as the feature weights the model uses."""

    LENGTH, VAR = 0.7, 1.3
    KERNELS = (Matern12, Matern32, Matern52, SquaredExponential)

    def kernel(self, cls):
        return cls(length_scale=Parameter(self.LENGTH), variance=Parameter(self.VAR))

    def test_spectral_density_integrates_to_the_variance(self):
        from scipy.integrate import quad

        for cls in self.KERNELS:
            k = self.kernel(cls)

            def integrand(q, k=k):
                return 2 * jnp.pi * q * float(jnp.squeeze(k.spectral_density(q)))

            total, _ = quad(integrand, 0, jnp.inf)
            assert jnp.isclose(total, self.VAR, rtol=1e-8), cls.__name__

    def test_covariance_at_zero_is_the_variance(self):
        for cls in self.KERNELS:
            assert jnp.isclose(self.kernel(cls).covariance(0.0), self.VAR), cls.__name__

    def test_spectral_density_is_the_hankel_transform_of_the_covariance(self):
        # S(k) = (1 / 2π) ∫ r k(r) J0(k r) dr in 2D
        from scipy.integrate import quad
        from scipy.special import j0

        for cls in self.KERNELS:
            kern = self.kernel(cls)
            for q in (0.0, 0.5, 2.0, 6.0):

                def integrand(r, kern=kern, q=q):
                    return r * float(jnp.squeeze(kern.covariance(r))) * j0(q * r)

                val, _ = quad(integrand, 0, 60 * self.LENGTH, limit=400)
                assert jnp.isclose(val / (2 * jnp.pi), kern.spectral_density(q), rtol=1e-7), (
                    cls.__name__,
                    q,
                )

    def test_spectral_density_has_the_shape_of_the_feature_weights(self):
        freqs = jnp.linspace(0.0, 20.0, 41)
        for cls in self.KERNELS:
            kern = self.kernel(cls)
            ratio = kern.spectral_density(freqs) / kern.feature_weights(freqs) ** 2
            assert jnp.allclose(ratio, ratio[0], rtol=1e-12), cls.__name__

    def test_kernels_without_a_continuous_form_say_so(self):
        import pytest

        from spectracles.model.kernels import LogLogLinear

        kern = LogLogLinear(m_0=Parameter(0.0), m_s=Parameter(-2.0))
        with pytest.raises(NotImplementedError):
            kern.spectral_density(1.0)
        with pytest.raises(NotImplementedError):
            kern.covariance(1.0)
