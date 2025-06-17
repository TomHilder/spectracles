"""test_spectral.py - tests for the modelling_lib.model.spectral module."""

import jax.numpy as jnp
from jax import vmap
from modelling_lib.model.data import SpatialDataGeneric
from modelling_lib.model.spatial import SpatialModel
from modelling_lib.model.spectral import Constant, Gaussian


class DummySpatialModel(SpatialModel):
    """Dummy spatial model that returns a constant value."""

    value: float

    def __init__(self, value: float = 1.0):
        self.value = value

    def __call__(self, data: SpatialDataGeneric):
        return self.value * jnp.ones(data.x.shape)


class TestConstant:
    def setup_method(self):
        # Common setup for Constant tests
        self.data = SpatialDataGeneric(
            x=jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            y=jnp.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            idx=jnp.array([0, 1, 2, 3, 4]),
        )
        self.λ = jnp.array([650.0, 651.0, 652.0, 653.0, 654.0])

    def test_initialization(self):
        # Test initialization
        const_model = DummySpatialModel(value=2.0)
        model = Constant(const=const_model)

        # Check attributes
        assert model.const is const_model

    def test_call(self):
        # Test function evaluation
        const_model = DummySpatialModel(value=2.0)
        model = Constant(const=const_model)

        # Use vmap to apply model over each wavelength
        def call_at_spatial_data(λ_i):
            return model(λ_i, self.data)

        vmap_out = vmap(call_at_spatial_data)(self.λ)

        # Check output shape
        expected_shape = (len(self.λ), len(self.data.idx))
        assert vmap_out.shape == expected_shape

        # Check output values (should be constant for all wavelengths)
        assert jnp.allclose(vmap_out, 2.0)

    def test_different_spatial_values(self):
        # Test with spatial model that returns different values per spaxel
        class VaryingSpatialModel(SpatialModel):
            def __call__(self, data: SpatialDataGeneric):
                return data.idx.astype(float)

        const_model = VaryingSpatialModel()
        model = Constant(const=const_model)

        def call_at_spatial_data(λ_i):
            return model(λ_i, self.data)

        vmap_out = vmap(call_at_spatial_data)(self.λ)

        # Output should be a 2D array with shape (n_wavelengths, n_spaxels)
        expected_shape = (len(self.λ), len(self.data.idx))
        assert vmap_out.shape == expected_shape

        # Each column should match the indices
        for i in range(len(self.data.idx)):
            assert jnp.allclose(vmap_out[:, i], float(self.data.idx[i]))


class TestGaussian:
    def setup_method(self):
        # Common setup for Gaussian tests
        self.data = SpatialDataGeneric(
            x=jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            y=jnp.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            idx=jnp.array([0, 1, 2, 3, 4]),
        )
        self.λ = jnp.linspace(650.0, 660.0, 100)

    def test_initialization(self):
        # Test initialization
        A = DummySpatialModel(value=1.0)
        λ0 = DummySpatialModel(value=655.0)
        σ = DummySpatialModel(value=1.0)
        model = Gaussian(A=A, λ0=λ0, σ=σ)

        # Check attributes
        assert model.A is A
        assert model.λ0 is λ0
        assert model.σ is σ

    def test_call(self):
        # Test function evaluation
        A = DummySpatialModel(value=1.0)
        λ0 = DummySpatialModel(value=655.0)
        σ = DummySpatialModel(value=1.0)
        model = Gaussian(A=A, λ0=λ0, σ=σ)

        # Use vmap to apply model over each wavelength
        def call_at_spatial_data(λ_i):
            return model(λ_i, self.data)

        vmap_out = vmap(call_at_spatial_data)(self.λ)

        # Check output shape
        expected_shape = (len(self.λ), len(self.data.idx))
        assert vmap_out.shape == expected_shape

        # Check output values (should be Gaussian centered at λ0)
        # Compute expected Gaussian manually
        A_norm = 1.0 / (1.0 * jnp.sqrt(2 * jnp.pi))
        expected = A_norm * jnp.exp(-0.5 * ((self.λ[:, None] - 655.0) / 1.0) ** 2)

        # All columns should be the same since parameters are constant
        assert jnp.allclose(vmap_out, expected)

    def test_varying_parameters(self):
        # Test with varying parameters across spaxels
        class PosSpatialModel(SpatialModel):
            def __call__(self, data: SpatialDataGeneric):
                return 650.0 + data.idx  # Different center for each spaxel

        A = DummySpatialModel(value=1.0)
        λ0 = PosSpatialModel()
        σ = DummySpatialModel(value=1.0)
        model = Gaussian(A=A, λ0=λ0, σ=σ)

        def call_at_spatial_data(λ_i):
            return model(λ_i, self.data)

        vmap_out = vmap(call_at_spatial_data)(self.λ)

        # Output should be a 2D array with shape (n_wavelengths, n_spaxels)
        expected_shape = (len(self.λ), len(self.data.idx))
        assert vmap_out.shape == expected_shape

        # Check that maximum value for each spaxel is at the corresponding center
        # Each column represents a spaxel with its own center wavelength
        max_indices = jnp.argmax(vmap_out, axis=0)  # Get argmax for each column (spaxel)
        λ_max = self.λ[max_indices]

        # The maximum should be close to the centers specified by λ0
        centers = 650.0 + self.data.idx
        assert jnp.allclose(λ_max, centers, atol=self.λ[1] - self.λ[0])

    def test_normalization(self):
        # Test that the Gaussian is properly normalized
        A = DummySpatialModel(value=2.0)  # Area = 2.0
        λ0 = DummySpatialModel(value=655.0)
        σ = DummySpatialModel(value=1.0)
        model = Gaussian(A=A, λ0=λ0, σ=σ)

        # Create a very dense wavelength grid to approximate the integral
        λ_dense = jnp.linspace(650.0, 660.0, 1000)
        dλ = λ_dense[1] - λ_dense[0]

        def call_at_spatial_data(λ_i):
            return model(λ_i, self.data)

        vmap_out = vmap(call_at_spatial_data)(λ_dense)

        # Integrate the output over wavelengths for the first spaxel
        # Since all spaxels have the same parameters, they should all integrate to the same value
        integral = jnp.sum(vmap_out[:, 0]) * dλ

        # Should be close to A
        assert jnp.isclose(integral, 2.0, rtol=1e-2)

    def test_different_width(self):
        # Test with different width
        A = DummySpatialModel(value=1.0)
        λ0 = DummySpatialModel(value=655.0)
        σ1 = DummySpatialModel(value=1.0)
        σ2 = DummySpatialModel(value=2.0)
        model1 = Gaussian(A=A, λ0=λ0, σ=σ1)
        model2 = Gaussian(A=A, λ0=λ0, σ=σ2)

        # Use vmap to apply model over each wavelength
        def call_at_spatial_data(λ_i):
            return model1(λ_i, self.data)

        def call_at_spatial_data2(λ_i):
            return model2(λ_i, self.data)

        out1 = vmap(call_at_spatial_data)(self.λ)
        out2 = vmap(call_at_spatial_data2)(self.λ)

        # Wider Gaussian should have lower peak (checking first spaxel)
        assert jnp.max(out1[:, 0]) > jnp.max(out2[:, 0])

        # But area should be approximately the same
        λ_range = self.λ[-1] - self.λ[0]
        dλ = λ_range / len(self.λ)
        integral1 = jnp.sum(out1[:, 0]) * dλ
        integral2 = jnp.sum(out2[:, 0]) * dλ
        assert jnp.isclose(integral1, integral2, rtol=0.05)
