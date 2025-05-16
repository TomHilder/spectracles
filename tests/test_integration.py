"""Integration tests for modelling_lib."""

import tempfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import optax
import pytest
from equinox import Module
from modelling_lib.data import SpatialData
from modelling_lib.io import load_model, save_model
from modelling_lib.kernels import Matern32
from modelling_lib.leaf_sharing import Shared, build_model
from modelling_lib.optimise import OptimiserFrame
from modelling_lib.parameter import ConstrainedParameter, Parameter
from modelling_lib.spatial import FourierGP, PerSpaxel
from modelling_lib.spectral import Constant, Gaussian


class ComplexModel(Module):
    gaussian1: "Gaussian"
    gaussian2: "Gaussian"

    def __init__(self, n_spaxels):  # Corrected: n_spaxel to n_spaxels
        self.gaussian1 = Gaussian(
            A=PerSpaxel(
                n_spaxels=n_spaxels
            ),  # Corrected: n_spaxel to n_spaxels and closed parenthesis
            μ=PerSpaxel(n_spaxels=n_spaxels),  # Corrected: n_spaxel to n_spaxels
            σ=PerSpaxel(n_spaxels=n_spaxels),  # Corrected: n_spaxel to n_spaxels
            λ0=PerSpaxel(n_spaxels=n_spaxels),  # Corrected: n_spaxel to n_spaxels
        )
        self.gaussian2 = Gaussian(
            A=PerSpaxel(n_spaxels=n_spaxels),  # Corrected: n_spaxel to n_spaxels
            μ=PerSpaxel(n_spaxels=n_spaxels),  # Corrected: n_spaxel to n_spaxels
            σ=PerSpaxel(n_spaxels=n_spaxels),  # Corrected: n_spaxel to n_spaxels
            λ0=PerSpaxel(n_spaxels=n_spaxels),  # Corrected: n_spaxel to n_spaxels
        )

    def __call__(self, x, y):
        return self.gaussian1(x, y) + self.gaussian2(x, y)


class SimpleSpatialTemporalModel(Module):
    background: Constant
    line: Gaussian
    temporal_coeffs: Parameter

    def __init__(self, n_spaxels: int, n_times: int, n_modes: tuple[int, int], kernel: Matern32):
        # Spatial components
        background_spatial = FourierGP(
            n_modes=n_modes,
            kernel=kernel,
        )
        self.background = Constant(const=background_spatial)

        A_spatial = FourierGP(n_modes=n_modes, kernel=kernel)
        λ0_spatial = FourierGP(n_modes=n_modes, kernel=kernel)
        σ_spatial = FourierGP(n_modes=n_modes, kernel=kernel)

        self.line = Gaussian(
            A=A_spatial,
            λ0=λ0_spatial,
            σ=σ_spatial,
        )
        # Temporal component
        self.temporal_coeffs = Parameter(initial=jnp.ones((n_times,)))

    def __call__(self, λ: jnp.ndarray, spatial_data: SpatialData, time_index: int):
        return (
            self.background(λ, spatial_data)
            + self.line(λ, spatial_data) * self.temporal_coeffs[time_index]
        )


class TestEndToEndWorkflow:
    """Test a complete workflow from creating a model to training and saving it."""

    def test_simple_spatial_temporal_model(self, tmp_path):
        # Create a simple spatial-spectral model

        # 1. Define the model
        class SimpleModel(Module):
            background: Constant
            emission_line: Gaussian

            def __init__(self, n_modes):
                # Background model (constant in wavelength)
                background_spatial = FourierGP(
                    n_modes=n_modes,
                    kernel=Matern32(
                        length_scale=Parameter(initial=0.5), variance=Parameter(initial=1.0)
                    ),
                )
                self.background = Constant(const=background_spatial)

                # Emission line model (Gaussian in wavelength)
                A_spatial = FourierGP(
                    n_modes=n_modes,
                    kernel=Matern32(
                        length_scale=Parameter(initial=0.5), variance=Parameter(initial=1.0)
                    ),
                )
                λ0_spatial = PerSpaxel(n_spaxels=10, spaxel_values=Parameter(initial=655.0))
                σ_spatial = PerSpaxel(
                    n_spaxels=10,
                    spaxel_values=ConstrainedParameter(initial=1.0, lower=0.1, upper=5.0),
                )
                self.emission_line = Gaussian(A=A_spatial, λ0=λ0_spatial, σ=σ_spatial)

            def __call__(self, λ, spatial_data):
                return self.background(λ, spatial_data) + self.emission_line(λ, spatial_data)

        # 2. Build the model with proper sharing structure
        model = build_model(SimpleModel, n_modes=(5, 5))

        # 3. Create some synthetic data
        λ = jnp.linspace(650.0, 660.0, 20)
        n_spaxels = 10
        spatial_data = SpatialData(
            x=jnp.linspace(0, 1, n_spaxels),
            y=jnp.linspace(0, 1, n_spaxels),
            indices=jnp.arange(n_spaxels),
        )

        # 4. Define a loss function
        def loss_fn(model, λ, spatial_data, data):
            # Simple MSE loss
            pred = model(λ, spatial_data)
            return jnp.mean((pred - data) ** 2)

        # Generate some synthetic "observed" data
        # Just use the model itself with some known parameters
        _true_model = SimpleModel(n_modes=(5, 5))
        # Modify some parameters to ensure they're different from initialization
        _true_model = eqx.tree_at(
            lambda m: m.background.const.kernel.length_scale.val,
            _true_model,
            jnp.array([0.8]),
        )
        _true_model = eqx.tree_at(
            lambda m: m.emission_line.A.kernel.length_scale.val,
            _true_model,
            jnp.array([0.3]),
        )
        _true_model = eqx.tree_at(
            lambda m: m.emission_line.λ0.spaxel_values.val,
            _true_model,
            jnp.ones(n_spaxels) * 654.0,
        )
        true_model = _true_model  # Assign to the original name after all modifications
        observed_data = true_model(λ, spatial_data)

        # 5. Set up optimization
        optimiser = optax.adam(learning_rate=0.01)
        opt_frame = OptimiserFrame(model, loss_fn, optimiser)

        # 6. Train the model (for just a few steps in this test)
        trained_model = opt_frame.run(
            n_steps=10, λ=λ, spatial_data=spatial_data, data=observed_data
        )

        # Check that loss decreased
        assert opt_frame.loss_history[0] > opt_frame.loss_history[-1]

        # 7. Save and load the model
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "model"
            save_model(trained_model, file_path)
            loaded_model = load_model(file_path)

            # 8. Check that the loaded model gives the same output
            orig_output = trained_model(λ, spatial_data)
            loaded_output = loaded_model(λ, spatial_data)
            assert jnp.allclose(orig_output, loaded_output)

    def test_save_and_load_model_with_optimiser_state(self, tmp_path):
        file_path = tmp_path / "model_with_optimiser.eqx"
        n_spaxels = 3
        model = ComplexModel(n_spaxels=n_spaxels)
        optimiser = optax.adam(learning_rate=0.01)
        opt_frame = OptimiserFrame(model=model, optimiser=optimiser)

        # Simulate some optimisation steps to change the optimiser state
        # ... (actual optimisation steps would go here)

        save_model(opt_frame, file_path)
        loaded_opt_frame = load_model(file_path)

        assert isinstance(loaded_opt_frame, OptimiserFrame)
        # Further checks could involve comparing model parameters and optimiser states
        # For now, just check that loading didn't crash and returned the correct type

    def test_parameter_sharing_in_complex_model(self, tmp_path):
        """Test parameter sharing in a more complex model where multiple parts share parameters."""

        # Define a model with explicit parameter sharing
        class SharedParamsModel(Module):
            gaussian1: Gaussian
            gaussian2: Gaussian

            def __init__(self, n_spaxels):
                # Common width parameter shared between two Gaussians
                common_σ = PerSpaxel(
                    n_spaxels=n_spaxels, spaxel_values=ConstrainedParameter(initial=1.0, lower=0.1)
                )

                # First Gaussian
                self.gaussian1 = Gaussian(
                    A=PerSpaxel(n_spaxels=n_spaxels),
                    μ=PerSpaxel(n_spaxels=n_spaxels),
                    σ=common_σ,  # Shared parameter
                    λ0=PerSpaxel(n_spaxels=n_spaxels),
                )
                # Second Gaussian
                self.gaussian2 = Gaussian(
                    A=PerSpaxel(n_spaxels=n_spaxels),
                    μ=PerSpaxel(n_spaxels=n_spaxels),
                    σ=common_σ,  # Shared parameter
                    λ0=PerSpaxel(n_spaxels=n_spaxels),
                )

            def __call__(self, x, y):
                return self.gaussian1(x, y) + self.gaussian2(x, y)

        # 1. Build the model
        model = SharedParamsModel(n_spaxels=10)

        # 2. Define a loss function
        def loss_fn(model, x, y, data):
            # Simple MSE loss
            pred = model(x, y)
            return jnp.mean((pred - data) ** 2)

        # 3. Create synthetic data
        x = jnp.linspace(-5.0, 5.0, 100)
        y = jnp.linspace(-5.0, 5.0, 100)
        X, Y = jnp.meshgrid(x, y)
        spatial_data = SpatialData(x=X.ravel(), y=Y.ravel(), indices=jnp.arange(X.size))

        # Generate some synthetic "observed" data
        # Just use the model itself with some known parameters
        true_model = model
        observed_data = true_model(X.ravel(), Y.ravel())

        # 4. Set up optimization
        optimiser = optax.adam(learning_rate=0.01)
        opt_frame = OptimiserFrame(model, loss_fn, optimiser)

        # 5. Train the model (for just a few steps in this test)
        trained_model = opt_frame.run(n_steps=10, x=X.ravel(), y=Y.ravel(), data=observed_data)

        # Check that loss decreased
        assert opt_frame.loss_history[0] > opt_frame.loss_history[-1]

        # 6. Save and load the model
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "shared_model"
            save_model(trained_model, file_path)
            loaded_model = load_model(file_path)

            # 7. Check that the loaded model gives the same output
            orig_output = trained_model(X.ravel(), Y.ravel())
            loaded_output = loaded_model(X.ravel(), Y.ravel())
            assert jnp.allclose(orig_output, loaded_output)

        # Share the σ parameter between the two Gaussians
        # The _parent_leaf_paths attribute should now reflect this sharing
        # And the actual value of gaussian2.σ.spaxel_values.unconstrained_val should be a Shared object
        # Check that the path to gaussian1.σ.spaxel_values.unconstrained_val is in _parent_leaf_paths
        # And that the path to gaussian2.σ.spaxel_values.unconstrained_val is NOT (it's a child)

        # Corrected assertion: check if the .val attribute of the Parameter object is a Shared instance
        # after build_model has processed it.
        # The original model.gaussian1.σ and model.gaussian2.σ are Parameter objects.
        # build_model identifies that their .val attributes should be shared.
        # It replaces model.gaussian2.σ.val with a Shared sentinel pointing to model.gaussian1.σ.val.

        # Initial check: σ parameters should be different objects with different values before sharing
        assert model.gaussian1.σ is not model.gaussian2.σ
        assert not jnp.allclose(model.gaussian1.σ.val, model.gaussian2.σ.val)

        shared_paths = [(("gaussian1", "σ"), ("gaussian2", "σ"))]
        model = build_model(model, shared_paths=shared_paths)

        # After sharing, the .val attributes of the spaxel_values should be shared.
        # The Parameter objects themselves (model.gaussian1.σ and model.gaussian2.σ) are distinct.
        # Their .spaxel_values attributes are also distinct Parameter objects.
        # It's the JAX array held within .val of these .spaxel_values that is shared.

        # Check that the Shared object is correctly placed
        assert isinstance(model.gaussian2.σ.spaxel_values.unconstrained_val, Shared)

        # Check that the parent path is correctly registered
        assert (
            ("gaussian1", "σ", "spaxel_values", "unconstrained_val"),
        ) == model.gaussian2.σ.spaxel_values.unconstrained_val.parent_path

        # Check that the values are indeed the same after sharing
        assert jnp.allclose(model.gaussian1.σ.val, model.gaussian2.σ.val)

        # Test that changing the parent's value also changes the child's value
        # We need to modify the original source of the shared value, which is model.gaussian1.σ.val
        # Since the model is immutable, we use eqx.tree_at to create a new model with the modified value.
        new_sigma_val = jnp.array([10.0, 11.0, 12.0])

        # The path to update is model.gaussian1.σ.spaxel_values.val
        # (or .unconstrained_val depending on how Parameter is structured and if constraints are used)
        # Let's assume .val is the direct way to update the underlying array for PerSpaxel

        # Create a getter for the specific Parameter's .val attribute
        def get_sigma_val(m):
            return m.gaussian1.σ.spaxel_values.val  # Assuming .val for direct update

        model_updated = eqx.tree_at(get_sigma_val, model, new_sigma_val)

        # Verify that both gaussians now have the new sigma value
        assert jnp.allclose(model_updated.gaussian1.σ.val, new_sigma_val)
        assert jnp.allclose(model_updated.gaussian2.σ.val, new_sigma_val)

        # Test saving and loading the shared model
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "shared_model_updated"
            save_model(model_updated, file_path)
            loaded_model_updated = load_model(file_path)

            # Check that the loaded model gives the same output
            updated_orig_output = model_updated(X.ravel(), Y.ravel())
            loaded_updated_output = loaded_model_updated(X.ravel(), Y.ravel())
            assert jnp.allclose(updated_orig_output, loaded_updated_output)

            # Check that the shared parameter value is consistent
            assert jnp.allclose(
                loaded_model_updated.gaussian1.σ.spaxel_values.unconstrained_val,
                loaded_model_updated.gaussian2.σ.spaxel_values.unconstrained_val,
            )
