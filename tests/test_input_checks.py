"""test_input_checks.py - Common mistakes raise errors that say what was expected."""

import equinox as eqx
import jax.numpy as jnp
import pytest
from jax import random

from spectracles import (
    FourierGP,
    Matern32,
    Parameter,
    PerSpaxel,
    SpatialDataGeneric,
    build_model,
)
from spectracles.inference import GaussianLikelihood, LatentSpace
from spectracles.model.spatial import MultiFourierGP, PerTile
from spectracles.priors import LogNormal


class Field(eqx.Module):
    gp: FourierGP

    def __call__(self, data):
        return self.gp(data)


def kernel():
    return Matern32(Parameter(prior=LogNormal(0.5, 0.2)), Parameter(prior=LogNormal(1.0, 0.5)))


def latent_space(n_modes=(5, 5)):
    return LatentSpace(build_model(Field, FourierGP(n_modes, kernel())))


# ---- Parameter ----


def test_prior_passed_positionally():
    with pytest.raises(TypeError, match="prior=..."):
        Parameter(0.5, LogNormal(0.5, 0.2))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prior=..."):
        Parameter(LogNormal(0.5, 0.2))  # type: ignore[arg-type]


def test_dims_and_fixed_are_checked():
    with pytest.raises(TypeError, match="dims must be an int or a tuple"):
        Parameter(dims=[3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dims must be positive"):
        Parameter(dims=0)
    with pytest.raises(TypeError, match="fixed must be True or False"):
        Parameter(1.0, fixed="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="doesn't match dims"):
        Parameter([1.0, 2.0], dims=3)


# ---- Kernels and GPs ----


def test_kernel_hyperparameters_must_be_parameters():
    with pytest.raises(TypeError, match="Matern32: length_scale must be a Parameter"):
        Matern32(0.5, Parameter(1.0))  # type: ignore[arg-type]


def test_fourier_gp_arguments():
    with pytest.raises(TypeError, match="kernel must be a Kernel"):
        FourierGP((5, 5), Parameter(1.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="n_modes must be two integers"):
        FourierGP(5, kernel())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive odd"):
        FourierGP((6, 6), kernel())
    with pytest.raises(TypeError, match="coefficients must be a Parameter"):
        FourierGP((5, 5), kernel(), coefficients=jnp.zeros((5, 5)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"coefficients must have shape \(5, 5\)"):
        FourierGP((5, 5), kernel(), coefficients=Parameter(dims=(3, 3)))
    with pytest.raises(TypeError, match="zeromode must be a Parameter"):
        FourierGP((5, 5), kernel(), zeromode=1.0)  # type: ignore[arg-type]
    assert FourierGP([5, 5], kernel()).n_modes == (5, 5)  # type: ignore[arg-type]


def test_multi_fourier_gp_kernel_count():
    with pytest.raises(ValueError, match="2 kernels for 3 components"):
        MultiFourierGP(n_components=3, n_modes=(5, 5), kernels=[kernel(), kernel()])


def test_per_spaxel_and_tile_values():
    with pytest.raises(TypeError, match="spaxel_values must be a Parameter"):
        PerSpaxel(4, jnp.zeros(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"spaxel_values must have shape \(4,\)"):
        PerSpaxel(4, Parameter(dims=3))
    with pytest.raises(ValueError, match=r"tile_values must have shape \(2,\)"):
        PerTile(2, Parameter(dims=5))


# ---- Data and model building ----


def test_spatial_data_lengths():
    with pytest.raises(ValueError, match="same length"):
        SpatialDataGeneric(x=jnp.zeros(4), y=jnp.zeros(3), idx=jnp.arange(4))


def test_build_model_takes_a_class_or_an_instance():
    gp = FourierGP((5, 5), kernel())
    from_class, from_instance = build_model(Field, gp), build_model(Field(gp))
    assert from_class.get_parameter_paths() == from_instance.get_parameter_paths()
    with pytest.raises(TypeError, match="or a model instance on its own"):
        build_model(Field(gp), gp)
    with pytest.raises(TypeError, match="already built"):
        build_model(from_class)
    with pytest.raises(TypeError, match="needs an equinox Module"):
        build_model(lambda: 3)


def test_set_unknown_path():
    model = build_model(Field, FourierGP((5, 5), kernel()))
    with pytest.raises(ValueError, match="No parameter at path 'gp.nope'"):
        model.set({"gp.nope": jnp.zeros(1)})


# ---- Latents ----


def test_latents_checked():
    latent = latent_space()
    ξ = latent.sample_prior_latent(random.PRNGKey(0))
    with pytest.raises(TypeError, match="got a model"):
        latent.to_model(latent.model)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"missing \['gp.coefficients'\]"):
        latent.to_params({p: v for p, v in ξ.items() if p != "gp.coefficients"})
    with pytest.raises(ValueError, match=r"unknown \['gp.nope'\]"):
        latent.to_params({**ξ, "gp.nope": jnp.zeros(1)})


def test_from_model_needs_the_same_structure():
    latent = latent_space((5, 5))
    other = build_model(Field, FourierGP((7, 7), kernel()))
    with pytest.raises(ValueError, match=r"shape \(7, 7\).*expects \(5, 5\)"):
        latent.from_model(other)


def test_posterior_summary_needs_a_sample_axis():
    latent = latent_space()
    with pytest.raises(ValueError, match="leading axis"):
        latent.posterior_summary(latent.sample_prior_latent(random.PRNGKey(0)), print_table=False)


def test_likelihood_needs_some_data():
    with pytest.raises(ValueError, match="No data points left"):
        GaussianLikelihood(jnp.zeros(4), jnp.ones(4), jnp.zeros(4, bool))
    with pytest.raises(ValueError, match="No data points left"):
        GaussianLikelihood(jnp.zeros(4), -jnp.ones(4))
