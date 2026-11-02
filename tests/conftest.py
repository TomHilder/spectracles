"""conftest.py - common fixtures for modelling_lib tests."""

import tempfile
from pathlib import Path

import jax
import pytest

# Float64 throughout, so cross-backend comparisons can be tight
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---- Tiny problems for the prior / latent-space / backend tests ----


@pytest.fixture
def tiny_gp_model():
    from .priors_models import make_gp_model

    return make_gp_model()


@pytest.fixture
def tiny_line_model():
    from .priors_models import make_line_model

    return make_line_model()


@pytest.fixture
def line_posterior():
    """tiny_line_model + masked data (NaN at ~10% of data points) as a Posterior."""
    from jax import random

    from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior

    from .priors_models import line_data, line_forward, make_line_model, noisy_data

    latent = LatentSpace(make_line_model())
    forward = line_forward(line_data())
    truth = latent.sample_prior_latent(random.PRNGKey(11))
    data, ivar, mask = noisy_data(forward(latent.to_model(truth)), 0.05, 0.1, seed=5)
    return Posterior(latent, forward, GaussianLikelihood(data, ivar, mask))


@pytest.fixture
def linear_gaussian():
    """A FourierGP with fixed hyperparameters: prediction is linear in the latents."""
    from jax import random

    from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior

    from .priors_models import gp_data, make_gp_model, noisy_data

    latent = LatentSpace(make_gp_model(fixed_hyperparameters=True))
    data_xy = gp_data()

    def forward(model):
        return model(data_xy)

    truth = latent.sample_prior_latent(random.PRNGKey(2))
    data, ivar, mask = noisy_data(forward(latent.to_model(truth)), 0.1, 0.0, seed=4)
    return Posterior(latent, forward, GaussianLikelihood(data, ivar, mask))


def make_xi_points(latent, n=5, seed=123):
    from jax import random

    return [latent.sample_prior_latent(k) for k in random.split(random.PRNGKey(seed), n)]
