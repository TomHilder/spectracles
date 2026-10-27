"""spectracles.inference - Latent space, likelihoods and the posterior shared by every backend."""

from spectracles.inference.latent import LatentSpace
from spectracles.inference.likelihood import GaussianLikelihood
from spectracles.inference.posterior import Posterior

__all__ = ["GaussianLikelihood", "LatentSpace", "Posterior"]
