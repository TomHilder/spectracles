"""posterior.py - The unnormalised posterior in latent space, shared by every backend.

    log p(ξ | d) = log L(d | forward(model(ξ))) - ½ |ξ|²  (+ const)

NIFTy.re (MGVI/geoVI), BlackJAX (NUTS) and optax (MAP) all work on this one
function, in the same coordinates.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from spectracles.inference.latent import Latent, LatentSpace, as_latent_dict
from spectracles.inference.likelihood import GaussianLikelihood
from spectracles.model.share_module import ShareModule


class Posterior:
    """A latent space, a forward model and a likelihood.

    Args:
        latent: The model's latent space.
        forward: Maps a model to a prediction with the data's shape, e.g.
            `lambda m: jax.vmap(m, in_axes=(0, None))(λ, spatial_data)`.
        likelihood: The likelihood of the data given the prediction.

    All methods taking ξ are pure JAX (jit, grad, vmap).
    """

    def __init__(
        self,
        latent: LatentSpace,
        forward: Callable[[ShareModule], Array],
        likelihood: GaussianLikelihood,
    ):
        if not isinstance(latent, LatentSpace):
            raise TypeError(f"latent must be a LatentSpace, got {type(latent).__name__}.")
        self.latent = latent
        self.forward = forward
        self.likelihood = likelihood

    def __repr__(self) -> str:
        return (
            f"Posterior({self.latent!r}, {type(self.likelihood).__name__} on "
            f"{self.likelihood.n_used} pixels)"
        )

    def predict(self, ξ: Latent) -> Array:
        """The forward model's prediction at ξ, with the data's shape."""
        return self.forward(self.latent.to_model(ξ))

    def log_likelihood(self, ξ: Latent) -> Array:
        return self.likelihood.log_likelihood(self.predict(ξ))

    def log_prior(self, ξ: Latent) -> Array:
        """The prior in latent space: -½|ξ|² (without the normalisation constant)."""
        ξ = as_latent_dict(ξ)
        return -0.5 * sum(jnp.sum(ξ[p] ** 2) for p in self.latent.paths)

    def log_density(self, ξ: Latent) -> Array:
        """The unnormalised log-posterior: log-likelihood - ½|ξ|²."""
        return self.log_likelihood(ξ) + self.log_prior(ξ)

    # Conveniences forwarded to the latent space

    def to_model(self, ξ: Latent) -> ShareModule:
        return self.latent.to_model(ξ)

    def to_params(self, ξ: Any) -> dict[str, Array]:
        return self.latent.to_params(ξ)

    def sample_prior_latent(self, key: PRNGKeyArray, n: int | None = None) -> Latent:
        return self.latent.sample_prior_latent(key, n)

    def sample_prior_params(self, key: PRNGKeyArray, n: int | None = None) -> dict[str, Array]:
        return self.latent.sample_prior_params(key, n)

    def sample_prior_model(
        self, key: PRNGKeyArray, n: int | None = None
    ) -> ShareModule | list[ShareModule]:
        return self.latent.sample_prior_model(key, n)

    def zeros(self) -> Latent:
        return self.latent.zeros()

    def value_and_grad(self, ξ: Latent) -> tuple[Array, Latent]:
        """log_density and its gradient with respect to ξ."""
        return jax.value_and_grad(self.log_density)(as_latent_dict(ξ))
