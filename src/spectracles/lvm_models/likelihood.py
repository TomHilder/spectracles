import jax
import jax.numpy as jnp


def ln_likelihood(vmapped_model, λ, xy_data, data, u_data, mask):
    # Model predictions
    pred = vmapped_model(λ, xy_data)
    # Likelihood
    return jnp.sum(
        jnp.where(
            mask,
            jax.scipy.stats.norm.logpdf(x=pred, loc=data, scale=u_data),
            0.0,
        )
    )
