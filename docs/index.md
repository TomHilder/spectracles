# spectracles

Continuous field models for integral field spectroscopy and beyond.

Spectracles fits models to IFU data where line or spectrum properties (line
flux or velocity, continuum, ...) vary continuously across the sky. These are
modelled as *fields*, so they are defined at every sky position spanned by the
data, not just at the spaxels.

More generally, spectracles is a [JAX](https://github.com/jax-ml/jax) library
for statistical models with latent fields, built for large problems. Models are
modular [Equinox](https://docs.kidger.site/equinox/) modules, and work with
most JAX inference tools: MGVI and geoVI through
[NIFTy.re](https://github.com/NIFTy-PPL/NIFTy), NUTS through
[BlackJAX](https://github.com/blackjax-devs/blackjax) or
[NumPyro](https://github.com/pyro-ppl/numpyro), or a best fit with
[optax](https://github.com/google-deepmind/optax).

## Installation

```bash
pip install spectracles            # or: uv add spectracles
pip install "spectracles[nifty]"   # to use MGVI and geoVI
pip install blackjax               # to use NUTS
```

For SDSS-V LVM data, [lvm_tools](https://tomhilder.github.io/lvm_tools/)
reads the DRP output (lazily, with dask) and gives you the data ready for
spectracles: `pip install "lvm-tools[spectracles]"`.

To run on a GPU, build
[jax-finufft](https://github.com/flatironinstitute/jax-finufft) from source as
its README describes.

## Example

A positive-valued field, observed with noise at 300 scattered points and
fitted with NUTS:

```python
import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from spectracles import FourierGP, Matern32, Parameter, SpatialDataGeneric, build_model
from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior
from spectracles.priors import LogNormal

jax.config.update("jax_enable_x64", True)


class LineMap(eqx.Module):
    log_flux: FourierGP

    def __call__(self, data):
        return jnp.exp(self.log_flux(data))


kernel = Matern32(
    length_scale=Parameter(prior=LogNormal(0.5, 0.2)),
    variance=Parameter(prior=LogNormal(1.0, 0.5)),
)
model = build_model(LineMap, FourierGP(kernel, n_modes=(15, 15)))
latent = LatentSpace(model)  # the parameters to fit

# Fake data from a random draw of the model
xy = random.uniform(random.PRNGKey(0), (2, 300), minval=-3.0, maxval=3.0)
spaxels = SpatialDataGeneric(x=xy[0], y=xy[1], idx=jnp.arange(300))
truth = latent.sample_prior_latent(random.PRNGKey(1))
flux = latent.to_model(truth)(spaxels) + 0.05 * random.normal(random.PRNGKey(2), (300,))
ivar = jnp.full(300, 0.05**-2)

posterior = Posterior(latent, lambda m: m(spaxels), GaussianLikelihood(flux, ivar))

# Sample with NUTS
n_steps = 300  # fast in tests: 10
warmup = blackjax.window_adaptation(blackjax.nuts, posterior.log_density)
(state, params), _ = warmup.run(random.PRNGKey(3), latent.zeros(), num_steps=n_steps)
nuts = blackjax.nuts(posterior.log_density, **params)


def step(state, key):
    state, _ = nuts.step(key, state)
    return state, state.position


_, samples = jax.lax.scan(step, state, random.split(random.PRNGKey(4), n_steps))

posterior.summary(samples, true_latents=truth)  # fitted parameters vs the truth
maps = posterior.draws(samples, to="model")(spaxels)  # the model for every sample
mean_map, std_map = maps.mean(), maps.std()
```

## Where to go next

- [Building models](models.md): fields, per-spaxel terms, sharing, and
  evaluating your model.
- [Priors and inference](inference.md): priors, the posterior, fitting with
  MGVI, geoVI, NUTS or a best fit, and working with the results.
- [LVM data](lvm.md): fitting SDSS-V LVM data with
  [lvm_tools](https://tomhilder.github.io/lvm_tools/).
- [Worked example](examples/inference.md): one field fitted three ways, with
  figures.
- [How it works](inference_design.md): the design, for when you rely on the
  details.

## Stability

From 1.0.0, spectracles follows semantic versioning: anything in
`spectracles`, `spectracles.priors`, `spectracles.inference` or
`spectracles.nifty` only changes in a major release, after a deprecation
warning. Two parts are outside this promise:

!!! warning "Experimental: `spectracles.lvm_models`"
    The LVM-specific models are being ported to priors and the latent space one
    at a time.

!!! note "Changing: `OptimiserFrame` and schedules"
    These will move onto the same posterior as the rest of spectracles. See the
    [roadmap](roadmap.md).

## Citation

If you use spectracles, please cite
[Hilder et al. (2026), ApJ 997, 339](https://doi.org/10.3847/1538-4357/ae274b).
The [README](https://github.com/TomHilder/spectracles#citation) has the BibTeX
entry and the other software to cite.
