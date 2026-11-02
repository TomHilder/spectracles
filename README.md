<div id="top"></div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Tests](https://img.shields.io/github/actions/workflow/status/TomHilder/spectracles/tests.yml?branch=main&label=tests&style=flat-square)](https://github.com/TomHilder/spectracles/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/codecov/c/github/TomHilder/spectracles?style=flat-square)](https://codecov.io/gh/TomHilder/spectracles)
[![PyPI](https://img.shields.io/pypi/v/spectracles?style=flat-square)](https://pypi.org/project/spectracles/)
[![Python](https://img.shields.io/pypi/pyversions/spectracles?style=flat-square)](https://pypi.org/project/spectracles/)
[![Docs](https://img.shields.io/github/actions/workflow/status/TomHilder/spectracles/docs.yml?branch=main&label=docs&style=flat-square)](https://tomhilder.github.io/spectracles/)

</div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/TomHilder/spectracles">
    <img src="https://raw.githubusercontent.com/TomHilder/spectracles/main/logo.png" alt="spectracles" width="420">
  </a>

  <p align="center">
    Unified spectrospatial models for integral field spectroscopy in JAX
  </p>
</div>

## What it does

Spectracles fits models to IFU data where line properties (flux, velocity,
width, ...) vary smoothly across the sky. Each one is a Gaussian process, and
the model is evaluated at the true position of every spaxel, so your data
never has to be resampled onto a grid.

Models are ordinary Python classes. You put priors on the parameters you want
to fit, then fit with whichever method suits the problem: MGVI or geoVI (fast
and approximate, through NIFTy), NUTS (MCMC, through BlackJAX), or a simple
best fit. They all use the same model, and you get back samples you can turn
into maps, error bars or anything else you compute from the model.

## Example

A positive map, observed with noise at 300 scattered points and fitted with
NUTS:

```python
import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from spectracles import FourierGP, Matern32, Parameter, SpatialDataGeneric, build_model
from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior
from spectracles.priors import LogNormal


class LineMap(eqx.Module):
    log_flux: FourierGP

    def __call__(self, data):
        return jnp.exp(self.log_flux(data))


kernel = Matern32(
    length_scale=Parameter(prior=LogNormal(0.5, 0.2)),
    variance=Parameter(prior=LogNormal(1.0, 0.5)),
)
model = build_model(LineMap, FourierGP(n_modes=(15, 15), kernel=kernel))
latent = LatentSpace(model)  # the parameters to fit

xy = random.uniform(random.PRNGKey(0), (2, 300), minval=-3.0, maxval=3.0)
spaxels = SpatialDataGeneric(x=xy[0], y=xy[1], idx=jnp.arange(300))
truth = latent.sample_prior_latent(random.PRNGKey(1))  # fake data from a random draw
flux = latent.to_model(truth)(spaxels) + 0.05 * random.normal(random.PRNGKey(2), (300,))
ivar = jnp.full(300, 0.05**-2)

posterior = Posterior(latent, lambda m: m(spaxels), GaussianLikelihood(flux, ivar))

# Sample with NUTS
warmup = blackjax.window_adaptation(blackjax.nuts, posterior.log_density)
(state, params), _ = warmup.run(random.PRNGKey(3), latent.zeros(), num_steps=300)
nuts = blackjax.nuts(posterior.log_density, **params)


def step(state, key):
    state, _ = nuts.step(key, state)
    return state, state.position


_, samples = jax.lax.scan(step, state, random.split(random.PRNGKey(4), 300))

posterior.summary(samples, true_latents=truth)  # fitted parameters vs the truth
maps = posterior.draws(samples, to="model")(spaxels)  # the model for every sample
mean_map, std_map = maps.mean(), maps.std()
```

The [docs](https://tomhilder.github.io/spectracles/inference/) go through this
step by step, and the
[worked example](https://tomhilder.github.io/spectracles/examples/inference/)
fits the same data with MGVI, NUTS and a best fit.

## Installation

```sh
pip install spectracles            # or: uv add spectracles
pip install "spectracles[nifty]"   # to use MGVI and geoVI
pip install blackjax               # to use NUTS
```

To run on a GPU, build [jax-finufft](https://github.com/flatironinstitute/jax-finufft)
from source as its README describes (this needs FFTW). To work on spectracles
itself, clone this repository and run `uv sync` (which also builds jax-finufft
from source).

## Features

- Gaussian-process maps with Matérn and squared-exponential kernels
- Spectral lines and other components whose parameters are maps
- Parameters shared between components (e.g. one kernel for two lines)
- Priors on any parameter, and fixing parameters at chosen values
- Bad pixels and NaNs masked in the likelihood
- Saving and loading models, and readable summaries of models and fits

## Stability

From 1.0.0, spectracles follows semantic versioning: anything in
`spectracles`, `spectracles.priors`, `spectracles.inference` or
`spectracles.nifty` only changes in a major release, after a deprecation
warning. `spectracles.lvm_models` and the optimisation schedules
(`OptimiserFrame`) are still changing. See the
[roadmap](https://tomhilder.github.io/spectracles/roadmap/) for what's planned.

## Documentation

[tomhilder.github.io/spectracles](https://tomhilder.github.io/spectracles/)

## Citation

Coming soon.

## License

MIT
