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

## Glasses for your spectra

Spectracles models IFU spectra as continuous functions of sky position. Line
fluxes, velocities and widths are Gaussian-process fields, represented by
their Fourier modes and evaluated exactly at each spaxel's position (with
non-uniform FFTs), so the data's irregular sampling is never regridded.

You write the model as an [Equinox](https://github.com/patrick-kidger/equinox)
module, attach priors to its parameters, and get one posterior that every
backend fits in the same coordinates: MGVI and geoVI (NIFTy.re), NUTS
(BlackJAX) or MAP (optax). Results come back as draws of anything you can
compute from the model.

## Example

A positive field, observed with noise at 300 scattered points, then sampled
with NUTS:

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
latent = LatentSpace(model)  # standard-normal coordinates for every free parameter

xy = random.uniform(random.PRNGKey(0), (2, 300), minval=-3.0, maxval=3.0)
spaxels = SpatialDataGeneric(x=xy[0], y=xy[1], idx=jnp.arange(300))
truth = latent.sample_prior_latent(random.PRNGKey(1))  # simulate from the prior
flux = latent.to_model(truth)(spaxels) + 0.05 * random.normal(random.PRNGKey(2), (300,))
ivar = jnp.full(300, 0.05**-2)

posterior = Posterior(latent, lambda m: m(spaxels), GaussianLikelihood(flux, ivar))

# Any backend works on posterior.log_density; here, NUTS
warmup = blackjax.window_adaptation(blackjax.nuts, posterior.log_density)
(state, params), _ = warmup.run(random.PRNGKey(3), latent.zeros(), num_steps=300)
nuts = blackjax.nuts(posterior.log_density, **params)


def step(state, key):
    state, _ = nuts.step(key, state)
    return state, state.position


_, samples = jax.lax.scan(step, state, random.split(random.PRNGKey(4), 300))

posterior.summary(samples, true_latents=truth)  # table of posterior mean ± std vs truth
maps = posterior.draws(samples, to="model")(spaxels)  # the model at every sample
mean_map, std_map = maps.mean(), maps.std()
```

The [inference guide](https://tomhilder.github.io/spectracles/inference/)
explains each step, and the
[worked example](https://tomhilder.github.io/spectracles/examples/inference/)
fits the same posterior with MGVI, NUTS and MAP and compares them.

## Installation

```sh
pip install spectracles            # or: uv add spectracles
pip install "spectracles[nifty]"   # with MGVI / geoVI through NIFTy.re
```

NUTS needs [BlackJAX](https://github.com/blackjax-devs/blackjax)
(`pip install blackjax`); MAP uses optax, which is included.

The NUFFTs come from [jax-finufft](https://github.com/flatironinstitute/jax-finufft),
which has CPU wheels on PyPI. For GPUs, build jax-finufft from source (this
needs FFTW) as its README describes.

For development, from source:

```sh
git clone git@github.com:TomHilder/spectracles.git
cd spectracles
uv sync  # builds jax-finufft from its git repository, which needs FFTW
uv run pytest
```

## Features

**Modelling**

- **Spatial fields** - Gaussian processes in Fourier space (`FourierGP`, with
  Matérn-1/2, 3/2, 5/2 and squared-exponential kernels), evaluated at
  arbitrary positions with NUFFTs
- **Spectral models** - spectral components (`Gaussian` lines, constants)
  whose parameters are spatial fields
- **Composable models** - plain Equinox modules; `build_model` turns one into
  a model with named parameter paths
- **Parameter sharing** - one parameter used in several places (e.g. a kernel
  shared by two lines) is one parameter
- **Free, fixed and `Known`** - choose what is inferred; fix and free
  parameters by path or glob pattern (`"*.kernel.*"`)

**Inference**

- **Priors on parameters** - `Normal`, `LogNormal`, `Laplace`, `Uniform`, ...;
  every free parameter maps to a standard-normal latent
- **One posterior, any backend** - MGVI/geoVI (NIFTy.re), NUTS (BlackJAX), MAP
  (optax), or anything else that takes a JAX log-density
- **Masked Gaussian likelihoods** - NaNs and bad data points are dropped
- **Results as draws** - summaries, parameter draws, the model or any part of
  it at every sample (`posterior.draws`, `map_samples`), ready for ArviZ
- **Staged fits** - freeze parameters for some VI iterations; `posterior.fix`
  holds them for a whole fit

**Also**

- **Optimisation schedules** - point estimates with your own loss, freeing and
  fixing parameter groups per phase
- **Rich output** - pretty-printed model trees, parameter tables and posterior
  summaries
- **Saving and loading** - `save_model` and `load_model`

## Stability

From 1.0.0, spectracles follows semantic versioning: the public API (the
`spectracles`, `spectracles.priors`, `spectracles.inference` and
`spectracles.nifty` namespaces) only changes incompatibly in a major release,
after a deprecation warning.

Not covered:

- **`spectracles.lvm_models`** is experimental. The LVM-specific models are
  being ported to priors and the latent space one at a time.
- **`OptimiserFrame` and schedules** will move onto the latent space, so that
  optimisation and inference share one way to fit; the `prior_logpdf` losses
  will then be deprecated.

Planned work is on the
[roadmap](https://tomhilder.github.io/spectracles/roadmap/).

## Documentation

[tomhilder.github.io/spectracles](https://tomhilder.github.io/spectracles/):
guides to [priors and inference](https://tomhilder.github.io/spectracles/inference/)
and [schedules](https://tomhilder.github.io/spectracles/schedule_builder/),
worked examples, and the API reference.

## Citation

Coming soon.

## License

MIT
