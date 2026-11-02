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

Spectracles fits models to IFU data where line or spectrum properties (line flux or velocity, continuum, ...) vary smoothly across the sky. These are modelled as *fields* and so they exist continuous at all sky locations spanned by the data.

More generically, Spectracles is a Python/[JAX](https://github.com/jax-ml/jax) library for statistical models, and particularly for models containing latent fields. It is designed for large-scale problems.

Spectracles has been designed for ease-of-use, both within and beyond the IFU community. Models are modular, and may be built from scratch or composed of many sub-models, which is particularly useful for hierarchical models (like spectrospatial IFU models). Spectracles is compatible with most JAX-supported inference algorithms out-of-the-box, including large-scale variational algorithms ([MGVI](https://arxiv.org/abs/1901.11033) or [geoVI](https://doi.org/10.3390/e23070853) via [NIFTy.re](https://github.com/NIFTy-PPL/NIFTy)) or [NUTS](https://jmlr.org/papers/v15/hoffman14a.html) (e.g. [BlackJAX](https://github.com/blackjax-devs/blackjax) or [NumPyro](https://github.com/pyro-ppl/numpyro)).

## Example

In this example, we fit a positive-valued field using observations at 300 scattered points. The inference machinery in this case is [BlackJAX](https://github.com/blackjax-devs/blackjax):

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

To run on a GPU, build [jax-finufft](https://github.com/flatironinstitute/jax-finufft) from source as its README describes (this needs [FFTW](https://www.fftw.org)). To work on spectracles itself, clone this repository and run `uv sync` with [uv](https://github.com/astral-sh/uv), which also builds jax-finufft from source.

## Features

IFU:

- Models including emission lines whose parameters are fields
- Models for systematics, for example per-pointing or per-spaxel parameters

Fields:

- Fast stationary Gaussian-process fields on arbitrary points via the equispaced Fourier representation ([EFGP](https://doi.org/10.1137/23M1565310)) and [FINUFFT](https://github.com/flatironinstitute/finufft) (through [jax-finufft](https://github.com/flatironinstitute/jax-finufft))
- Squared exponential (RBF) and Matern 1/2, 3/2 and 5/2 kernels
- Easy dense evaluation after fitting (i.e. between the data points)
- Runs on GPU through JAX

Generic:

- Parameter sharing: pass the same `Parameter` (or kernel, or whole field) to several components, e.g. one velocity field for every line, and it is fitted as one parameter
- Priors and fixed parameters set by path or glob pattern, e.g. `model.with_priors({"**.length_scale": LogNormal(0.5, 0.2), "log_flux.*.variance": "fixed"})`
- Fix parameters at chosen values for a fit with `posterior.fix({"log_flux.kernel.length_scale": 0.3})`
- Works with any JAX inference algorithm: `posterior.log_density` is a plain JAX function, and `to_nifty(posterior)` hands the same posterior to NIFTy.re
- Gaussian likelihoods that drop NaNs, zero inverse variances and masked pixels automatically: `GaussianLikelihood(data, ivar, mask=good)`
- The model, or any sub-component, at every posterior sample: `posterior.draws(samples, to="model").log_flux(spaxels).mean()`
- Fast mapping of any function over posterior or prior samples
- PPL-like features including easy prior specification and sampling
- Plots of the model structure, showing which parameters are shared where: `model.plot_model_graph()`
- Saving and loading models
- Readable summaries of models and fits
- Optimisation schedules for staged best fits (experimental; see the [docs](https://tomhilder.github.io/spectracles/schedule_builder/))

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

If you find this software useful, please cite the paper:

[Hilder et al. (2026)](https://doi.org/10.3847/1538-4357/ae274b):

```bibtex
@article{Hilder2026,
    author        = {{Hilder}, Thomas and {Casey}, Andrew R. and {Dalcanton}, Julianne J. and
                     {Kreckel}, Kathryn and {Stutz}, Amelia M. and {Singh}, Amrita and
                     {Blanc}, Guillermo A. and {S{\'a}nchez}, Sebasti{\'a}n F. and
                     {M{\'e}ndez-Delgado}, J. E. and {Saydjari}, Andrew K. and
                     {Vargas-Herrera}, Luciano and {Drory}, Niv and {Bizyaev}, Dmitry and
                     {Fern{\'a}ndez-Trincado}, Jos{\'e} G. and {Rom{\'a}n-Z{\'u}{\~n}iga}, Carlos G. and
                     {Kollmeier}, Juna A. and {Johnston}, Evelyn J.},
    title         = {{Unified Spectrospatial Forward Models: Spatially Continuous Maps of Weak
                     Emission Lines in the Rosette Nebula with SDSS-V LVM}},
    journal       = {The Astrophysical Journal},
    year          = {2026},
    volume        = {997},
    number        = {2},
    eid           = {339},
    doi           = {10.3847/1538-4357/ae274b},
    eprint        = {2510.07395},
    archivePrefix = {arXiv},
    primaryClass  = {astro-ph.GA},
}
```

You should also cite [JAX](https://github.com/jax-ml/jax#citing-jax) and
[Equinox](https://github.com/patrick-kidger/equinox#citation).

If you also use [NIFTy.re](https://github.com/NIFTy-PPL/NIFTy), [NumPyro](https://github.com/pyro-ppl/numpyro), [BlackJAX](https://github.com/blackjax-devs/blackjax), or any other inference machinery that is not part of Spectracles, make sure you cite the relevant works!

## License

MIT

## Other useful software

Spectracles has many possible applications, but there are also many other great libraries out there! Here are just a few that may be useful to users of Spectracles:

- [NIFTy.re](https://github.com/NIFTy-PPL/NIFTy): JAX library for Gaussian-process
  models and variational inference (MGVI, geoVI), widely used for large imaging
  problems. Spectracles can hand its posteriors to NIFTy.re.
- [GraphGP](https://github.com/stanford-ism/graphgp): scalable Gaussian processes in
  JAX using nearest-neighbour graphs (Vecchia's approximation), in linear time
  and memory at arbitrary points.
- [efgp_jax](https://github.com/pgree/efgp_jax): JAX implementation of the
  equispaced Fourier Gaussian processes that Spectracles' fields are based on.
- [tinygp](https://github.com/dfm/tinygp) and
  [GPJax](https://github.com/JaxGaussianProcesses/GPJax): general-purpose
  Gaussian-process libraries in JAX.
- [pPXF](https://pypi.org/project/ppxf/): full-spectrum fitting of stellar and gas
  kinematics and stellar populations, usually spaxel by spaxel or bin by bin.
- [PowerBin](https://pypi.org/project/powerbin/) (successor to
  [VorBin](https://pypi.org/project/vorbin/)): adaptive spatial binning of IFU
  data to a target signal-to-noise.
