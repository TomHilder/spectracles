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

Spectracles is a Python library for inferring properties of IFU/IFS spectra as continuous functions of sky position.

It can also be used as a general-purpose statistical model library that extends [`equinox`](https://github.com/patrick-kidger/equinox) to allow for composable models that may have *coupled* parameters. It also implements some other nice features that are a bit awkward in `equinox` out of the box, like easily updating model parameters between fixed and varying.

## Installation

From PyPI with `pip`:

```sh
pip install spectracles
```

Or with `uv` (recommended):

```sh
uv add spectracles
```

From source:

```sh
git clone git@github.com:TomHilder/spectracles.git
cd spectracles
pip install -e .
```

**Note:** `fftw` must be installed or the dependency `jax-finufft` will fail to build.

## Features

- **Priors and inference** - Priors on parameters, and one posterior that MGVI/geoVI (NIFTy.re), NUTS (BlackJAX) and MAP (optax) all fit
- **Staged VI** - Freeze parameter groups across VI iterations with the schedule-builder vocabulary
- **Parameter sharing** - Couple parameters across model components
- **Declarative optimization schedules** - Specify which parameters are free/fixed per phase
- **Glob patterns** - Use wildcards like `"gp.kernel.*"` to match parameters
- **JAX integration** - Built on equinox, fully compatible with JAX transformations
- **Rich output** - Pretty-printed model trees and gradient diagnostics

## Status and plans

- **Experimental: `spectracles.lvm_models`.** The LVM-specific models are not
  covered by the stability promise. They are being ported to the new prior and
  latent-space API one at a time.
- **Planned: one way to fit.** Models can currently be fitted two ways:
  - with `OptimiserFrame` and schedules, using `prior_logpdf` losses;
  - through the latent space (`Posterior.log_density` with NIFTy.re, BlackJAX
    or optax).

  These will be unified later. Optimisation will move to the latent space,
  schedules and VI stages will share one vocabulary, and the `prior_logpdf`
  losses will be deprecated.

- **Planned (1.2): lazy draws for large fields.** `posterior.draws(samples,
  to="model")` evaluates and holds every draw in memory. For many samples of a
  field on many points, summaries will be computed in chunks of samples
  (streaming mean and variance; quantiles in chunks of points), with the full
  draws built only on request.

- **Planned: corner plots from draws.** `Draws.corner(...)`, a small wrapper
  around [`corner`](https://corner.readthedocs.io) (an optional extra):
  hyperparameters only by default (`exclude_fourier_coeffs=True`), labels
  from the parameter paths, vector parameters split into columns, ringed
  scatter points with step histograms, and every `corner` keyword (including
  `fig`, to overlay several methods) passed through.

## Documentation

Full documentation: [tomhilder.github.io/spectracles](https://tomhilder.github.io/spectracles/)

## Citation

Coming soon.

## License

MIT
