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

- **Parameter sharing** - Couple parameters across model components
- **Declarative optimization schedules** - Specify which parameters are free/fixed per phase
- **Glob patterns** - Use wildcards like `"gp.kernel.*"` to match parameters
- **JAX integration** - Built on equinox, fully compatible with JAX transformations
- **Rich output** - Pretty-printed model trees and gradient diagnostics

## Documentation

Full documentation: [tomhilder.github.io/spectracles](https://tomhilder.github.io/spectracles/)

## Citation

Coming soon.

## License

MIT
