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
    <img src="https://raw.githubusercontent.com/TomHilder/spectracles/main/logo.png" alt="spectrackles" width="420">
  </a>

<!--  <h3 align="center">Wakeflow</h3> -->

  <p align="center">
    Unified spectrospatial models for integral field spectroscopy in jax
  </p>
</div>

<!-- <div align="center">
<img src="https://raw.githubusercontent.com/TomHilder/spectracles/main/logo.png" alt="spectracles" width="420"></img>
</div> -->

## Glasses for your spectra

Spectracles is a Python library for inferring properties of IFU/IFS spectra as continuous functions of sky position.

It can also be used as a general-purpose statistical model library that extends [`equinox`](https://github.com/patrick-kidger/equinox) to allow for composable models that may have *coupled* parameters. It also implements some other nice features that are a bit awkward in `equinox` out of the box, like easily updating model parameters between fixed and varying.

## Installation

Easiest is from PyPI either with `pip`

```sh
pip install spectracles
```

or `uv` (recommended)

```sh
uv add spectracles
```

Or, you can clone and build from source

```sh
git clone git@github.com:TomHilder/spectracles.git
cd spectracles
pip install -e .
```

**Important:** `fftw` must be installed or the required dependency`jax-finufft` will fail to build.

## Usage

TODO

## Citation

TODO

## Help

TODO

### TODO

- [ ] Relax version requirements from being strictly my environment (which is very up-to-date)
- [ ] Migrate some stuff from the `model` subpackage to a new subpackage called `spectroscopy` or something, idea to separate generic modelling stuff from applied spectrospatial models stuff
- [x] Instead of replacing shared leaves with `0`, replace with some class/object instead
- [ ] Nicer `__repr__` for `ShareModule` that actually says the memory address
- [ ] Add memory address to the top of `print_model_tree`
- [ ] Support tuples, lists and dicts of models as attributes of models
- [ ] Handle non-odd number of modes
- [ ] Write better tests
- [ ] Rigorously type check the tests
