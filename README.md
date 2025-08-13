<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!-- [![PyPI Package][pypi-shield]][pypi-url] -
<!-- [![JOSS][JOSS-shield]][JOSS-url] -->

<!-- ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/TomHilder/wakeflow/Tests.yml?label=tests&style=flat-square)
[![Docs][docs-status-shield]][docs-status-url] -->

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
- [x] Instead of replacing shared leaves with `0`, replace with some class/object instead
- [ ] Nicer `__repr__` for `ShareModule` that actually says the memory address
- [ ] Add memory address to the top of `print_model_tree`
- [ ] Support tuples, lists and dicts of models as attributes of models
- [ ] Handle non-odd number of modes
- [ ] Write better tests
- [ ] Rigorously type check the tests
