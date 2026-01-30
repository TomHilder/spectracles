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

## Quick Start

```python
import spectracles as sp
import jax.numpy as jnp

# Define your model as an equinox Module with Parameters
class MyModel(eqx.Module):
    amplitude: sp.Parameter
    scale: sp.Parameter

    def __init__(self):
        self.amplitude = sp.Parameter(initial=1.0)
        self.scale = sp.Parameter(initial=1.0)

    def __call__(self, x):
        return self.amplitude.val * jnp.sin(self.scale.val * x)

# Build a ShareModule that handles parameter sharing
model = sp.build_model(MyModel)

# Define a loss function
def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

# Create an optimization schedule
schedule = sp.build_schedule(
    model, loss_fn,
    phases=[
        (100, 0.1),   # 100 steps at lr=0.1
        (50, 0.01),   # 50 steps at lr=0.01
    ],
    params={
        "amplitude": sp.free_in(0, 1),  # Free in both phases
        "scale": sp.free_after(1),       # Free only in phase 1
    },
)

# Run optimization
schedule.run_all(x=x_data, y=y_data)
final_model = schedule.model_history[-1]
```

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
