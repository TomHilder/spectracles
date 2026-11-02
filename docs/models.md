# Building models

A spectracles model is an ordinary [Equinox](https://docs.kidger.site/equinox/)
module. You write a class that holds spectracles components (fields,
per-spaxel terms, parameters) and say in `__call__` how they combine into a
prediction. `build_model` then turns it into a model you can inspect, change
and fit.

This page builds a small two-line model step by step. To fit it, see
[Priors and inference](inference.md).

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from spectracles import (
    FourierGP,
    Known,
    Matern32,
    Parameter,
    PerSpaxel,
    SpatialDataGeneric,
    build_model,
)
from spectracles.priors import LogNormal, Normal

jax.config.update("jax_enable_x64", True)
```

## Where the data are

A model is evaluated at a set of points on the sky. `SpatialDataGeneric` holds
their positions `x` and `y`, and an index `idx` for each point (usually which
spaxel it is):

```python
N_SPAXELS = 400
xy = random.uniform(random.PRNGKey(0), (2, N_SPAXELS), minval=-3.0, maxval=3.0)
spaxels = SpatialDataGeneric(x=xy[0], y=xy[1], idx=jnp.arange(N_SPAXELS))
```

The positions can be anywhere: they don't need to lie on a grid. Fields are
defined over a box a bit larger than the data, so scale your coordinates to
roughly ±3 (see the roadmap for physical units).

For LVM data, `SpatialDataLVM` also carries the tile, IFU and fibre of each
point. [lvm_tools](https://tomhilder.github.io/lvm_tools/) builds it for you
from the DRP output: its `FitData.αδ_data` is a `SpatialDataLVM`. See
[LVM data](lvm.md). If your models need something else per point (an exposure number, say),
subclass `SpatialData` and add it:

```python
from equinox import field

from spectracles import SpatialData
from spectracles.model.data import convert_to_flat_array


class ExposureData(SpatialData):
    x: jax.Array = field(converter=convert_to_flat_array)
    y: jax.Array = field(converter=convert_to_flat_array)
    idx: jax.Array = field(converter=convert_to_flat_array)
    exposure: jax.Array = field(converter=convert_to_flat_array)
```

## Fields

A field is a quantity defined continuously across the sky, such as a line flux
or a velocity. `FourierGP` is a Gaussian-process field. It needs a kernel,
which sets how smooth the field is, and the number of Fourier modes in each
direction (two odd numbers; more modes allow finer structure):

```python
field = FourierGP(
    n_modes=(25, 25),
    kernel=Matern32(
        length_scale=Parameter(prior=LogNormal(1.0, 0.3)),
        variance=Parameter(prior=LogNormal(1.0, 0.5)),
    ),
)
```

The kernels are `Matern12`, `Matern32`, `Matern52` and `SquaredExponential`,
from roughest to smoothest. Each takes a `length_scale` and a `variance`.

A field can be evaluated at any points, not just where the data are. That is
how you make a dense map after fitting.

## Per-spaxel terms

Some quantities don't vary continuously across the sky, such as a calibration
offset for each spaxel.
`PerSpaxel` has one value per spaxel, looked up by `idx`:

```python
per_spaxel = PerSpaxel(N_SPAXELS, Parameter(dims=N_SPAXELS, prior=Normal(0.0, 0.01)))
```

For LVM data there are also `PerIFU`, `PerTile` and `PerIFUAndTile` in
`spectracles.model.spatial`.

## Your model

Now write the model itself. Here, two emission lines share one velocity field,
and each has its own flux field. The rest wavelengths are `Known`: constants
that are never fitted.

```python
C_KMS = 299_792.458


class Line(eqx.Module):
    rest_wavelength: Known
    log_flux: FourierGP
    velocity: FourierGP
    width: Parameter

    def __call__(self, wavelength, data):
        centre = self.rest_wavelength.val * (1 + self.velocity(data) / C_KMS)
        flux = jnp.exp(self.log_flux(data))
        σ = self.width.val
        return flux / (σ * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * ((wavelength - centre) / σ) ** 2)


class TwoLines(eqx.Module):
    halpha: Line
    nii: Line
    offset: PerSpaxel

    def __call__(self, wavelength, data):
        return self.halpha(wavelength, data) + self.nii(wavelength, data) + self.offset(data)
```

A parameter's value is `.val`. Any JAX code can go in `__call__`.

## Building it, and sharing

`build_model` takes the class and its components. To share something between
components, pass the same object to both.

First make the components. There is one velocity field, `velocity`, and a
function that makes a new flux field each time it is called:

```python
velocity = FourierGP(
    n_modes=(25, 25),
    kernel=Matern32(
        length_scale=Parameter(prior=LogNormal(1.0, 0.3)),
        variance=Parameter(prior=LogNormal(400.0, 200.0)),  # (km/s)²
    ),
)


def flux_field():
    kernel = Matern32(
        length_scale=Parameter(prior=LogNormal(1.0, 0.3)),
        variance=Parameter(prior=LogNormal(1.0, 0.5)),
    )
    return FourierGP(n_modes=(25, 25), kernel=kernel)
```

Then build the model. The same `velocity` object goes to both lines, so the
model has one velocity field, not two. Each line gets its own flux field:

```python
halpha = Line(
    rest_wavelength=Known(6562.8),
    log_flux=flux_field(),
    velocity=velocity,  # shared
    width=Parameter(prior=LogNormal(1.0, 0.2)),
)
nii = Line(
    rest_wavelength=Known(6583.5),
    log_flux=flux_field(),
    velocity=velocity,  # shared
    width=Parameter(prior=LogNormal(1.0, 0.2)),
)
offset = PerSpaxel(N_SPAXELS, Parameter(dims=N_SPAXELS, prior=Normal(0.0, 0.01)))

model = build_model(TwoLines, halpha=halpha, nii=nii, offset=offset)
```

A shared parameter is fitted once, with one prior. Two separate parameters that
happen to have the same prior, like the two `width`s here, are fitted
independently.

## Looking at the model

`get_parameter_summary` lists every parameter by its path, with its shape,
prior and whether it is free. A shared parameter appears at every location it
is used, with "Shared from" naming the one place it is stored:

```python
model.get_parameter_summary()
```

`print_model_tree(show_sharing=True)` prints the structure, and
`plot_model_graph()` draws it, with an arrow from each component to what it
uses:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(18, 8))
model.plot_model_graph(ax=ax, show=False)
fig.savefig("model_graph.png")
```

![The two-line model as a graph: both lines point to one velocity field](img/model_graph.png)

Both lines point to the same `velocity` node: it is one field.

## Evaluating

Call the model with a wavelength and the data. To get a spectrum at every
spaxel, map over wavelengths with `jax.vmap`:

```python
wavelengths = jnp.linspace(6550.0, 6600.0, 200)
spectra = jax.vmap(model, in_axes=(0, None))(wavelengths, spaxels)
print(spectra.shape)  # (wavelengths, spaxels)
```

```text
(200, 400)
```

To call a single part of the model, such as the velocity field on a dense
grid, use `for_evaluation()` first. It gives a copy in which shared values are
filled in everywhere:

```python
gx, gy = jnp.meshgrid(jnp.linspace(-3, 3, 100), jnp.linspace(-3, 3, 100))
grid = SpatialDataGeneric(x=gx, y=gy, idx=jnp.arange(gx.size))
velocity_map = model.for_evaluation().nii.velocity(grid).reshape(gx.shape)
```

(Without `for_evaluation()`, `model.nii.velocity` would fail: the shared field
lives at `halpha.velocity`, and `nii` only holds a reference to it.) Use the
copy for evaluation only; keep the original for fitting.

## Setting and fixing values

`set` takes a dict from path to value (an array of the parameter's shape), and
returns a new model:

```python
model = model.set({"halpha.width": jnp.array([1.2]), "nii.width": jnp.array([1.2])})
```

`set_fixed_status` fixes or frees parameters by path. A fixed parameter keeps
its value and isn't fitted:

```python
model = model.set_fixed_status(["offset.spaxel_values"], [True])
```

Priors and fixing can also be set by glob pattern with `with_priors`, e.g.
`model.with_priors({"*.width": LogNormal(1.0, 0.2), "offset.*": "fixed"})`.
[Free, fixed and Known](inference.md#free-fixed-and-known) explains the
difference between the three kinds of parameter.

## Saving and loading

```python
from pathlib import Path

from spectracles import load_model, save_model

save_model(model, Path("two_lines"))
model = load_model(Path("two_lines"))
```

A saved file holds the whole model, including the code of your classes, so it
only loads with the Python and package versions that wrote it. This is likely
to change (see the [roadmap](roadmap.md#saving-fits-rather-than-models)).

## Built-in spectral components

For simple cases you don't need to write the line yourself:
`Gaussian(A, λ0, σ)` is a Gaussian line whose amplitude, centre and width are
each a spatial model, `Constant(c)` is a flat continuum, and
`WindowConstant(c, λ_min, λ_max)` is flat within a wavelength window. See the
[API reference](api/models.md).

## Current limitations

- **Parameters must be attributes**, not items in a list, tuple or dict.
  Putting one in a container raises an error; support is planned for 1.1.
- **`MultiFourierGP`** (several fields evaluated together) is disabled until
  1.1.
- **Fields are stationary**: the kernel is the same everywhere on the sky.
