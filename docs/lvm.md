# LVM data with lvm_tools

[lvm_tools](https://tomhilder.github.io/lvm_tools/) reads SDSS-V Local Volume
Mapper data reduction pipeline (DRP) files and prepares them for spectracles.
It selects a wavelength window and a region of sky, drops bad fibres and
pixels, scales the flux, and gives each spaxel's position in arcseconds from
the centre of the field. Files are read lazily with dask, so only the window
you ask for is loaded.

```bash
pip install "lvm-tools[spectracles]>=0.2"
```

This page fits an Hα map to LVM data, from DRP files to a map on the sky. The
code needs your own data files, so it isn't run when the docs are built.

## Loading the data

Each DRP file (an `lvmSFrame`) is one exposure of one pointing. Load the ones
you want and combine them:

```python
# sketch
from pathlib import Path

import jax
import jax.numpy as jnp

from lvm_tools import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection

jax.config.update("jax_enable_x64", True)

files = sorted(Path("path/to/drp").glob("lvmSFrame-*.fits"))
tiles = LVMTileCollection.from_tiles([LVMTile.from_file(f) for f in files])
```

`DataConfig` says which part of the data to use. Here, a 20 Å window around Hα;
the sky region defaults to everything in the files:

```python
# sketch
config = DataConfig.from_tiles(tiles, λ_range=(6553.0, 6573.0))
data = FitDataBuilder(tiles, config).build()
```

`print(config)` shows every setting, including how bad data are handled and
how the flux is scaled. The
[lvm_tools docs](https://tomhilder.github.io/lvm_tools/) cover the options.

## What you get

`data` holds everything the fit needs:

| | Shape | |
|---|---|---|
| `data.flux`, `data.i_var` | (wavelengths, spaxels) | flux and inverse variance, scaled |
| `data.mask` | (wavelengths, spaxels) | `True` where a pixel is good |
| `data.λ` | (wavelengths,) | wavelengths in Å |
| `data.lsf_σ` | (wavelengths, spaxels) | the line-spread function's width in Å |
| `data.sky_data` | | the spaxel positions, as a `SpatialDataLVM` |

`data.sky_data` is what a spectracles model is evaluated at. Its positions `x`
and `y` are each spaxel's offset east and north of the field's centre
(`data.sky_centre`), in arcseconds, on the tangent plane. It also carries each
spaxel's tile, IFU and fibre, for the per-tile and per-IFU models in
`spectracles.model.spatial`.

The flux and inverse variance are scaled consistently, so fitted fluxes are in
the scaled units. `config.normalise_F_scale` and `config.normalise_F_offset`
convert them back.

By default the flux is divided by its maximum. If there are bright stars in the
field, the maximum is a star, and nebular lines end up small in the scaled
units. `DataConfig.from_tiles(..., normalise_F_strategy="98 only")` scales by
the 98th percentile instead.

## The model

An Hα line whose flux, velocity and width are all fields, plus a constant
continuum per spaxel. The width is a typical value times a positive field, so
it can vary across the sky. The line-spread function is a `Known` value per
spaxel, taken from the data and added to the line's width:

```python
# sketch
import equinox as eqx

from spectracles import FourierGP, Known, Matern32, Parameter, PerSpaxel, build_model
from spectracles.priors import LogNormal, Normal

C_KMS = 299_792.458
HALPHA = 6562.8


class Halpha(eqx.Module):
    log_flux: FourierGP
    velocity: FourierGP
    log_width: FourierGP
    typical_width: Parameter
    lsf: Known
    continuum: PerSpaxel

    def __call__(self, λ, spaxels):
        centre = HALPHA * (1 + self.velocity(spaxels) / C_KMS)
        width = self.typical_width.val * jnp.exp(self.log_width(spaxels))
        σ = jnp.sqrt(width**2 + self.lsf.val[spaxels.idx] ** 2)
        line = jnp.exp(self.log_flux(spaxels)) / (σ * jnp.sqrt(2 * jnp.pi))
        return line * jnp.exp(-0.5 * ((λ - centre) / σ) ** 2) + self.continuum(spaxels)
```

The fields work in the same units as the positions, so length scales are in
arcseconds. A `Domain` describes the region the spaxels cover, and each field
chooses its Fourier modes for the length scales its prior allows (see
[Geometry and discretisation](discretisation.md)):

```python
# sketch
from spectracles import Domain

spaxels = data.sky_data
domain = Domain.from_points(spaxels.x, spaxels.y)
n_spaxels = data.flux.shape[1]


def field(variance_prior):
    kernel = Matern32(
        length_scale=Parameter(prior=LogNormal(60.0, 20.0)),  # arcsec
        variance=Parameter(prior=variance_prior),
    )
    return FourierGP(kernel, domain=domain)


model = build_model(
    Halpha,
    log_flux=field(LogNormal(1.0, 0.5)),
    velocity=field(LogNormal(100.0, 50.0)),  # (km/s)²
    log_width=field(LogNormal(0.1, 0.05)),
    typical_width=Parameter(prior=LogNormal(0.3, 0.2)),  # Å
    lsf=Known(jnp.median(data.lsf_σ, axis=0)),
    continuum=PerSpaxel(n_spaxels, Parameter(dims=n_spaxels, prior=Normal(0.0, 0.05))),
)
```

## The posterior

The forward model evaluates the model at every wavelength, which gives an array
with the same (wavelengths, spaxels) shape as `data.flux`. Pass `data.mask` to
the likelihood so that bad pixels are skipped:

```python
# sketch
from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior


def forward(m):
    return jax.vmap(m, in_axes=(0, None))(data.λ, spaxels)


posterior = Posterior(
    LatentSpace(model),
    forward,
    GaussianLikelihood(data.flux, data.i_var, mask=data.mask),
)
```

From here, fitting and results are the same as for any other model: see
[Priors and inference](inference.md).

## Maps on the sky

Fields can be evaluated anywhere in the domain, so a map can be drawn on a
regular grid rather than at the fibres. `data.sky_to_radec` turns the grid's
offsets back into RA and Dec for the axes:

```python
# sketch
from spectracles import SpatialDataGeneric

gx, gy = jnp.meshgrid(
    jnp.linspace(spaxels.x.min(), spaxels.x.max(), 200),
    jnp.linspace(spaxels.y.min(), spaxels.y.max(), 200),
)
grid = SpatialDataGeneric(x=gx, y=gy, idx=jnp.arange(gx.size))
ra, dec = data.sky_to_radec(gx, gy)

flux = posterior.map_samples(lambda m: jnp.exp(m.log_flux(grid)), samples)
flux_map = flux.mean().reshape(gx.shape)
```

Areas far from any fibre are only constrained by the prior, so mask or grey
them out when you show the map.

## Other things in the data

- `data.v_bary`: the barycentric velocity correction for each spaxel, in km/s.
- `data.mjd`: the time of each exposure.
- `data.tile_idx`, `data.ifu_idx`, `data.fibre_idx`: which tile, IFU and fibre
  each spaxel comes from, also in `data.sky_data`.
- `data.with_sky_origin(ra, dec)`: the same data with offsets measured from a
  centre of your choice, e.g. to put several datasets on one frame.
