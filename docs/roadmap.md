# Roadmap

Planned work, roughly in order. Nothing here is covered by the
[stability promise](index.md#stability) until it is released.

## Writing results to FITS

Fit results will be writable to FITS files, the format most astronomy tools
read: posterior summaries (mean, standard deviation, quantiles) and draws of
model quantities, evaluated either on the data's spatial points (as binary
tables) or on a regular grid over the `Domain` (as image extensions, with a
celestial WCS when the domain's reference point on the sky is given). Units are
written in the headers, and the parameter paths name the columns and
extensions, so a file can be read without spectracles. astropy will be an
optional extra.

## One way to fit

Models can currently be fitted two ways: with `OptimiserFrame` and schedules
(using `prior_logpdf` losses), or through the latent space
([`Posterior`](inference.md) with NIFTy.re, BlackJAX or optax). These will be
unified: optimisation will move to the latent space, schedules and VI stages
will share one vocabulary, and the `prior_logpdf` losses will be deprecated.

## Power-law kernels with deviations

A kernel whose power spectrum is a power law plus smooth deviations from it,
as in NIFTy's correlated-field model: the log amplitude is linear in log k,
plus an integrated Wiener process in log k with a learnable flexibility (and
optionally asperity), with its linear trend removed so the slope stays
identifiable. It suits broadband fields that have no single correlation
length. Alongside it:

- `LogLogLinear`'s amplitude `m_0` will be defined at a pivot wavenumber on
  the mode grid, rather than at k = 1, so that it doesn't depend strongly on
  the slope;
- modes and padding chosen automatically for kernels without a length scale;
- the discretisation diagnostic and docs extended to these kernels.

## Parameters in lists, tuples and dicts (1.1)

Parameters, and modules holding them, must currently be named attributes of a
module; `build_model` raises a clear error otherwise. 1.1 will support lists,
tuples and dicts, with paths such as `lines[0].flux` or `lines['Ha'].flux`, so
models with a variable number of components (e.g. emission lines) can be built
in a loop. Existing paths are unchanged.

It also brings back `MultiFourierGP`, which holds its kernels in a list and
raises `NotImplementedError` until then. It batches several fields through one
NUFFT, since fields in one model usually share a frequency grid, to make
multi-field models faster on GPU.
([#4](https://github.com/TomHilder/spectracles/issues/4))

## Lazy draws for large fields (1.2)

`posterior.draws(samples, to="model")` evaluates and holds every draw in
memory. For many samples of a field on many points, summaries will be computed
in chunks of samples (streaming mean and variance; quantiles in chunks of
points), with the full draws built only on request.

## 1D and 3D fields

Fields are two-dimensional: `FourierGP` takes positions `x`, `y` and a square
`Domain`. Fields in one dimension (e.g. along wavelength or time) and three
(e.g. position on the sky and wavelength, or a volume) will use the same
Fourier representation, through jax-finufft's 1D and 3D transforms. That means
a `Domain` and spatial data with any number of axes, kernels and the
discretisation diagnostic in d dimensions (the error rates depend on d), and
the automatic choice of modes and padding per axis.

## Saving fits rather than models

`save_model` pickles the whole model with dill, including the code of your
model classes, so a saved file is tied to the Python and package versions that
wrote it. Since the model's structure comes from your own code and can be
rebuilt, saving is likely to change to store only what a fit produces: the
parameter values or latents (and samples), loaded back into a model you build
the same way. `save_model` and `load_model` would then be deprecated.

## Corner plots from draws

`Draws.corner(...)`, a small wrapper around
[`corner`](https://corner.readthedocs.io) (an optional extra): hyperparameters
only by default (`exclude_fourier_coeffs=True`), labels from the parameter
paths, vector parameters split into columns, ringed scatter points with step
histograms, and every `corner` keyword (including `fig`, to overlay several
methods) passed through.

## Docs on Zensical

The documentation is built with Material for MkDocs, which reaches end of life
in May 2027. It will move to [Zensical](https://zensical.org), its successor
from the same team, which reads the existing `mkdocs.yml`. lvm_tools' docs,
built the same way, will move with it.
