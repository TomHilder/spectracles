# Model Components

Building blocks for spectrospatial models. Combine them inside your own Equinox
module and wrap it with [`build_model`](share_module.md#building-models).

## Parameters

::: spectracles.Parameter

::: spectracles.ConstrainedParameter

::: spectracles.Known

## Kernels

Kernels define the power spectrum of a `FourierGP`. A subclass only needs to
implement `feature_weights(freqs)`, returning `sqrt(P(k))` on the mode grid.

::: spectracles.Kernel

::: spectracles.Matern12

::: spectracles.Matern32

::: spectracles.Matern52

::: spectracles.SquaredExponential

::: spectracles.model.kernels.LogLogLinear

## Spatial Models

::: spectracles.SpatialModel

::: spectracles.FourierGP

::: spectracles.model.spatial.FourierBasis

::: spectracles.model.spatial.MultiFourierGP

::: spectracles.PerSpaxel

::: spectracles.model.spatial.PerIFU

::: spectracles.model.spatial.PerTile

::: spectracles.model.spatial.PerIFUAndTile

::: spectracles.model.spatial.PerTilePinned

## Spectral Models

::: spectracles.SpectralSpatialModel

::: spectracles.Constant

::: spectracles.WindowConstant

::: spectracles.Gaussian

## Data

`SpatialData` is the base class for the points a spatial model is evaluated at.
Subclass it to carry extra per-point fields your models need.

::: spectracles.SpatialData

::: spectracles.SpatialDataGeneric

::: spectracles.SpatialDataLVM

## Saving and Loading

::: spectracles.save_model

::: spectracles.load_model
