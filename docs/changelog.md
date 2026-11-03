# Changelog

All notable changes to **spectracles** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-03

### Added

- Priors on parameters (`spectracles.priors`: `StdNormal`, `Normal`, `LogNormal`, `Laplace`, `Uniform`). Give one with `Parameter(prior=...)`, or set them afterwards by glob with `model.with_priors({...})`; `get_priors()` lists them and model summaries have a Prior column. `FourierGP` coefficients have a standard-normal prior by default.
- A latent space, likelihoods and a posterior (`spectracles.inference`: `LatentSpace`, `Likelihood`, `GaussianLikelihood`, `Posterior`). Every free parameter gets a standard-normal latent, so one model gives the same posterior, in the same coordinates, to MGVI/geoVI (NIFTy.re), NUTS (BlackJAX) and MAP (optax). `GaussianLikelihood` drops masked, non-finite and zero-weight data points.
- `spectracles.nifty`: `to_nifty(posterior)` and `stages(...)` for staged variational inference with NIFTy.re. NIFTy is an optional extra: `pip install "spectracles[nifty]"`.
- Results from any backend's samples: `Posterior.summary` (with optional true values), `draws` (returning `Draws` of parameter values, or `ModelDraws` with `to="model"` to evaluate any part of the model at every sample), `map_samples`, `to_model` and `to_params`. `Draws` summaries reduce over the sample axes only (`axis=` picks some; `reduce(fn)` takes any statistic).
- `Posterior.fix` and `LatentSpace.fix`: hold chosen parameters at given values for a whole fit.
- `ShareModule.for_evaluation()`: an evaluable copy of a model, with shared values filled in.
- `ShareModule.set` accepts a dict from path to value; `build_model` accepts an instance as well as a class.
- Physical units: `Domain(centre, extent)` (or `Domain.from_points(x, y)`) describes the region the data cover, in the data's own units. `FourierGP` and `FourierBasis` take `domain=`, `padding=` (0.15 of the extent on each side by default) and `nufft_tol=`. With a domain, length scales are in the coordinates' units. Positions outside the periodic box raise an error.
- `Domain.grid((n, m), padding=0.0)`: a regular grid of points over the domain (or over a GP's whole box with `padding=gp.padding`), e.g. to map a field after fitting.
- `FourierGP(kernel, domain=...)` chooses its own modes and padding for a tolerance (`tol=1e-3` by default) over the length-scale prior's 1% to 99% quantiles (or `length_scale=`, or `quantiles=`). The choice is recorded in `gp.design`. `n_modes=` and `padding=` can still be given by hand.
- `spectracles.discretisation`: `discretisation_error` (the truncation and aliasing errors of a `FourierGP`, or what they would be with other settings; an array of length scales gives a curve), `required_modes`, `required_padding`, `realised_covariance`, `report` (every `FourierGP` in a model, latent space or posterior, optionally across samples) and `plot_discretisation`.
- Kernels: `spectral_density` and `covariance`, the continuous forms of the built-in kernels.
- Fields: `FourierGP(..., mean=Parameter(...))` adds a constant, and `link="identity" | "exp" | "softplus"` (with `lower=`) transforms the field, e.g. for a positive line flux. With a mean, the GP's own k = 0 mode is dropped and the mean alone sets the field's average over the box.
- `FieldFromRatio` (a base field times 10 to the power of a log-ratio field) and `FieldPlusScatter` (a field plus independent scatter), exported from `spectracles` and built on `mean=` and `link=`. The experimental `lvm_models` keeps its own versions.
- Clear errors where mistakes are common: wrong `dims`, `n_modes` that aren't two odd integers, a `Prior` passed positionally, wrong-sized per-spaxel, per-IFU or per-tile values (previously clamped silently), and a model passed to `Posterior` as `forward`.
- Documentation: Building models, Priors and inference, How it works, Geometry and discretisation and LVM data guides; a worked inference example; a roadmap; this changelog and a contributing guide. The code on the guide pages is run by the tests.
- Python 3.13 and 3.14 are tested in CI and listed in the package classifiers.

### Changed

- `FourierGP` takes the kernel first: `FourierGP(kernel, n_modes=..., ...)`. The old `FourierGP(n_modes, kernel)` still works, with a `DeprecationWarning`.
- `nufft_tol=` on `FourierGP` and `FourierBasis` replaces the module-level `FINUFFT_KWDS` (the default accuracy, 1e-10, is unchanged).
- A parameter with a prior and no initial value starts at the prior's median (it used to start at 0, possibly outside the prior's support).
- `SpatialData` is a base class that `SpatialDataGeneric` and `SpatialDataLVM` subclass, so you can define your own spatial data classes. It used to be a type alias.
- `spectracles.lvm_models` is marked experimental: it is not covered by the API stability promise and is not type-checked.
- `MultiFourierGP` raises `NotImplementedError` until 1.1. Use one `FourierGP` per field.
- A `Parameter` inside a list, tuple or dict raises a clear `TypeError` (support is planned for 1.1).
- Models saved with `save_model` by earlier versions don't load correctly (models have new fields).
- Summary tables wrap long parameter paths instead of truncating them.
- The package passes pyright with no errors (excluding `lvm_models`).

### Deprecated

- `FourierGP(n_modes, kernel)`: use `FourierGP(kernel, n_modes=n_modes)`, or give a `domain` and leave out `n_modes`.
- `zeromode=` and `FourierGP.zeromode`: use `mean=` and `.mean`. Note that with a kernel that has a k = 0 mode (e.g. the Matérn kernels), a `zeromode` now drops that mode, as `mean=` does; it used to add a constant on top of it.
- `ShareModule.get_locked_model()`: use `for_evaluation()`. It still works and warns.

### Removed

- The module-level `FINUFFT_KWDS` setting (see `nufft_tol=` above).

### Fixed

- Path patterns with `**` in `build_schedule` could match the wrong parameter: `"gp.**.ls"` matched `"gp2.k.ls"`. Only patterns that matched wrongly are affected.
- `save_model` works on Python 3.15 for models whose classes are defined in a script.

## [0.7.0] - 2026-10-26

### Added

- `LogLogLinear` kernel (in `spectracles.model.kernels`): a power spectrum that is linear in log-log space, with no k = 0 mode.
- `FourierGP(..., zeromode=...)`: a constant added to the field, for kernels with no k = 0 mode.

### Changed

- Requires Python 3.11 or later (3.10 is no longer tested).
- The documentation is deployed when a version is tagged, refreshed for changes since 0.5.5, and docstring examples render as code.

### Fixed

- The zero mode of a Fourier GP had twice its intended prior variance, so the field's spatial mean varied more than the kernel says (≈0.3% of the variance at a length scale of 0.08 box units, ≈5% at 0.3). This changes results for every `FourierGP`, and slightly for `FourierBasis`.

## [0.6.5] - 2026-03-20

### Added

- Experimental LVM models: fields that sum two fields or scale one, a blended-doublet line ratio model and an [O II] model.

## [0.6.4] - 2026-03-11

### Added

- Experimental LVM models: `LVMModelSingle` using the new field types; kinematics that need not be shared; line ratio model changes.

## [0.6.3] - 2026-02-26

### Added

- `WindowConstant`: a constant confined to a wavelength window.
- `PerTilePinned`: a per-tile value with one tile held fixed, to remove a degeneracy between tiles.
- Experimental LVM line ratio model: a separate continuum per line, a systematic velocity per line and wavelength calibration.

### Fixed

- `FourierGP.prior_logpdf` had its log normalisation off by a factor of 2.
- The log10-ratio convenience function in the LVM ratio model was missing its mean.

## [0.6.2] - 2026-02-20

### Added

- A custom loss-change criterion in the schedule builder.

## [0.6.1] - 2026-02-03

### Fixed

- Printing a model with shared `Known` parameters no longer fails, and the message about locked models is clearer.
- Spacing and newlines in printed output.

## [0.6.0] - 2026-02-02

### Added

- `Known`, for values that are never optimised.
- `FourierBasis` (unregularised sines and cosines) and `MultiFourierGP`.
- LVM models (`spectracles.lvm_models`): emission lines (`EmissionLine`, `LinkedEmissionLine`), field types built on `FourierGP`, a DRP nuisance model and a line ratio model.
- The parameter-centric schedule builder (`build_schedule`, `fixed_in`, `free_in`, ...).
- Model inspection on `ShareModule`: `parameter_summary`, `get_parameter_paths`, `get_sharing_summary`, `get_shared_components`, `validate_sharing`, `fix_all` / `free_all`, `print_model_tree` and `plot_model_graph` with sharing levels, and Rich-based printing.
- Python 3.10 support.
- Documentation with MkDocs Material and an API reference; tests in CI.

### Changed

- **Breaking:** `Parameter` and `ConstrainedParameter` take the initial value first, then `dims`.
- `OptimiserScheduleUnsafe` renamed to `ManagedOptimiserSchedule`.
- networkx requirement relaxed to 3.4.

## [0.5.5] - 2025-11-19

### Changed

- Requires Python 3.10 or later (was 3.13).

## [0.5.4] - 2025-08-14

### Changed

- Model graphs draw links between components at the same level as curved arrows; old graphing code removed.

### Fixed

- The JAX requirement is a minimum (`jax>=0.5.3`) rather than pinned.

## [0.5.3] - 2025-08-13

### Changed

- Relaxed the JAX requirement.

## [0.5.2] - 2025-08-13

### Changed

- JAX is required without the `[cpu]` extra, so you can install whichever JAX build you need.
- Installation instructions, including the FFTW requirement for jax-finufft.

## [0.5.1] - 2025-08-13

### Changed

- Packaging for PyPI.

## [0.5.0] - 2025-08-13

Initial release.
