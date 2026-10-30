# Roadmap

Planned work, roughly in order. Nothing here is covered by the
[stability promise](index.md#status) until it is released.

## One way to fit

Models can currently be fitted two ways: with `OptimiserFrame` and schedules
(using `prior_logpdf` losses), or through the latent space
([`Posterior`](inference.md) with NIFTy.re, BlackJAX or optax). These will be
unified: optimisation will move to the latent space, schedules and VI stages
will share one vocabulary, and the `prior_logpdf` losses will be deprecated.

## Parameters in lists, tuples and dicts (1.1)

Parameters, and modules holding them, must currently be named attributes of a
module; `build_model` raises a clear error otherwise. 1.1 will support lists,
tuples and dicts, with paths such as `lines[0].flux` or `lines['Ha'].flux`, so
models with a variable number of components (e.g. emission lines) can be built
in a loop. This also makes `MultiFourierGP`, which holds its kernels in a
list, usable in a built model. Existing paths are unchanged.
([#4](https://github.com/TomHilder/spectracles/issues/4))

## Lazy draws for large fields (1.2)

`posterior.draws(samples, to="model")` evaluates and holds every draw in
memory. For many samples of a field on many points, summaries will be computed
in chunks of samples (streaming mean and variance; quantiles in chunks of
points), with the full draws built only on request.

## Corner plots from draws

`Draws.corner(...)`, a small wrapper around
[`corner`](https://corner.readthedocs.io) (an optional extra): hyperparameters
only by default (`exclude_fourier_coeffs=True`), labels from the parameter
paths, vector parameters split into columns, ringed scatter points with step
histograms, and every `corner` keyword (including `fig`, to overlay several
methods) passed through.
