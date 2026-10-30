# spectracles

Spectrospatial modelling with JAX and Equinox.

## Installation

```bash
pip install spectracles
pip install "spectracles[nifty]"   # optional: MGVI / geoVI through NIFTy.re
```

NUTS needs [BlackJAX](https://github.com/blackjax-devs/blackjax)
(`pip install blackjax`); MAP uses optax, which is included.

## Status

From 1.0.0, spectracles follows semantic versioning: the public API (the
`spectracles`, `spectracles.priors`, `spectracles.inference` and
`spectracles.nifty` namespaces) only changes incompatibly in a major release,
after a deprecation warning. Two parts are outside this promise:

!!! warning "Experimental: `spectracles.lvm_models`"
    The LVM-specific models are not covered by the stability promise. They are
    being ported to priors and the latent space one at a time.

!!! note "Changing: `OptimiserFrame` and schedules"
    Models can currently be fitted two ways: with `OptimiserFrame` and
    schedules (using `prior_logpdf` losses), or through the latent space
    ([`Posterior`](inference.md) with NIFTy.re, BlackJAX or optax). These
    will be unified, as the [roadmap](roadmap.md) describes.

## Quick Start

Build a model from spectracles components inside your own Equinox module, then
fit it one of two ways.

**Bayesian inference** (priors on parameters, then MGVI/geoVI, NUTS or MAP on
one posterior). See [Priors and inference](inference.md):

```python
# sketch
from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior
from spectracles.nifty import to_nifty

model = sp.build_model(MyModel, ...)  # parameters declared with prior=...
posterior = Posterior(LatentSpace(model), forward, GaussianLikelihood(flux, ivar, mask))
samples, _ = jft.optimize_kl(to_nifty(posterior), ...)  # MGVI / geoVI
```

**Optimisation schedules** (point estimates with your own loss). See the
[Schedule Builder](schedule_builder.md):

```python
# sketch
schedule = sp.build_schedule(
    model,
    loss_fn,
    phases=[(100, 0.1), (50, 0.01)],
    params={
        "*.coefficients": sp.free_in(0, 1),
        "*.kernel.*": sp.free_after(1),
    },
)
schedule.run_all(x=data_x, y=data_y)
```
