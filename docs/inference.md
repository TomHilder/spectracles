# Priors and inference

spectracles models can be fitted with Bayesian inference: put a prior on every
parameter you want to infer, describe your data with a likelihood, and hand the
resulting posterior to an inference backend.

- **MGVI and geoVI** run through [NIFTy.re](https://ift.pages.mpcdf.de/nifty/).
- **NUTS** runs through [BlackJAX](https://blackjax-devs.github.io/blackjax/).
- **MAP** runs through [optax](https://optax.readthedocs.io/).

All three work on the same function, in the same coordinates. Switching backend
changes how the posterior is explored, not what the posterior is.

This page is the practical guide. [How it works](inference_design.md) explains
the design and the subtler points in more depth.

## Priors, likelihoods and posteriors in one minute

When you fit a model to an IFU cube you are estimating maps, e.g. a line flux or
a velocity field, together with a few global quantities such as a correlation
length.

- **Prior.** Before looking at the data, what values are plausible? For example,
  "the correlation length is around 0.5 in model units, give or take 0.2", or
  "per-spaxel offsets are small and mostly zero". A prior is that statement,
  written as a probability distribution.
- **Likelihood.** Given some parameter values, how probable are the observed
  spectra? For data with Gaussian noise of known variance, this is the familiar
  χ², written as a probability.
- **Posterior.** Prior and likelihood together, which tells you what is
  plausible *after* seeing the data. Inference means characterising the
  posterior. A single best fit is its peak (MAP). Samples from it give
  uncertainties on every map and every derived quantity.

## Setup

NIFTy.re is an optional dependency:

```bash
pip install "spectracles[nifty]"   # for MGVI / geoVI
pip install blackjax               # only if you want NUTS
```

Inference here is done in double precision:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## A running example

A line-flux map: a log-normal Gaussian-process field, plus a small offset per
spaxel. It's observed at 300 spaxels.

```python
import equinox as eqx
import jax.numpy as jnp
from jax import random

from spectracles import (
    FourierGP,
    Matern32,
    Parameter,
    PerSpaxel,
    SpatialDataGeneric,
    build_model,
)
from spectracles.priors import Laplace, LogNormal


class LineMap(eqx.Module):
    """A line-flux map: a log-normal GP field plus a small per-spaxel offset."""

    log_flux: FourierGP
    offset: PerSpaxel

    def __call__(self, data):
        return jnp.exp(self.log_flux(data)) + self.offset(data)


N_SPAXELS = 300
xy = random.uniform(random.PRNGKey(0), (2, N_SPAXELS), minval=-3.0, maxval=3.0)
spaxels = SpatialDataGeneric(x=xy[0], y=xy[1], idx=jnp.arange(N_SPAXELS))
```

## Declaring priors

A prior is attached to the `Parameter` it describes:

```python
kernel = Matern32(
    length_scale=Parameter(prior=LogNormal(0.5, 0.2)),
    variance=Parameter(prior=LogNormal(1.0, 0.5)),
)
model = build_model(
    LineMap,
    log_flux=FourierGP(n_modes=(21, 21), kernel=kernel),
    offset=PerSpaxel(N_SPAXELS),
)
```

The available priors are in `spectracles.priors`:

| Prior | Parameters | Use it for |
|---|---|---|
| `StdNormal()` | none | GP coefficients (the default for them) |
| `Normal(mean, std)` | mean and standard deviation | offsets, calibration terms |
| `LogNormal(mean, std)` | mean and standard deviation **of the value itself** | positive quantities: length scales, variances, amplitudes |
| `Laplace(alpha)` | scale; location 0 | sparse corrections that are mostly near zero |
| `Uniform(low, high)` | bounds | a bounded quantity with no preferred value |

A parameter with a prior and no value starts at the prior's median (its
latent value ξ = 0), which is also where `latent.zeros()` starts. Give a value,
`Parameter(0.7, prior=...)`, only to start somewhere else; it is the parameter's
value outside inference and the point `latent.from_model` maps back to ξ.

Hyperparameters are plain numbers, applied to every element of the parameter.
The note in [How it works](inference_design.md#limitations-and-extension-points)
explains how to give each element its own prior.

A `FourierGP` gives its coefficients a `StdNormal` prior automatically, because
the kernel already sets their scale. If you pass your own `coefficients`
parameter (to set initial values, or to share one set between two fields), give
it `prior=StdNormal()` yourself.

You can also set priors afterwards, by path or by glob pattern. The syntax is
the same as for [schedules](schedule_builder.md#pattern-matching):

```python
model = model.with_priors({"offset.spaxel_values": Laplace(0.02)})
```

`with_priors` also takes `"fixed"` to fix a parameter, and returns a new model.

## What gets inferred

| Parameter | Inferred? |
|---|---|
| `Known(value)` | no; it's a constant |
| `Parameter(value, fixed=True)` | no; it keeps its value |
| `Parameter(value, prior=P)`, free | **yes** |
| `Parameter(value)`, free, no prior | **error**: say what you believe about it, or fix it |
| `ConstrainedParameter(...)`, free | **error**: use a `Parameter` with a prior; the prior sets the bounds (e.g. `LogNormal` for positive values) |

!!! note "A shared parameter is one parameter"
    If two parts of a model hold the same `Parameter` (for example two emission
    lines sharing one velocity kernel), it is inferred once, with one prior. Two
    separate parameters that merely have equal priors are inferred
    independently.

## Checking your model

Priors show up in the usual inspection tools:

```python
print(kernel.length_scale)
model.get_parameter_summary()
```

```text
Parameter(0.4642, free, ~LogNormal(0.5, 0.2))
                                        Parameter Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Path                         ┃ Shape    ┃ Bounds  ┃ Prior               ┃ Status ┃ Shared from ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│ log_flux.coefficients        │ (21, 21) │ (-∞, ∞) │ StdNormal           │  free  │ -           │
│ log_flux.kernel.length_scale │ (1,)     │ (-∞, ∞) │ LogNormal(0.5, 0.2) │  free  │ -           │
│ log_flux.kernel.variance     │ (1,)     │ (-∞, ∞) │ LogNormal(1, 0.5)   │  free  │ -           │
│ log_flux.zeromode            │ (1,)     │ (-∞, ∞) │ -                   │ fixed  │ -           │
│ offset.spaxel_values         │ (300,)   │ (-∞, ∞) │ Laplace(0.02)       │  free  │ -           │
└──────────────────────────────┴──────────┴─────────┴─────────────────────┴────────┴─────────────┘
5 parameters · 4 free, 1 fixed
```

`print_model_tree()` and the model's repr also show priors. Missing or
conflicting priors are caught when you build the latent space (next section),
with every offending path listed. Before the `with_priors` call above, you would
have seen:

```text
ValueError: Cannot build the latent space; free parameters without a prior (give them one
with Parameter(prior=...) or model.with_priors, or fix them): ['offset.spaxel_values']
```

## The latent space

Inference works on one standard-normal number, ξ, per element of every free
parameter. The prior turns ξ into the parameter's value. For example,
`LogNormal(0.5, 0.2)` maps ξ = 0 to its median and ξ = ±1 to roughly the ±1σ
values. The `LatentSpace` holds this mapping:

```python
from spectracles.inference import LatentSpace

latent = LatentSpace(model)
print(latent)
print(latent.paths)
```

```text
LatentSpace(4 parameters, 743 latent dimensions)
['log_flux.coefficients', 'log_flux.kernel.length_scale', 'log_flux.kernel.variance', 'offset.spaxel_values']
```

ξ is a dict keyed by parameter path. The latent space converts between three
forms of the same draw: the latents ξ, the parameter values, and a model.

- `latent.to_params(ξ)` gives the free parameters' values, a dict keyed by the
  same paths;
- `latent.to_model(ξ)` gives an ordinary model with those values (the same as
  `model.set(latent.to_params(ξ))`);
- `latent.from_model(model)` goes the other way, e.g. to start inference from
  an earlier fit.

Draws from the prior come in the same three forms:
`latent.sample_prior_latent(key)`, `latent.sample_prior_params(key)` and
`latent.sample_prior_model(key)`. With `n=...`, the first two return n draws
stacked along a leading axis (the layout posterior samples have too), and
`sample_prior_model` returns a list of n models. These are useful for checking
your priors, and for prior-predictive checks: do maps drawn from your priors
look plausible? A `Posterior` has the same methods.

```python
draw = latent.sample_prior_model(random.PRNGKey(1))
prior_map = draw(spaxels)  # a line-flux map drawn from the priors

ls = latent.sample_prior_params(random.PRNGKey(2), n=1000)["log_flux.kernel.length_scale"]
print(ls.shape, float(ls.mean()))  # 1000 prior draws of the length scale
```

## The posterior

A `Posterior` combines the latent space with a **forward model** (how a model
turns into a prediction with the data's shape) and a **likelihood**. Here we
simulate data from a known truth, with 10 masked (NaN) spaxels:

```python
import numpy as np

from spectracles.inference import GaussianLikelihood, Posterior

truth = latent.sample_prior_latent(random.PRNGKey(2))
noise = 0.05
rng = np.random.default_rng(0)
flux = np.asarray(latent.to_model(truth)(spaxels)) + noise * rng.standard_normal(N_SPAXELS)
ivar = np.full(N_SPAXELS, noise**-2)
flux[:10] = np.nan  # e.g. bad fibres


def forward(m):
    return m(spaxels)


posterior = Posterior(latent, forward, GaussianLikelihood(flux, ivar))
print(posterior)
```

```text
Posterior(LatentSpace(4 parameters, 743 latent dimensions), GaussianLikelihood on 290 pixels)
```

For spectra, the forward model is usually the model evaluated over wavelength,
`lambda m: jax.vmap(m, in_axes=(0, None))(wavelengths, spatial_data)`.

`GaussianLikelihood(data, ivar, mask)` drops masked pixels, and pixels with
non-finite data or non-positive inverse variance, so NaNs in real data are
safe.

`posterior.log_density(ξ)` is the function every backend works with. It is
pure JAX, so it works under `jit`, `grad` and `vmap`.

## Fitting

### MGVI and geoVI (NIFTy.re)

`to_nifty` turns the posterior into a NIFTy likelihood on ξ. NIFTy's own
standard-normal prior on ξ is then exactly the prior you declared.

```python
import nifty.re as jft

from spectracles.nifty import to_nifty

lh = to_nifty(posterior)
key_init, key_vi = random.split(random.PRNGKey(3))
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(key_init)),
    key=key_vi,
    n_total_iterations=4,
    n_samples=2,  # pairs of mirrored samples
    sample_mode="linear_resample",  # MGVI; "nonlinear_resample" for geoVI
    kl_kwargs={"minimize_kwargs": {"maxiter": 20}},
)
```

Real fits use more iterations and samples. See NIFTy's documentation for the
optimiser settings.

### Staged fits

To hold some parameters fixed for some VI iterations, use `stages`. It writes
this with the same vocabulary as the
[schedule builder](schedule_builder.md) (`free_in`, `free_after`, `free_until`,
`fixed_in`), counting VI iterations instead of optimisation phases:

```python
from spectracles import fixed_in
from spectracles.nifty import stages

staging = stages(latent, {"offset.*": fixed_in(0, 1)}, n_iterations=4)
print(staging["constants"](0), staging["constants"](2))
```

```text
('offset.spaxel_values',) ()
```

Pass it straight to NIFTy as `jft.optimize_kl(..., **staging)`. A parameter
that isn't free in an iteration is fully frozen there: it isn't moved, and no
samples are drawn along it.
[How it works](inference_design.md#staging) explains why that needs two of
NIFTy's settings.

### NUTS (BlackJAX)

The posterior's `log_density` is a valid BlackJAX target. NIFTy isn't needed.

```python
# sketch
import blackjax

warmup = blackjax.window_adaptation(blackjax.nuts, posterior.log_density)
(state, parameters), _ = warmup.run(key, latent.zeros(), num_steps=500)
nuts = blackjax.nuts(posterior.log_density, **parameters)
# ...then the usual BlackJAX sampling loop over nuts.step
```

The [worked example](examples/inference.md) runs this end to end.

### MAP (optax)

The maximum a posteriori fit minimises `-log_density` over ξ:

```python
import optax

optimiser = optax.adam(0.05)
ξ = latent.zeros()
opt_state = optimiser.init(ξ)


@jax.jit
def step(ξ, opt_state):
    loss, grad = jax.value_and_grad(lambda x: -posterior.log_density(x))(ξ)
    updates, opt_state = optimiser.update(grad, opt_state)
    return optax.apply_updates(ξ, updates), opt_state, loss


for _ in range(300):
    ξ, opt_state, loss = step(ξ, opt_state)
map_model = latent.to_model(ξ)
```

## Getting results back

Samples come back as latents. Turn them into parameter values with the same
latent space:

```python
summary = latent.posterior_summary(samples)  # also returns {path: {"mean", "std"}}
fitted = latent.to_model(samples.pos)  # the latent mean, as an ordinary model
maps = jnp.stack([latent.to_model(s)(spaxels) for s in samples])  # one map per sample
values = latent.to_params(samples)  # {path: stacked values}, e.g. for corner plots
```

```text
                                         Posterior Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Path                         ┃ Shape    ┃ Prior               ┃ Posterior mean ± std             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ log_flux.coefficients        │ (21, 21) │ StdNormal           │ mean 0.003982, median std 0.58   │
│ log_flux.kernel.length_scale │ (1,)     │ LogNormal(0.5, 0.2) │ 0.8031 ± 0.168                   │
│ log_flux.kernel.variance     │ (1,)     │ LogNormal(1, 0.5)   │ 0.2602 ± 0.111                   │
│ offset.spaxel_values         │ (300,)   │ Laplace(0.02)       │ mean 0.0002367, median std       │
│                              │          │                     │ 0.0182                           │
└──────────────────────────────┴──────────┴─────────────────────┴──────────────────────────────────┘
4 samples · 4 parameters
```

Uncertainties on any derived quantity come from evaluating it once per sample,
as with `maps` above: take the mean and spread across samples.

## Two ways to fit

spectracles also has an older optimisation path:
[`OptimiserFrame`](api/opt_schedule.md) and
[schedules](schedule_builder.md), with a loss you write yourself, often using
`FourierGP.prior_logpdf`. It is unchanged, and models with priors still work
with it; it simply ignores the priors.

- **Use the posterior** (this page) when you want uncertainties, or want to
  compare inference methods.
- **Use the optimiser path** for quick point estimates with a custom loss.

The two are not the same problem even at their optimum: a MAP fit in ξ and a fit
of the old losses generally land in different places when hyperparameters are
free (see [How it works](inference_design.md#map-in-latent-space-vs-the-old-losses)).
Unifying the two is planned.

## What this is not

- **Not a general probabilistic programming language.** Each free parameter
  has its own independent prior. Priors can't depend on other parameters,
  beyond the GP hierarchy that `FourierGP` builds in.
- **No per-element hyperparameters** (yet). There's a workaround in
  [How it works](inference_design.md#limitations-and-extension-points).
- **`from_model` can only invert some priors.** For `StdNormal`, `Normal` and
  `LogNormal` it is exact. Others (`Laplace`, `Uniform`) fall back to the prior
  median, with a warning.
- **NIFTy's correlated-field models aren't used.** spectracles' own `FourierGP`
  is the field model.
