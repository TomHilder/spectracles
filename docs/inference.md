# Priors and inference

To fit a spectracles model with Bayesian inference, you put a prior on each
parameter you want to fit, describe the noise in your data with a likelihood,
and hand the result to a fitting method. The same model works with:

- MGVI and geoVI, through [NIFTy.re](https://github.com/NIFTy-PPL/NIFTy);
- NUTS, through [BlackJAX](https://github.com/blackjax-devs/blackjax);
- a best fit (MAP), through [optax](https://github.com/google-deepmind/optax).

Switching method changes how the fit is done, not the problem being solved.

This page is the practical guide. [How it works](inference_design.md) explains
the design in more depth. To build the model in the first place, see
[Building models](models.md).

## Priors, likelihoods and posteriors in one minute

When you fit a model to an IFU cube, you are estimating maps (a line flux, a
velocity field) together with a few global numbers, such as how quickly those
maps vary.

- A **prior** says which values are plausible before you look at the data. For
  example, "the correlation length is around 0.5, give or take 0.2", or "the
  per-spaxel offsets are small".
- The **likelihood** says how probable the observed spectra are for given
  parameter values. For Gaussian noise of known variance, it is the familiar
  χ².
- The **posterior** combines the two: what is plausible *after* seeing the
  data. A best fit is its peak. Samples from it give uncertainties on every map
  and every quantity you derive from them.

## Setup

```bash
pip install "spectracles[nifty]"   # for MGVI and geoVI
pip install blackjax               # for NUTS
```

Inference should be done in double precision:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## A running example

A line-flux map, made of a positive Gaussian-process field plus a small offset
per spaxel, observed at 300 spaxels:

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

Give a `Parameter` its prior when you create it:

```python
kernel = Matern32(
    length_scale=Parameter(prior=LogNormal(0.5, 0.2)),
    variance=Parameter(prior=LogNormal(1.0, 0.5)),
)
model = build_model(
    LineMap,
    log_flux=FourierGP(kernel, n_modes=(21, 21)),
    offset=PerSpaxel(N_SPAXELS),
)
```

The priors are in `spectracles.priors`:

| Prior | Arguments | Use it for |
|---|---|---|
| `StdNormal()` | none | GP coefficients (the default for them) |
| `Normal(mean, std)` | mean and standard deviation | offsets, calibration terms |
| `LogNormal(mean, std)` | mean and standard deviation **of the value itself** | positive quantities: length scales, variances, amplitudes |
| `Laplace(alpha)` | scale (centred on 0) | corrections that are mostly near zero |
| `Uniform(low, high)` | bounds | a bounded quantity with no preferred value |

A parameter with a prior starts at the prior's median, unless you give it a
starting value: `Parameter(0.7, prior=...)`.

A prior's arguments are single numbers, and apply to every element of the
parameter (e.g. every spaxel of a `PerSpaxel`).

`FourierGP` gives its coefficients a `StdNormal` prior automatically, since the
kernel already sets their scale. If you pass in your own `coefficients`
parameter, give it `prior=StdNormal()` yourself.

You can also set priors after building the model, by path or by glob pattern
(`*` matches one part of a path, `**` any number of parts):

```python
model = model.with_priors({"offset.spaxel_values": Laplace(0.02)})
```

A value of `"fixed"` fixes the matching parameters instead, e.g.
`model.with_priors({"**.variance": "fixed"})`. `with_priors` returns a new
model and leaves the old one unchanged.

## Free, fixed and Known

Every number in a model that isn't data is one of three kinds.

A **free `Parameter`**, such as `Parameter(prior=LogNormal(0.5, 0.2))`, is
something you want to learn from the data. It needs a prior.

A **fixed `Parameter`**, such as `Parameter(0.5, fixed=True)`, is something
you've chosen not to fit, for now. It keeps its value, but it is still a
parameter: it shows up in summaries, `model.set` can change it, and it can be
freed again. `with_priors({...: "fixed"})`, `posterior.fix` and
[staged fits](#staged-fits) all fix parameters.

**`Known(value)`** is a constant, such as a rest wavelength or the instrument's
line-spread width. It can never be fitted. Summaries leave it out (pass
`show_knowns=True` to include it), and `model.set` won't change it unless you
pass `allow_set_knowns=True`.

If you might ever want to fit it, make it a `Parameter`. If it's a fact about
the world or the instrument, make it `Known`. While fitting, fixed parameters
and `Known` values behave the same way.

| Declared as | Fitted? |
|---|---|
| `Parameter(prior=P)` or `Parameter(value, prior=P)` | **yes** |
| `Parameter(value, fixed=True)` | no; it keeps its value |
| `Known(value)` | no; it's a constant |
| `Parameter(value)`, free, no prior | **error**: give it a prior, or fix it |
| `ConstrainedParameter(...)`, free | **error**: use a `Parameter` with a prior instead; the prior sets the range (e.g. `LogNormal` for positive values) |

A parameter shared between two parts of the model (say, one velocity field
used by two lines) is fitted once, with one prior. Two separate parameters that
happen to have the same prior are fitted independently.

## Checking your model

The parameter summary shows each parameter's prior:

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
│ offset.spaxel_values         │ (300,)   │ (-∞, ∞) │ Laplace(0.02)       │  free  │ -           │
└──────────────────────────────┴──────────┴─────────┴─────────────────────┴────────┴─────────────┘
4 parameters · 4 free, 0 fixed
```

If a free parameter has no prior, you find out when you set up the fit (next
section), with every such parameter listed. Before the `with_priors` call
above, you would have seen:

```text
ValueError: Cannot build the latent space; free parameters without a prior (give them one
with Parameter(prior=...) or model.with_priors, or fix them): ['offset.spaxel_values']
```

## The posterior

A `Posterior` brings together three things:

- the model's free parameters and their priors, collected by `LatentSpace`;
- a **forward model**: a function that turns the model into a prediction with
  the same shape as your data;
- a **likelihood** for the data.

Here we simulate data from a random draw of the model, with 10 spaxels set to
NaN:

```python
import numpy as np

from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior

latent = LatentSpace(model)

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
Posterior(LatentSpace(4 parameters, 743 latent dimensions), GaussianLikelihood on 290 data points)
```

For spectra, the forward model is usually the model evaluated at every
wavelength:
`lambda m: jax.vmap(m, in_axes=(0, None))(wavelengths, spatial_data)`.

`GaussianLikelihood(data, ivar, mask)` skips data points that are masked, NaN,
or have an inverse variance of zero, so bad pixels in real data are safe.

For other noise models, write your own likelihood by subclassing
`spectracles.inference.Likelihood` (see its
[API reference](api/inference.md)). `GaussianLikelihood` is a short example of
one.

## Fitting

Every method works on the same function, `posterior.log_density`. It is plain
JAX, so `jit`, `grad` and `vmap` all work on it.

### MGVI and geoVI (NIFTy.re)

`to_nifty` turns the posterior into something NIFTy can fit:

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

Real fits use more iterations and samples. NIFTy's documentation covers the
settings.

### NUTS (BlackJAX)

`posterior.log_density` can be passed straight to BlackJAX. NIFTy isn't needed:

```python
# sketch
import blackjax

warmup = blackjax.window_adaptation(blackjax.nuts, posterior.log_density)
(state, parameters), _ = warmup.run(key, latent.zeros(), num_steps=500)
nuts = blackjax.nuts(posterior.log_density, **parameters)
# ...then the usual BlackJAX sampling loop over nuts.step
```

The [worked example](examples/inference.md) runs this end to end.

### Best fit (MAP, with optax)

The best fit minimises `-posterior.log_density`:

```python
import optax

optimiser = optax.adam(0.05)
ξ = latent.zeros()  # start from the priors' medians
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

The fit works on rescaled versions of the parameters (ξ, see
[Under the hood](#under-the-hood-the-latent-space)), and `latent.to_model`
turns the result back into a model.

### Fixing parameters for a whole fit

`fix` gives a new posterior with some parameters held at chosen values. It
works with every method, and the original posterior is unchanged:

```python
fixed = posterior.fix({"log_flux.kernel.length_scale": 0.5})
print(len(posterior.latent.paths), len(fixed.latent.paths))
```

```text
4 3
```

A single number fills a parameter of any shape.

### Staged fits

With NIFTy, `stages` holds some parameters fixed for the first few
iterations only. It uses the same words as the
[schedule builder](schedule_builder.md) (`free_in`, `free_after`, `free_until`,
`fixed_in`), counting VI iterations:

```python
from spectracles import fixed_in
from spectracles.nifty import stages

staging = stages(latent, {"offset.*": fixed_in(0, 1)}, n_iterations=4)
print(staging["constants"](0), staging["constants"](2))
```

```text
('offset.spaxel_values',) ()
```

Pass it to NIFTy with `jft.optimize_kl(..., **staging)`.

## Getting results back

Every method gives you samples. The posterior turns them into what you want,
whichever method made them.

### A summary table

```python
summary = posterior.summary(samples)  # prints a table; also returns {path: {"mean", "std"}}
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

For simulated data, pass the truth as `true_params=` (parameter values) or
`true_latents=` (as drawn with `sample_prior_latent`). The table then shows
the true values, and how many standard deviations each fit is from them.

### Parameter values

`posterior.draws(samples)` gives the parameter values at every sample. Index it
by path, and summarise over the samples:

```python
params = posterior.draws(samples)
params.mean()  # {path: mean}
params["log_flux.kernel.length_scale"].std()
```

`.values` holds the samples themselves, e.g. for a corner plot.

### Maps

With `to="model"`, you get the model at every sample. Call it like the model,
and summarise the result:

```python
models = posterior.draws(samples, to="model")
maps = models(spaxels)  # the whole model, at every sample
mean_map, std_map = maps.mean(), maps.std()
lower, upper = maps.quantile(0.16), maps.quantile(0.84)
```

Any part of the model works the same way, even parts that use shared
parameters:

```python
flux = models.log_flux(spaxels).mean()
```

`maps.reduce(fn)` applies any other statistic, e.g. `maps.reduce(jnp.var)`.

### Anything else you compute from the model

For a quantity that isn't a single call, such as a ratio of two fields, give
`map_samples` a function of the model:

```python
ratio = posterior.map_samples(lambda m: m.log_flux(spaxels) - m.offset(spaxels), samples)
predictions = posterior.map_samples(posterior.forward, samples).mean()
```

### Several chains

NUTS chains stacked as `(chain, draw, ...)` are summarised over both axes.
Give `axis=` to summarise over some of them, e.g. `maps.mean(axis=1)` for the
mean of each chain.

Everything is held in memory. For very large fields, evaluate on fewer points
or fewer samples at a time.

!!! note "Which posterior mean?"
    For a nonlinear model, these give different answers:

    - the mean of each parameter: `posterior.draws(samples).mean()`;
    - the model at the mean parameters:
      `posterior.to_model(posterior.draws(samples).mean(), space="params")`;
    - the mean of a derived quantity, such as a map: `maps.mean()`.

    For maps and their uncertainties, use the last. Averaging the GP
    coefficients first gives a map smoother than any sample.

### With ArviZ

Parameter values keep the layout of the samples, `(draw, ...)` or
`(chain, draw, ...)`. That is what [ArviZ](https://www.arviz.org) expects:

```python
import arviz as az


def for_arviz(values, chains=False):
    """{path: array} -> ArviZ variables, adding a chain axis if there isn't one.
    GP coefficients are left out: they are large and rarely wanted."""
    return {p: v if chains else v[None] for p, v in values.items() if "coefficients" not in p}


idata = az.from_dict(
    {
        "posterior": for_arviz(posterior.draws(samples).values),
        "prior": for_arviz(latent.sample_prior_params(random.PRNGKey(3), n=500)),
    }
)
print(az.summary(idata, var_names=["log_flux.kernel.length_scale"], kind="stats"))
```

```text
                                mean    sd eti89_lb eti89_ub
log_flux.kernel.length_scale[0]  0.8  0.19     0.63     0.98
```

For chains run in parallel, pass `chains=True`.

!!! warning "Convergence diagnostics are for MCMC only"
    MGVI and geoVI samples are not Markov chains, so R-hat and effective sample
    sizes mean nothing for them. Use ArviZ's summaries and plots for VI
    samples, and its diagnostics only for NUTS.

## Checking your priors

Before fitting, it's worth checking that your priors give sensible maps. Draws
from the prior come as a model, as parameter values, or as latents:

```python
draw = latent.sample_prior_model(random.PRNGKey(1))
prior_map = draw(spaxels)  # a line-flux map drawn from the priors

ls = latent.sample_prior_params(random.PRNGKey(2), n=1000)["log_flux.kernel.length_scale"]
print(ls.shape, float(ls.mean()))  # 1000 prior draws of the length scale
```

Prior draws of parameter values can be summarised like posterior samples. Say
that they are already parameter values with `space="params"`:

```python
prior_values = latent.sample_prior_params(random.PRNGKey(4), n=500)
prior_summary = posterior.summary(prior_values, space="params", print_table=False)
```

## Under the hood: the latent space

You rarely need this section, but it explains what the fitting methods
actually see.

Each element of each free parameter is written as a standard-normal number, ξ,
passed through its prior. For example, `LogNormal(0.5, 0.2)` turns ξ = 0 into
its median and ξ = ±1 into roughly its ±1σ values. Every fitting method works
with ξ, where all parameters are on the same scale. `LatentSpace` holds this
mapping:

```python
print(latent)
print(latent.paths)
```

```text
LatentSpace(4 parameters, 743 latent dimensions)
['log_flux.coefficients', 'log_flux.kernel.length_scale', 'log_flux.kernel.variance', 'offset.spaxel_values']
```

ξ is a dict keyed by parameter path. The latent space converts it:

- `latent.to_params(ξ)` gives the parameter values;
- `latent.to_model(ξ)` gives a model with those values;
- `latent.from_model(model)` goes back from a model to ξ, e.g. to start a fit
  from an earlier one.

Functions that take samples assume they are ξ, as fitting methods return them.
For parameter values, pass `space="params"`.

## Two ways to fit

spectracles also has an older way to get a best fit:
[`OptimiserFrame`](api/opt_schedule.md) and [schedules](schedule_builder.md),
with a loss you write yourself. It ignores priors. Use it for quick best fits
with a custom loss, and the posterior for uncertainties. The two will be
merged (see the [roadmap](roadmap.md)).

## Current limitations

- **Priors are independent.** A prior can't depend on another parameter,
  except through the field hierarchy that `FourierGP` builds in.
- **Prior arguments are single numbers.** There's a workaround for giving each
  element its own prior in
  [How it works](inference_design.md#limitations-and-extension-points).
- **`from_model` is exact only for some priors** (`StdNormal`, `Normal`,
  `LogNormal`). Others fall back to the prior median, with a warning.
- **Parameters in lists, tuples or dicts** aren't supported yet (planned for
  1.1), and `MultiFourierGP` is disabled until then. See
  [Building models](models.md#current-limitations).
