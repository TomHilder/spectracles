# How it works

The [practical guide](inference.md) shows how to use priors, the latent space
and the backends. This page explains the design behind them, and the points
that matter when you rely on the results.

## One problem, in latent coordinates

Every free parameter θᵢ is written as a deterministic transform of
standard-normal latents:

$$
\theta_i = T_i(\xi_i), \qquad \xi_i \sim \mathcal{N}(0, 1)\ \text{(elementwise)},
$$

where Tᵢ is the transform of the parameter's prior. The posterior over the
latents is then

$$
\log p(\xi \mid d) = \log L\big(d \mid F(\theta(\xi))\big) - \tfrac12 \lVert \xi \rVert^2 + \text{const},
$$

with F the forward model and L the likelihood. This one function is
`Posterior.log_density`. Every backend works on it, in these coordinates.

**Why standard-normal coordinates:**

- **They are what MGVI and geoVI assume.** NIFTy's methods are built around a
  standard-normal prior on the latents. Their metric is the likelihood's
  Fisher information plus the identity (see
  [below](#what-each-backend-sees)).
- **They are good coordinates for NUTS and optimisers.** Parameters with
  wildly different scales become order-one numbers. The GP hierarchy is
  non-centred (next section), which removes the "funnel" between a field and
  its hyperparameters that makes centred parametrisations hard to sample.
- **The prior needs no density.** In ξ the prior is always −½|ξ|², so priors
  are only ever transforms. That's why each prior class implements a transform
  and (optionally) its inverse, and no log-density.
- **Every method sees the same problem.** Results from MGVI, NUTS and MAP differ
  because the methods differ, not because they were handed different
  parametrisations of the model.

## Why no probabilistic programming language

The only hierarchy in these models is GP hyperparameters → field, and
`FourierGP` already handles it internally. The field is

$$
f(x) = \sum_k \sqrt{P(k;\ell, s^2)}\; c_k\, e^{i k \cdot x},
$$

where the coefficients c are **whitened**: the power spectrum P scales them.
The coefficients' prior is therefore N(0, 1) whatever the hyperparameters are,
and each hyperparameter can have its own independent prior. This is the
non-centred parametrisation, built into the model. Priors can stay as
independent data on each parameter, with no graph of dependencies, which is why
spectracles has `Parameter(prior=...)` rather than `sample()` statements.

## Transforms and conventions

| Prior | T(ξ) | Inverse |
|---|---|---|
| `StdNormal()` | ξ | ✓ |
| `Normal(m, s)` | m + s ξ | ✓ |
| `LogNormal(mean, std)` | exp(μ + σ ξ), with σ² = log(1 + std²/mean²), μ = log(mean) − σ²/2 | ✓ |
| `Laplace(α)` | the Laplace quantile function of Φ(ξ), computed from log Φ for accuracy in the tails | — |
| `Uniform(a, b)` | a + (b − a) Φ(ξ) | — |

- `LogNormal` is parametrised by the mean and standard deviation of the value
  itself, not of its logarithm. It is moment-matched, as in NIFTy.re.
- spectracles implements these transforms itself, so the latent space and the
  log-density work without NIFTy installed. The tests check them against
  NIFTy's own priors to rtol 1e-14 when NIFTy is available.
- **Inverses** are only used by `LatentSpace.from_model`, which maps existing
  values back to ξ (for example, to start a fit from an earlier one).
  `Uniform`'s inverse is badly conditioned at the bounds (values there map to
  ξ = ±∞), and neither `Uniform` nor `Laplace` has one yet. `from_model` falls
  back to ξ = 0 for them, and for values outside a prior's support, with a
  warning naming the parameters.

## Shared parameters are one parameter

Sharing in spectracles is by object identity: one `Parameter` object used in
several places. For inference that means:

- **one latent**, keyed by the parent path;
- **one prior and one value.** The value is written once and read everywhere;
- **gradients add up** from every place the parameter is used.

Two separate `Parameter`s that happen to have equal priors (even the same
`Prior` object) are two independent parameters. This is why the prior is a
*static* field on `Parameter`: if it were an array leaf, a `Prior` reused
between parameters would look like shared arrays to `ShareModule`, and an
optimiser would try to train its hyperparameters.

If parameters share only their value array (e.g. via `eqx.tree_at`) but carry
different priors, `validate_sharing` reports it and `LatentSpace` refuses the
model. One parameter can't have two priors.

## What each backend sees

Write J for the Jacobian of the forward model with respect to ξ, and N for the
noise covariance (for `GaussianLikelihood`, diagonal with entries 1/ivar).

- **MAP (optax)** maximises `log_density`.
- **NUTS (BlackJAX)** samples `exp(log_density)`.
- **MGVI and geoVI (NIFTy).**
  - `to_nifty` gives NIFTy the forward model and the likelihood. NIFTy works
    with an *energy* E(ξ) = −log L(ξ) + ½|ξ|², without normalisation
    constants.
  - MGVI approximates the posterior around the current mean ξ̄ by a Gaussian
    whose precision is the metric M(ξ̄) = Jᵀ N⁻¹ J + 1.
  - geoVI additionally uses the metric's square root, Jᵀ N^{-1/2}, to curve
    the approximation.

The test suite checks that these are all the same object, without running any
inference:

- NIFTy's forward model and energy gradient match spectracles';
- NIFTy's metric equals Jᵀ N⁻¹ J computed independently, and the square root
  composes to it;
- NIFTy's own BlackJAX target and a BlackJAX NUTS state match `log_density`;
- for a linear-Gaussian model, the Hessian of `log_density` equals the metric
  plus identity exactly, so MGVI's Gaussian and NUTS's target are literally the
  same distribution there.

All of this agrees to rtol 1e-14 on CPU.

## Masking without NaN gradients

Real data have masked pixels whose values are often NaN. A likelihood written as
`jnp.where(mask, logpdf(data, pred), 0)` gives the right *value*, but its
*gradient* is NaN: `where` multiplies the unused branch's derivative by zero, and
0 × NaN = NaN. `GaussianLikelihood` removes masked pixels when it is built
(keeping the indices of the used pixels), so masked values never enter a
computation. It also drops pixels with non-finite data or inverse variance ≤ 0.

## Staging

NIFTy has two separate per-iteration settings, and `stages` sets both:

- `constants`: these latents are not updated by the KL minimisation;
- `point_estimates`: no samples are drawn along these latents.

With `constants` alone, a "frozen" parameter keeps its mean but samples are
still drawn along it, so it isn't really frozen. On a test problem, the sample
spread of frozen hyperparameters was 0.03–0.8 in ξ with `constants` alone, and
exactly 0 with both settings. A parameter that `stages` reports as not free is
therefore in both lists, and fully frozen.

## Hyperparameters are the hard part

GP hyperparameters (length scale, variance) are weakly constrained by the data
and strongly correlated with the field, so they are usually the hardest part of
the posterior to explore:

- MGVI and geoVI approximate the posterior locally (by a Gaussian, or a curved
  one for geoVI), built from a finite number of samples. Treat their
  hyperparameter estimates and uncertainties as approximate.
- NUTS can mix slowly along these directions. Run long chains from several
  starting points and check convergence, e.g. with R-hat.

## MAP in latent space vs the old losses

The older losses in `spectracles.lvm_models` add `FourierGP.prior_logpdf` to the
log-likelihood. `prior_logpdf` is the density of the *scaled* coefficients
√P · c, which includes the Jacobian term −Σ log √P(k; ℓ, s²). That term depends
on the hyperparameters. A MAP in ξ has no such term: its prior is just −½|ξ|².

MAP is not invariant to reparametrisation, so the two optima are different
points whenever hyperparameters are free. Neither is wrong; they are different
problems. Results from the old path and from the posterior should not be
expected to agree exactly.

## Reproducibility of VI runs

For fixed inputs and keys, NIFTy's MGVI and geoVI are bit-for-bit reproducible.
They are, however, sensitive to the last bit of the input.

On a 6-iteration test problem, changing the data by one unit in the last place
moved the posterior mean by:
- MGVI: about 0.06σ rms, up to about 0.8σ in places;
- geoVI: about 0.07σ rms, up to about 1σ in places.

Here σ is the posterior standard deviation. The minimisers hit their iteration
limits, so tiny differences take different paths. Two mathematically identical
implementations of a model can therefore give visibly different VI results.
Compare runs statistically (several seeds, or against NUTS), not bit for bit.

## Limitations and extension points

- **Scalar hyperparameters only.** Every element of a parameter gets the same
  prior. To give elements different priors, put a `StdNormal` on the parameter
  and build the value in the model from `Known` arrays:

  ```python
  # sketch
  class PerTileOffset(eqx.Module):
      raw: Parameter  # prior=StdNormal()
      mean: Known  # per-tile prior means
      std: Known  # per-tile prior widths

      def __call__(self, data):
          return (self.mean.val + self.std.val * self.raw.val)[data.tile_idx]
  ```

- **No dependent priors** beyond the GP hierarchy. A prior can't depend on
  another parameter.
- **Likelihoods:** only `GaussianLikelihood` so far. A Student-t likelihood and
  a learned noise scaling are planned. They will map to NIFTy's `StudentT` and
  `VariableCovarianceGaussian`.
- **Two ways to fit.** The optimiser path (`OptimiserFrame`, schedules) ignores
  priors. Unifying it with the latent space, so that optimisation also happens
  in ξ, is planned.
