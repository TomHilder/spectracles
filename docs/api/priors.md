# Priors

Priors on parameters, each a transform of a standard-normal latent ξ. Attach one
with `Parameter(..., prior=...)` or `ShareModule.with_priors`. See
[Priors and inference](../inference.md) for how they're used, and
[How it works](../inference_design.md#transforms-and-conventions) for the
transforms.

::: spectracles.priors.Prior

::: spectracles.priors.StdNormal

::: spectracles.priors.Normal

::: spectracles.priors.LogNormal

::: spectracles.priors.Laplace

::: spectracles.priors.Uniform
