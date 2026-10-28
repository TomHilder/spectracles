# Graphical models

`spectracles.pgm` draws a model, or a posterior with its data, as a
probabilistic graphical model with [daft](https://docs.daft-pgm.org/). It needs
the optional dependency: `pip install "spectracles[pgm]"`. Importing the module
doesn't import daft; only `plot_pgm` does. The same plot is available as
`ShareModule.plot_pgm` and `Posterior.plot_pgm`.

Two views:

- **`"parameter"`** (default): the model as usually written. Free parameters
  are random variables, with their priors as small fixed nodes; a `FourierGP`
  is a random field whose parents are its kernel's hyperparameters; every
  other module is a deterministic function of what it holds.
- **`"latent"`**: the model as the inference backends see it. The random
  variables are the standard-normal latents ξ, each parameter is a
  deterministic transform of its latent, and a `FourierGP` is a deterministic
  function of its whitened coefficients and its kernel's hyperparameters.

Edges follow the module tree: a parameter feeds the module that holds it, and
each module feeds the module that holds it. A shared parameter or component is
one node, with an edge to everything that uses it.

::: spectracles.pgm.plot_pgm

::: spectracles.pgm.pgm_graph
