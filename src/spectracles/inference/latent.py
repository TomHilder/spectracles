"""latent.py - The latent space of a model: standard-normal latents for its free parameters.

Every free parameter with a prior gets one entry in a dict ξ of standard-normal
latents, keyed by its (parent) parameter path. Its value is `prior.transform(ξ[path])`.
Shared parameters are one parameter, so they have one entry, at the parent path.
"""

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, PRNGKeyArray

from spectracles.inference.draws import Draws, ModelDraws
from spectracles.model.parameter import Known, Parameter
from spectracles.model.share_module import ShareModule
from spectracles.priors import Prior
from spectracles.tree.path_utils import str_to_leafpath, use_path_get_leaf

Latent = dict[str, Array]


def as_latent_dict(ξ: Any) -> Latent:
    """Accept a plain dict or a dict-like wrapper with a `.tree` (e.g. a NIFTy `Vector`)."""
    return ξ.tree if hasattr(ξ, "tree") else ξ


def as_latent_samples(samples: Any) -> Latent:
    """Like `as_latent_dict`, also accepting a NIFTy `Samples` object (stacked samples)."""
    if hasattr(samples, "samples") and not isinstance(samples, dict):
        samples = samples.samples
    return as_latent_dict(samples)


class LatentSpace:
    """The map from standard-normal latents ξ to a model's parameter values.

    Args:
        model: An unlocked ShareModule. Every free parameter must have a prior,
            and must be a plain `Parameter` (not a `ConstrainedParameter`).

    Attributes:
        model: The model the latent space was built from. Fixed parameters and
            Known values keep the values they have here.
        paths: The latent keys, i.e. the unique free parameter paths, in tree order.
        priors: Dict from path to Prior.
        shapes: Dict from path to `jax.ShapeDtypeStruct` of the latent.
    """

    def __init__(self, model: ShareModule):
        if not isinstance(model, ShareModule):
            raise TypeError(
                f"LatentSpace requires a ShareModule (use build_model), got {type(model).__name__}."
            )
        if model._locked:
            raise ValueError("LatentSpace requires an unlocked model.")
        # A shared parameter must have one prior (and one fixed status) everywhere
        model.validate_sharing(raise_on_error=True)
        self.model = model

        paths, priors, shapes = [], {}, {}
        no_prior, constrained = [], []
        for path in model.get_parameter_paths(show_shared=False, show_knowns=False):
            param = use_path_get_leaf(model.model, str_to_leafpath(path))
            if isinstance(param, Known) or param.fix:
                continue
            if type(param) is not Parameter:
                constrained.append(path)
                continue
            if param.prior is None:
                no_prior.append(path)
                continue
            paths.append(path)
            priors[path] = param.prior
            shapes[path] = jax.ShapeDtypeStruct(param.val.shape, param.val.dtype)

        errors = []
        if no_prior:
            hint = ""
            if any(p.endswith("coefficients") for p in no_prior):
                hint = (
                    " (FourierGP coefficients are whitened, so their prior is StdNormal: "
                    "pass Parameter(..., prior=StdNormal()) when supplying your own)"
                )
            errors.append(
                "free parameters without a prior (give them one with Parameter(prior=...) "
                f"or model.with_priors, or fix them): {no_prior}{hint}"
            )
        if constrained:
            errors.append(
                "free parameters that are not plain Parameters (e.g. ConstrainedParameter; "
                f"use a Parameter with a prior instead, or fix them): {constrained}"
            )
        if errors:
            raise ValueError("Cannot build the latent space; " + "; ".join(errors))

        self.paths: list[str] = paths
        self.priors: dict[str, Prior] = priors
        self.shapes: dict[str, jax.ShapeDtypeStruct] = shapes

    def __repr__(self) -> str:
        return f"LatentSpace({len(self.paths)} parameters, {self.size} latent dimensions)"

    @property
    def size(self) -> int:
        """Total number of latent dimensions."""
        return sum(math.prod(s.shape) for s in self.shapes.values())

    def to_params(self, ξ: Any) -> dict[str, Array]:
        """The free parameters' values at ξ: `prior.transform(ξ[path])` for each path.

        Accepts one draw of the latents, draws stacked along a leading axis
        (the values are then stacked the same way), or a NIFTy `Vector` or
        `Samples` object. Keys are the latent paths; fixed parameters and Known
        values are not included. Works under jit/grad/vmap.
        """
        ξ = self._checked(as_latent_samples(ξ))
        return {p: self.priors[p].transform(ξ[p]) for p in self.paths}

    def _checked(self, ξ: Any, what: str = "latents") -> Latent:
        """ξ after checking it is a dict with exactly this latent space's keys."""
        expected = "latents ξ" if what == "latents" else what
        if isinstance(ξ, ShareModule | eqx.Module):
            hint = (
                " latent.from_model(model) gives a model's latents." if what == "latents" else ""
            )
            raise TypeError(
                f"Expected {expected} (a dict keyed by parameter path), got a model.{hint}"
            )
        if not isinstance(ξ, dict):
            raise TypeError(
                f"Expected {expected} (a dict keyed by parameter path), got {type(ξ).__name__}."
            )
        missing = [p for p in self.paths if p not in ξ]
        unknown = [p for p in ξ if p not in self.priors]
        if missing or unknown:
            parts = [
                f"missing {missing}" if missing else "",
                f"unknown {unknown}" if unknown else "",
            ]
            raise ValueError(
                f"These {what} don't match the latent space: "
                + "; ".join(p for p in parts if p)
                + ". From a different model?"
            )
        return ξ

    def to_model(self, ξ: Latent) -> ShareModule:
        """The model with every free parameter set to its value at ξ (one draw).

        Shared parameters are written once, at their parent path. Fixed
        parameters and Known values keep their values. Works under jit/grad/vmap.
        """
        return self.model.set(self.to_params(as_latent_dict(ξ)))  # type: ignore[no-any-return]

    def from_model(
        self, model: ShareModule, fallback: Literal["zero", "raise"] = "zero"
    ) -> Latent:
        """The latents ξ that reproduce `model`'s current free parameter values.

        Needs each prior's inverse transform. For priors without one, and for
        values outside a prior's support (e.g. 0 for a LogNormal, which gives a
        non-finite latent), `fallback` decides: "zero" sets ξ = 0 (the prior
        median) and warns, listing those parameters; "raise" raises.
        """
        if fallback not in ("zero", "raise"):
            raise ValueError(f"fallback must be 'zero' or 'raise', got {fallback!r}.")
        if not isinstance(model, ShareModule):
            raise TypeError(f"from_model needs a built model, got {type(model).__name__}.")
        locked = model.for_evaluation() if not model._locked else model
        for p in self.paths:
            try:
                shape = use_path_get_leaf(locked.model, str_to_leafpath(p)).val.shape
            except (AttributeError, KeyError, IndexError, TypeError):
                raise ValueError(
                    f"The model has no parameter '{p}': it doesn't have this latent space's structure."
                ) from None
            if shape != self.shapes[p].shape:
                raise ValueError(
                    f"The model's '{p}' has shape {shape}, but the latent space expects "
                    f"{self.shapes[p].shape}: it doesn't have this latent space's structure."
                )
        no_inverse = [p for p in self.paths if not self.priors[p].has_inverse]
        if no_inverse and fallback == "raise":
            raise ValueError(f"These parameters' priors have no inverse transform: {no_inverse}")
        ξ, outside = {}, []
        for p in self.paths:
            if p in no_inverse:
                continue
            val = use_path_get_leaf(locked.model, str_to_leafpath(p)).val
            ξ_p = self.priors[p].inverse(val)
            if not bool(jnp.all(jnp.isfinite(ξ_p))):
                outside.append(p)
            else:
                ξ[p] = ξ_p
        if outside and fallback == "raise":
            raise ValueError(
                f"These parameters' values are outside their prior's support: {outside}"
            )
        for p in no_inverse + outside:
            ξ[p] = jnp.zeros(self.shapes[p].shape, self.shapes[p].dtype)
        messages = []
        if no_inverse:
            messages.append(f"priors with no inverse transform: {no_inverse}")
        if outside:
            messages.append(f"values outside the prior's support: {outside}")
        if messages:
            warnings.warn(
                "from_model: latents set to 0 (the prior median) for " + "; ".join(messages),
                UserWarning,
                stacklevel=2,
            )
        return {p: ξ[p] for p in self.paths}

    def sample_prior_latent(self, key: PRNGKeyArray, n: int | None = None) -> Latent:
        """Draw latents from the prior, i.e. independent standard normals.

        With `n`, returns n draws stacked along a leading axis of each array.
        """
        batch = () if n is None else (n,)
        keys = random.split(key, len(self.paths))
        return {
            p: random.normal(k, batch + self.shapes[p].shape, self.shapes[p].dtype)
            for p, k in zip(self.paths, keys)
        }

    def sample_prior_params(self, key: PRNGKeyArray, n: int | None = None) -> dict[str, Array]:
        """Draw the free parameters' values from their priors.

        With `n`, returns n draws stacked along a leading axis of each array.
        """
        return self.to_params(self.sample_prior_latent(key, n))

    def sample_prior_model(
        self, key: PRNGKeyArray, n: int | None = None
    ) -> ShareModule | list[ShareModule]:
        """A model with parameters drawn from their priors; with `n`, a list of n models."""
        ξ = self.sample_prior_latent(key, n)
        if n is None:
            return self.to_model(ξ)
        return [self.to_model({p: v[i] for p, v in ξ.items()}) for i in range(n)]

    def zeros(self) -> Latent:
        """Latents at zero (every parameter at its prior median)."""
        return {p: jnp.zeros(s.shape, s.dtype) for p, s in self.shapes.items()}

    def _sample_values(self, samples: Any, samples_type: str, caller: str) -> dict[str, Array]:
        """Parameter values of stacked samples, given as latents or as values."""
        if samples_type == "latent":
            values = self.to_params(samples)
        elif samples_type == "params":
            values = self._checked(as_latent_samples(samples), what="parameter values")
        else:
            raise ValueError(f'samples_type must be "latent" or "params", got {samples_type!r}.')
        if any(jnp.ndim(values[p]) <= len(self.shapes[p].shape) for p in self.paths):
            raise ValueError(
                f"{caller} needs samples stacked along a leading axis; got a single "
                "draw (use to_params or to_model for one draw)."
            )
        return values

    def map_samples(
        self,
        fn: Callable[[ShareModule], Any],
        samples: Any,
        samples_type: Literal["latent", "params"] = "latent",
    ) -> Draws:
        """Evaluate `fn(model)` at every sample, stacked along the samples' leading axes.

        For any quantity derived from the model (a field on a grid, the
        predictions, a line ratio), this gives its value at each posterior
        sample; its posterior mean and spread are then the mean and std over
        the leading axis. Several leading axes, e.g. (chain, draw), are kept.

        Args:
            fn: Takes a model and returns an array (or a pytree of arrays).
                Must be JAX-traceable: it runs under `jax.vmap`. The model it
                gets has shared values filled in (as by `for_evaluation`), so
                any part of it can be called.
            samples: Stacked samples, as for `posterior_summary`.
            samples_type: "latent" (default) or "params", as for `posterior_summary`.

        Returns:
            `Draws` of `fn`'s output: `.mean()`, `.std()`, `.quantile(q)` etc.
            summarise over the samples; `.values` holds the stacked outputs.
        """
        values = self._sample_values(samples, samples_type, "map_samples")
        batch = self._sample_shape(values)
        flat = {p: jnp.reshape(v, (-1, *self.shapes[p].shape)) for p, v in values.items()}
        out = jax.vmap(lambda θ: fn(self.model.set(θ).for_evaluation()))(flat)
        stacked = jax.tree.map(lambda x: jnp.reshape(x, (*batch, *jnp.shape(x)[1:])), out)
        return Draws(stacked, n_sample_axes=len(batch))

    def draws(
        self,
        samples: Any,
        of: Literal["params", "latent", "model"] = "params",
        samples_type: Literal["latent", "params"] = "latent",
    ) -> "Draws | ModelDraws":
        """Samples as draws of the parameter values (default), the latents, or the model.

        For "params" and "latent", returns `Draws`: `.mean()`, `.std()`,
        `.quantile(q)` etc. give {path: summary}, and `draws[path]` is one
        parameter's draws. For "model", returns `ModelDraws`: call it, or any
        part of it, to get `Draws` of that part's output, e.g.
        `draws(samples, of="model").field(grid).mean()`.

        Args:
            samples: Stacked samples, as for `posterior_summary`.
            of: "params" (default), "latent" or "model".
            samples_type: What `samples` are: "latent" (default) or "params".
        """
        if of == "model":
            return ModelDraws(self, self._sample_values(samples, samples_type, "draws"))
        if of == "params":
            values = self._sample_values(samples, samples_type, "draws")
        elif of == "latent":
            if samples_type != "latent":
                raise ValueError(
                    "Draws of the latents need latent samples (samples_type='latent')."
                )
            values = self._checked(as_latent_samples(samples))
            self._sample_values(values, "latent", "draws")  # checks the sample axis
        else:
            raise ValueError(f'of must be "params", "latent" or "model", got {of!r}.')
        return Draws(values, n_sample_axes=len(self._sample_shape(values)))

    def _sample_shape(self, values: dict[str, Array]) -> tuple[int, ...]:
        p0 = self.paths[0]
        return tuple(jnp.shape(values[p0])[: jnp.ndim(values[p0]) - len(self.shapes[p0].shape)])

    def posterior_summary(
        self,
        samples: Any,
        samples_type: Literal["latent", "params"] = "latent",
        print_table: bool = True,
        true_latents: Any = None,
        true_params: Any = None,
    ) -> dict[str, Any]:
        """Summarise samples as parameter values (mean and std).

        Args:
            samples: Samples with a leading sample axis: by default latents (a
                NIFTy `Samples` object, or a dict from path to stacked latents,
                as inference returns them).
            samples_type: "latent" (default) or "params". Use "params" for
                samples that are already parameter values, e.g. from `to_params`
                or `sample_prior_params`; with the default they would be
                transformed a second time.
            print_table: Print a table like `get_parameter_summary`.
            true_latents: Optionally, the true latents (one draw), e.g. for
                simulated data. Adds the truth and z = (mean - truth) / std.
            true_params: The same, given as parameter values. Give at most one
                of `true_latents` and `true_params`.

        Returns:
            Dict from path to {"mean": array, "std": array} of the parameter
            values, with "truth" and "z" added when a truth is given.
        """
        values = self._sample_values(samples, samples_type, "posterior_summary")
        draws = Draws(values, n_sample_axes=len(self._sample_shape(values)))
        means, stds = draws.mean(), draws.std()
        summary = {p: {"mean": means[p], "std": stds[p]} for p in values}
        truth = self._truth(true_latents, true_params)
        if truth is not None:
            for p, s in summary.items():
                s["truth"] = truth[p]
                s["z"] = jnp.where(s["std"] > 0, (s["mean"] - truth[p]) / s["std"], jnp.nan)
        if print_table:
            self._print_summary(summary, n=draws.n_draws)
        return summary

    def _truth(self, true_latents: Any, true_params: Any) -> dict[str, Array] | None:
        """The true parameter values (one draw) from either form, or None."""
        if true_latents is not None and true_params is not None:
            raise ValueError("Give true_latents or true_params, not both.")
        if true_latents is not None:
            truth = self.to_params(as_latent_dict(true_latents))
        elif true_params is not None:
            truth = self._checked(as_latent_dict(true_params), what="parameter values")
        else:
            return None
        for p in self.paths:
            if jnp.shape(truth[p]) != self.shapes[p].shape:
                raise ValueError(
                    f"The truth must be one draw: '{p}' has shape {jnp.shape(truth[p])}, "
                    f"expected {self.shapes[p].shape}."
                )
        return truth

    def _print_summary(self, summary: dict[str, Any], n: int) -> None:
        from rich.table import Table
        from rich.text import Text

        from spectracles.model.formatting import ARRAY_DISPLAY_THRESHOLD, get_console

        def fmt(mean: Array, std: Array) -> str:
            if mean.size <= ARRAY_DISPLAY_THRESHOLD:
                pairs = [f"{m:.4g} ± {s:.3g}" for m, s in zip(mean.ravel(), std.ravel())]
                return pairs[0] if len(pairs) == 1 else "[" + ", ".join(pairs) + "]"
            return f"mean {jnp.mean(mean):.4g}, median std {jnp.median(std):.3g}"

        def fmt_values(x: Array, fmt_one: str = ".4g") -> str:
            if x.size <= ARRAY_DISPLAY_THRESHOLD:
                items = [f"{v:{fmt_one}}" for v in x.ravel()]
                return items[0] if len(items) == 1 else "[" + ", ".join(items) + "]"
            return f"mean {jnp.mean(x):.4g}"

        def fmt_z(z: Array) -> str:
            if z.size <= ARRAY_DISPLAY_THRESHOLD:
                return fmt_values(z, "+.2f")
            return f"rms {jnp.sqrt(jnp.nanmean(z**2)):.2f}, max |z| {jnp.nanmax(jnp.abs(z)):.2f}"

        with_truth = "truth" in summary[self.paths[0]]

        table = Table(
            title="Posterior Summary",
            show_header=True,
            header_style="bold",
            border_style="dim",
            title_style="bold",
        )
        table.add_column("Path", style="white", overflow="fold")
        table.add_column("Shape", style="dim")
        table.add_column("Prior", style="dim")
        table.add_column("Posterior mean ± std")
        if with_truth:
            table.add_column("Truth")
            table.add_column("z")
        for p in self.paths:
            s = summary[p]
            row = [p, str(self.shapes[p].shape), repr(self.priors[p]), fmt(s["mean"], s["std"])]
            if with_truth:
                row += [fmt_values(s["truth"]), fmt_z(s["z"])]
            table.add_row(*row)
        console = get_console()
        console.print(table)
        footer = f"{n} samples · {len(self.paths)} parameters"
        if with_truth:
            footer += " · z = (mean − truth) / std"
        console.print(Text(footer, style="dim"))
