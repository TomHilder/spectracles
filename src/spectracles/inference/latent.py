"""latent.py - The latent space of a model: standard-normal latents for its free parameters.

Every free parameter with a prior gets one entry in a dict ξ of standard-normal
latents, keyed by its (parent) parameter path. Its value is `prior.transform(ξ[path])`.
Shared parameters are one parameter, so they have one entry, at the parent path.
"""

import math
import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, PRNGKeyArray

from spectracles.model.parameter import Known, Parameter
from spectracles.model.share_module import ShareModule
from spectracles.priors import Prior
from spectracles.tree.path_utils import str_to_leafpath, use_path_get_leaf

Latent = dict[str, Array]


def as_latent_dict(ξ: Any) -> Latent:
    """Accept a plain dict or a dict-like wrapper with a `.tree` (e.g. a NIFTy `Vector`)."""
    return ξ.tree if hasattr(ξ, "tree") else ξ


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

    def to_model(self, ξ: Latent) -> ShareModule:
        """The model with every free parameter set to `prior.transform(ξ[path])`.

        Shared parameters are written once, at their parent path. Fixed
        parameters and Known values keep their values. Works under jit/grad/vmap.
        """
        ξ = as_latent_dict(ξ)
        values = [self.priors[p].transform(ξ[p]) for p in self.paths]
        return self.model.set(self.paths, values)  # type: ignore[no-any-return]

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
        locked = model.get_locked_model() if not model._locked else model
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

    def sample(self, key: PRNGKeyArray) -> Latent:
        """Draw latents from the prior, i.e. independent standard normals."""
        keys = random.split(key, len(self.paths))
        return {
            p: random.normal(k, self.shapes[p].shape, self.shapes[p].dtype)
            for p, k in zip(self.paths, keys)
        }

    def prior_sample(self, key: PRNGKeyArray) -> ShareModule:
        """A model with parameters drawn from their priors."""
        return self.to_model(self.sample(key))

    def zeros(self) -> Latent:
        """Latents at zero (every parameter at its prior median)."""
        return {p: jnp.zeros(s.shape, s.dtype) for p, s in self.shapes.items()}

    def posterior_summary(self, samples: Any, print_table: bool = True) -> dict[str, Any]:
        """Summarise samples of the latents as parameter values (mean and std).

        Args:
            samples: A NIFTy `Samples` object, or a dict from path to stacked
                latent samples with a leading sample axis.
            print_table: Print a table like `get_parameter_summary`.

        Returns:
            Dict from path to {"mean": array, "std": array} of the parameter values.
        """
        if hasattr(samples, "samples") and not isinstance(samples, dict):
            samples = samples.samples  # NIFTy Samples: stacked, offset by pos
        stacked = as_latent_dict(samples)
        summary = {}
        for p in self.paths:
            values = jax.vmap(self.priors[p].transform)(stacked[p])
            summary[p] = {"mean": jnp.mean(values, axis=0), "std": jnp.std(values, axis=0)}
        if print_table:
            self._print_summary(summary, n=int(jnp.shape(stacked[self.paths[0]])[0]))
        return summary

    def _print_summary(self, summary: dict[str, Any], n: int) -> None:
        from rich.table import Table
        from rich.text import Text

        from spectracles.model.formatting import ARRAY_DISPLAY_THRESHOLD, get_console

        def fmt(mean: Array, std: Array) -> str:
            if mean.size <= ARRAY_DISPLAY_THRESHOLD:
                pairs = [f"{m:.4g} ± {s:.3g}" for m, s in zip(mean.ravel(), std.ravel())]
                return pairs[0] if len(pairs) == 1 else "[" + ", ".join(pairs) + "]"
            return f"mean {jnp.mean(mean):.4g}, median std {jnp.median(std):.3g}"

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
        for p in self.paths:
            table.add_row(
                p,
                str(self.shapes[p].shape),
                repr(self.priors[p]),
                fmt(summary[p]["mean"], summary[p]["std"]),
            )
        console = get_console()
        console.print(table)
        console.print(Text(f"{n} samples · {len(self.paths)} parameters", style="dim"))
