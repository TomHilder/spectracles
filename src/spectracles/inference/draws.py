"""draws.py - Draws of a quantity from a posterior (or prior), with summaries over the draws."""

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

if TYPE_CHECKING:
    from spectracles.inference.latent import LatentSpace


class Draws:
    """Draws of a quantity, stacked along leading sample axes, with summaries over them.

    The quantity can be an array (e.g. a field on a grid) or a dict of arrays
    (e.g. parameter values keyed by path). The summaries reduce over the sample
    axes only, so `draws.mean()` has the quantity's own shape (or is a dict of
    them). Draws from several chains, stacked as (chain, draw, ...), reduce
    over both axes.

    Args:
        values: The stacked draws: an array, or a dict of arrays.
        n_sample_axes: How many leading axes index the draws (default 1).

    Attributes:
        values: The stacked draws, as given.
    """

    def __init__(self, values: Any, n_sample_axes: int = 1):
        self.values = values
        self.n_sample_axes = n_sample_axes
        leaves = jax.tree.leaves(values)
        if not leaves:
            raise ValueError("Draws needs at least one array.")
        shapes = {jnp.shape(x)[:n_sample_axes] for x in leaves}
        if any(jnp.ndim(x) < n_sample_axes for x in leaves) or len(shapes) != 1:
            raise ValueError(
                f"Every array needs the same {n_sample_axes} leading sample axes; "
                f"got leading shapes {sorted(shapes)}."
            )

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """The shape of the sample axes, e.g. (n_draws,) or (n_chains, n_draws)."""
        return tuple(jnp.shape(jax.tree.leaves(self.values)[0])[: self.n_sample_axes])

    @property
    def n_draws(self) -> int:
        """The total number of draws (over all sample axes)."""
        n = 1
        for size in self.sample_shape:
            n *= size
        return n

    def _reduce(self, fn: Any) -> Any:
        axes = tuple(range(self.n_sample_axes))
        return jax.tree.map(lambda x: fn(x, axes), self.values)

    def mean(self) -> Any:
        """The mean over the draws."""
        return self._reduce(lambda x, axes: jnp.mean(x, axis=axes))

    def std(self) -> Any:
        """The standard deviation over the draws."""
        return self._reduce(lambda x, axes: jnp.std(x, axis=axes))

    def median(self) -> Any:
        """The median over the draws."""
        return self._reduce(lambda x, axes: jnp.median(x, axis=axes))

    def quantile(self, q: float | Array) -> Any:
        """Quantile(s) `q` (in [0, 1]) over the draws; several q add a leading axis."""
        return self._reduce(lambda x, axes: jnp.quantile(x, jnp.asarray(q), axis=axes))

    def __getitem__(self, key: str) -> "Draws":
        """The draws of one entry, when the quantity is a dict (e.g. one parameter)."""
        if not isinstance(self.values, dict):
            raise TypeError("Only draws of a dict (e.g. of parameters) can be indexed by key.")
        if key not in self.values:
            raise KeyError(f"No {key!r} in these draws; they have {list(self.values)}.")
        return Draws(self.values[key], self.n_sample_axes)

    def keys(self) -> list[str]:
        """The keys, when the quantity is a dict."""
        if not isinstance(self.values, dict):
            raise TypeError("These draws are not of a dict.")
        return list(self.values)

    def __repr__(self) -> str:
        if isinstance(self.values, dict):
            what = f"{len(self.values)} entries"
        else:
            what = f"shape {tuple(jnp.shape(self.values)[self.n_sample_axes :])}"
        return f"Draws({self.n_draws} draws, sample shape {self.sample_shape}, {what})"


class ModelDraws:
    """The model at every sample: call it, or any part of it, to get `Draws`.

    `models(data)` evaluates the whole model at every sample;
    `models.field(data)` or `models.line.flux(data)` evaluates that component.
    Attribute paths are checked against the model straight away. Components
    are evaluated with shared values filled in (as by `for_evaluation`), so
    parts that hold shared parameters can be called. For parameter values, use
    `posterior.draws(samples)` instead.

    Obtained with `LatentSpace.draws(samples, of="model")` (or the same on a
    `Posterior`); not constructed directly.
    """

    def __init__(
        self, latent: "LatentSpace", params: dict[str, Array], path: tuple[str, ...] = ()
    ):
        object.__setattr__(self, "_latent", latent)
        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_target", self._resolve(latent.model.for_evaluation().model))

    def _resolve(self, model: Any) -> Any:
        obj = model
        for i, name in enumerate(self._path):
            try:
                obj = getattr(obj, name)
            except AttributeError:
                where = ".".join(self._path[:i]) or "the model"
                fields = (
                    [f.name for f in dataclasses.fields(obj)]
                    if dataclasses.is_dataclass(obj)
                    else []
                )
                hint = f"; it has {fields}" if fields else ""
                raise AttributeError(f"{where} has no attribute {name!r}{hint}.") from None
        return obj

    def __getattr__(self, name: str) -> "ModelDraws":
        if name.startswith("_"):
            raise AttributeError(name)
        return ModelDraws(self._latent, self._params, (*self._path, name))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ModelDraws is read-only.")

    def __call__(self, *args: Any, **kwargs: Any) -> Draws:
        """Call this part of the model at every sample, with these arguments."""
        name = ".".join(self._path) or "the model"
        if not callable(self._target):
            raise TypeError(
                f"{name} is not callable. For parameter values, use "
                f"posterior.draws(samples)[path]."
            )

        def evaluate(model: Any) -> Any:
            return self._resolve(model.model)(*args, **kwargs)

        return self._latent.map_samples(evaluate, self._params, samples_type="params")

    def __repr__(self) -> str:
        name = ".".join(self._path) or "model"
        n = Draws(self._params, len(self._latent._sample_shape(self._params))).n_draws
        return f"ModelDraws({name}, {n} draws)"
