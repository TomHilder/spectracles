"""nifty.py - NIFTy.re adapter (optional: pip install spectracles[nifty]).

`to_nifty` turns a `Posterior` into a NIFTy.re likelihood on the same latents ξ,
for MGVI/geoVI with `jft.optimize_kl`. `stages` writes staged VI (which
parameters are frozen at which iterations) in the schedule-builder vocabulary.

Importing this module never imports NIFTy; only `to_nifty` needs it.
"""

from typing import Any

from spectracles.inference.latent import LatentSpace
from spectracles.inference.posterior import Posterior
from spectracles.optimise.schedule_builder import ParamSpec, _resolve_free_phases
from spectracles.tree.path_utils import resolve_patterns_to_parents


def _import_nifty():
    """Import nifty.re, with a helpful error if it isn't installed."""
    try:
        import nifty.re as jft
    except ImportError as e:
        raise ImportError(
            "This needs NIFTy.re, which is an optional dependency of spectracles. "
            "Install it with: pip install 'spectracles[nifty]'"
        ) from e
    return jft


def to_nifty(posterior: Posterior) -> Any:
    """The posterior's likelihood as a NIFTy.re likelihood on the latents ξ.

    The returned `jft.Likelihood` is already amended with the forward model, so
    its domain is the dict of latents (keyed by parameter path) and NIFTy's
    standard-normal prior on them is exactly the latent-space prior. Use it with
    `jft.optimize_kl`; turn results back into models with
    `posterior.latent.to_model` and `posterior.latent.posterior_summary`.
    """
    jft = _import_nifty()
    likelihood = posterior.likelihood
    domain = {p: jft.ShapeWithDtype(s.shape, s.dtype) for p, s in posterior.latent.shapes.items()}

    def forward(ξ):
        return likelihood.compress(posterior.predict(ξ))

    model = jft.Model(forward, domain=domain, white_init=True)
    return likelihood.to_nifty().amend(model)


def _max_phase_index(spec: ParamSpec) -> int:
    """The largest iteration index a spec refers to explicitly (-1 if none)."""
    indices = []
    for phase in spec.free_phases:
        if phase >= 0:
            indices.append(phase)
        elif phase <= -1000:
            indices.append(-phase - 1000)  # fixed_in
        else:
            indices.append(-phase - 1)  # free_after
    return max(indices, default=-1)


def stages(latent: LatentSpace, spec: dict[str, ParamSpec], n_iterations: int) -> dict:
    """Staged VI in the schedule-builder vocabulary, for `jft.optimize_kl`.

    Each pattern (globs as in `build_schedule`) maps to a `ParamSpec` built with
    `free_in`, `free_after`, `free_until` or `fixed_in`, with VI iterations in
    place of optimisation phases. A parameter not mentioned is free in every
    iteration. A parameter that is not free at iteration i is held fixed at its
    current value: it goes in both `constants` (not updated by the KL
    minimisation) and `point_estimates` (no samples drawn along it).

    Args:
        latent: The posterior's latent space.
        spec: Mapping from path pattern to ParamSpec. A shared parameter can be
            named by any of its paths.
        n_iterations: The number of VI iterations (`n_total_iterations`). Specs
            may not refer to later iterations; past the end, the pattern of
            the helpers continues naturally (e.g. free_after stays free).

    Returns:
        {"constants": callable, "point_estimates": callable}, each taking the
        iteration number and returning a tuple of latent keys. Pass with
        `jft.optimize_kl(..., **stages(...))`.
    """
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be at least 1, got {n_iterations}.")

    # schedule_builder encodes free_after(p) as -p - 1 and fixed_in(p) as -1000 - p,
    # which collide from p = 999 on
    if n_iterations > 999:
        raise ValueError(f"stages supports at most 999 iterations, got {n_iterations}.")

    for pattern, param_spec in spec.items():
        if not isinstance(param_spec, ParamSpec):
            raise TypeError(
                f"stages: value for '{pattern}' must be a ParamSpec (free_in, free_after, "
                f"free_until or fixed_in), got {param_spec!r}."
            )
        # fixed_in stores a sentinel init spec at phase -1; real ones are at phases >= 0
        if any(phase >= 0 for phase in param_spec.init_specs):
            raise ValueError(
                f"stages: '{pattern}' has an init_* spec; initialisation has no meaning for "
                f"VI stages (start from a chosen ξ instead)."
            )
        if _max_phase_index(param_spec) >= n_iterations:
            raise ValueError(
                f"stages: '{pattern}' refers to iteration {_max_phase_index(param_spec)}, "
                f"but there are only {n_iterations} iterations."
            )

    # Shared (child) paths resolve to their parent, the latent key
    child_to_parent = latent.model._child_to_parent()
    candidates = list(latent.paths) + [c for c, p in child_to_parent.items() if p in latent.paths]
    resolved = resolve_patterns_to_parents(spec, candidates, child_to_parent, context="stages")
    expanded: dict[str, ParamSpec] = {}
    for key, entries in resolved.items():
        combined = entries[0][1]
        for _, param_spec in entries[1:]:
            combined = combined | param_spec
        expanded[key] = combined
    order = [p for p in latent.paths if p in expanded]

    def frozen(nit: int) -> tuple[str, ...]:
        n = max(n_iterations, nit + 1)
        return tuple(p for p in order if nit not in _resolve_free_phases(expanded[p], n))

    return {"constants": frozen, "point_estimates": frozen}
