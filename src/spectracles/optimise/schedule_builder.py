"""schedule_builder.py - Parameter-centric API for building optimization schedules with less boilerplate."""

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Callable, Literal, Optional, Union

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation, sgd, adam  # type: ignore[import]

from spectracles.model.share_module import ShareModule
from spectracles.optimise.opt_frame import get_opt_filter_spec
from spectracles.optimise.opt_schedule import (
    ManagedOptimiserSchedule,
    OptimiserSchedule,
    PhaseConfig,
)


# --- Parameter Spec Types ---


@dataclass
class ParamSpec:
    """Specification for a parameter across phases."""

    free_phases: set[int] = field(default_factory=set)
    init_specs: dict[int, "InitSpec"] = field(default_factory=dict)

    def __or__(self, other: "ParamSpec") -> "ParamSpec":
        """Combine two ParamSpecs using the | operator."""
        combined_free = self.free_phases | other.free_phases
        combined_init = {**self.init_specs, **other.init_specs}
        return ParamSpec(free_phases=combined_free, init_specs=combined_init)


@dataclass
class InitSpec:
    """Specification for parameter initialization."""

    kind: Literal["normal", "value", "uniform"]
    value: Optional[float] = None
    mean: float = 0.0
    std: float = 1.0
    low: float = 0.0
    high: float = 1.0


# --- Helper Functions ---


def free_in(*phases: int) -> ParamSpec:
    """Parameter is free (optimized) in the specified phases.

    Args:
        *phases: Phase indices where the parameter should be free.

    Returns:
        ParamSpec with the specified free phases.

    Example:
        >>> spec = free_in(0, 1, 2)  # Free in phases 0, 1, and 2
    """
    return ParamSpec(free_phases=set(phases))


def free_after(phase: int) -> ParamSpec:
    """Parameter is free from the specified phase onwards.

    Note: This creates a ParamSpec with a marker for "free from phase N onwards".
    The actual phases will be resolved when build_schedule is called.

    Args:
        phase: Phase index from which the parameter should be free.

    Returns:
        ParamSpec with a marker for late resolution.

    Example:
        >>> spec = free_after(1)  # Free in phases 1, 2, 3, ...
    """
    # We use a negative phase index as a sentinel for "free_after"
    # The actual resolution happens in build_schedule when we know n_phases
    return ParamSpec(free_phases={-phase - 1})  # -1 means "after 0", -2 means "after 1", etc.


def free_until(phase: int) -> ParamSpec:
    """Parameter is free up to and including the specified phase.

    Args:
        phase: Last phase where the parameter should be free.

    Returns:
        ParamSpec with the specified free phases.

    Example:
        >>> spec = free_until(2)  # Free in phases 0, 1, and 2
    """
    return ParamSpec(free_phases=set(range(phase + 1)))


def fixed_in(*phases: int) -> ParamSpec:
    """Parameter is fixed (not optimized) in the specified phases.

    This is syntactic sugar - you specify which phases to FIX, and
    the parameter is free in all other phases.

    Note: Requires knowing total number of phases at resolution time.

    Args:
        *phases: Phase indices where the parameter should be fixed.

    Returns:
        ParamSpec with a marker for "fixed in these phases".

    Example:
        >>> spec = fixed_in(0)  # Fixed only in phase 0, free elsewhere
    """
    # Use a special marker (frozenset in the init_specs) to indicate "fixed_in"
    spec = ParamSpec()
    spec.init_specs[-1] = InitSpec(kind="value", value=None)  # Sentinel
    spec.free_phases = {-1000 - p for p in phases}  # Sentinel values
    return spec


def init_normal(phase: int, mean: float = 0.0, std: float = 1.0) -> ParamSpec:
    """Initialize parameter to random normal values at the start of a phase.

    Args:
        phase: Phase index at which to initialize.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.

    Returns:
        ParamSpec with the initialization specification.

    Example:
        >>> spec = free_in(0, 1) | init_normal(0, std=0.1)
    """
    return ParamSpec(init_specs={phase: InitSpec(kind="normal", mean=mean, std=std)})


def init_value(phase: int, value: float) -> ParamSpec:
    """Initialize parameter to a specific value at the start of a phase.

    Args:
        phase: Phase index at which to initialize.
        value: The value to set.

    Returns:
        ParamSpec with the initialization specification.

    Example:
        >>> spec = free_after(1) | init_value(1, 1.5)
    """
    return ParamSpec(init_specs={phase: InitSpec(kind="value", value=value)})


def init_uniform(phase: int, low: float = 0.0, high: float = 1.0) -> ParamSpec:
    """Initialize parameter to random uniform values at the start of a phase.

    Args:
        phase: Phase index at which to initialize.
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.

    Returns:
        ParamSpec with the initialization specification.

    Example:
        >>> spec = free_in(0) | init_uniform(0, low=-1, high=1)
    """
    return ParamSpec(init_specs={phase: InitSpec(kind="uniform", low=low, high=high)})


# --- Path Matching ---


def _match_pattern(pattern: str, path: str) -> bool:
    """Match a glob pattern against a parameter path.

    Supports:
    - '*' matches any single path component
    - '**' matches any number of components (including zero)
    - Exact matches

    Args:
        pattern: Glob pattern (e.g., "gp.*.coefficients", "**.lengthscale")
        path: Parameter path to match (e.g., "gp.kernel.lengthscale")

    Returns:
        True if the pattern matches the path.
    """
    # Handle ** (recursive match)
    if "**" in pattern:
        # Split on ** and handle each part
        parts = pattern.split("**")
        if len(parts) == 2:
            prefix, suffix = parts
            prefix = prefix.rstrip(".")
            suffix = suffix.lstrip(".")

            # Check prefix
            if prefix and not (path.startswith(prefix) or fnmatch(path.split(".")[0], prefix.split(".")[0] if prefix else "*")):
                if prefix:
                    prefix_parts = prefix.split(".")
                    path_parts = path.split(".")
                    if len(path_parts) < len(prefix_parts):
                        return False
                    for pp, pat in zip(path_parts, prefix_parts):
                        if not fnmatch(pp, pat):
                            return False

            # Check suffix
            if suffix:
                suffix_parts = suffix.split(".")
                path_parts = path.split(".")
                if len(path_parts) < len(suffix_parts):
                    return False
                # Match from the end
                for pp, pat in zip(reversed(path_parts), reversed(suffix_parts)):
                    if not fnmatch(pp, pat):
                        return False

            # If we got here with valid prefix/suffix checks, it matches
            if not prefix and not suffix:
                return True  # "**" alone matches everything
            if not prefix:
                # Just suffix - check if path ends with suffix pattern
                return _simple_match(suffix, ".".join(path.split(".")[-len(suffix.split(".")):]))
            if not suffix:
                # Just prefix - check if path starts with prefix pattern
                return _simple_match(prefix, ".".join(path.split(".")[:len(prefix.split("."))]))
            return True

    # Use simple matching for non-recursive patterns
    return _simple_match(pattern, path)


def _simple_match(pattern: str, path: str) -> bool:
    """Simple glob matching where * matches a single component."""
    pattern_parts = pattern.split(".")
    path_parts = path.split(".")

    if len(pattern_parts) != len(path_parts):
        return False

    for pat, p in zip(pattern_parts, path_parts):
        if not fnmatch(p, pat):
            return False

    return True


def expand_patterns(
    patterns: dict[str, ParamSpec],
    parent_paths: list[str],
    shared_paths: list[str],
) -> dict[str, ParamSpec]:
    """Expand glob patterns to actual parameter paths.

    Only matches against parent (non-shared) paths. If a pattern matches
    only shared paths, raises an error explaining that shared paths should
    not be specified directly.

    Args:
        patterns: Dictionary mapping patterns to ParamSpecs.
        parent_paths: List of parent (non-shared) parameter paths.
        shared_paths: List of shared parameter paths.

    Returns:
        Dictionary mapping actual paths to ParamSpecs.

    Raises:
        ValueError: If a pattern doesn't match any parent paths, or if it
            only matches shared paths.
    """
    expanded: dict[str, ParamSpec] = {}

    for pattern, spec in patterns.items():
        matched_parents = []
        matched_shared = []

        # Check matches against parent paths
        for path in parent_paths:
            if _match_pattern(pattern, path):
                matched_parents.append(path)

        # Check matches against shared paths
        for path in shared_paths:
            if _match_pattern(pattern, path):
                matched_shared.append(path)

        # Validate matches
        if not matched_parents:
            if matched_shared:
                # Pattern matched shared paths but no parent paths
                raise ValueError(
                    f"Pattern '{pattern}' only matches shared parameters: {matched_shared}. "
                    f"Shared parameters cannot be specified directly - use the parent path instead. "
                    f"Available parent paths: {parent_paths}"
                )
            else:
                # Pattern matched nothing
                raise ValueError(
                    f"Pattern '{pattern}' did not match any parameters. "
                    f"Available paths: {parent_paths}"
                )

        # Add matched parent paths
        for path in matched_parents:
            if path in expanded:
                # Combine specs for the same path
                expanded[path] = expanded[path] | spec
            else:
                expanded[path] = spec

    return expanded


# --- Phase Resolution ---


def _resolve_free_phases(spec: ParamSpec, n_phases: int) -> set[int]:
    """Resolve the free phases, handling sentinels like free_after."""
    resolved = set()

    for phase in spec.free_phases:
        if phase >= 0:
            # Normal phase index
            resolved.add(phase)
        elif phase <= -1000:
            # fixed_in sentinel - skip, handled separately
            pass
        else:
            # free_after sentinel: -1 means after 0, -2 means after 1, etc.
            start_phase = -phase - 1
            resolved.update(range(start_phase, n_phases))

    # Handle fixed_in by computing complement
    fixed_phases = {-p - 1000 for p in spec.free_phases if p <= -1000}
    if fixed_phases:
        # free in all phases except the fixed ones
        all_phases = set(range(n_phases))
        resolved = all_phases - fixed_phases

    return resolved


# --- Schedule Building ---


PhaseSpec = tuple[int, float]  # (n_steps, learning_rate)
PhaseSpecFull = tuple[int, float, Optional[GradientTransformation]]  # (n_steps, lr, optimizer)


def build_schedule(
    model: ShareModule,
    loss_fn: Callable[..., float],
    phases: list[Union[PhaseSpec, PhaseSpecFull]],
    params: dict[str, ParamSpec],
    default_optimizer: Literal["sgd", "adam"] = "adam",
    managed: bool = False,
    get_filter_spec_fn: Callable[[ShareModule], Callable] = get_opt_filter_spec,
    key: Optional[PRNGKeyArray] = None,
) -> Union[OptimiserSchedule, ManagedOptimiserSchedule]:
    """Build an optimization schedule using a parameter-centric specification.

    This function provides a declarative way to specify which parameters should
    be free or fixed in each phase, with automatic pattern expansion for
    parameter paths.

    Args:
        model: The ShareModule to optimize.
        loss_fn: Loss function to minimize.
        phases: List of phase specifications. Each is either:
            - (n_steps, learning_rate): Uses default optimizer
            - (n_steps, learning_rate, optimizer): Uses specified optimizer
        params: Dictionary mapping parameter patterns to ParamSpecs.
            Patterns support glob-style matching:
            - "*" matches any single component
            - "**" matches any number of components
        default_optimizer: Default optimizer type ("sgd" or "adam").
        managed: If True, return a ManagedOptimiserSchedule with state tracking.
        get_filter_spec_fn: Function to get filter spec for optimization.
        key: PRNG key for random initializations. Required if any init_normal
            or init_uniform specs are used.

    Returns:
        An OptimiserSchedule or ManagedOptimiserSchedule ready to run.

    Example:
        >>> schedule = build_schedule(
        ...     model, loss_fn,
        ...     phases=[
        ...         (100, 0.1),    # Phase 0: 100 steps, lr=0.1
        ...         (50, 0.01),    # Phase 1: 50 steps, lr=0.01
        ...         (50, 0.001),   # Phase 2: 50 steps, lr=0.001
        ...     ],
        ...     params={
        ...         "gp.coefficients": free_in(0, 1, 2),
        ...         "gp.kernel.*": free_after(1),
        ...         "line.amplitude": free_in(1, 2) | init_normal(0),
        ...     },
        ... )
        >>> schedule.run_all(data)
    """
    n_phases = len(phases)

    # Get parent paths (unique parameters) and shared paths separately
    parent_paths = model.get_parameter_paths(show_shared=False)
    all_paths = model.get_parameter_paths(show_shared=True)
    shared_paths = [p for p in all_paths if p not in parent_paths]

    # Expand patterns to actual paths (validates that patterns only match parent paths)
    expanded_params = expand_patterns(params, parent_paths, shared_paths)

    # Resolve free phases for each parameter
    resolved_params: dict[str, tuple[set[int], dict[int, InitSpec]]] = {}
    for path, spec in expanded_params.items():
        free_phases = _resolve_free_phases(spec, n_phases)
        resolved_params[path] = (free_phases, spec.init_specs)

    # Build PhaseConfigs
    phase_configs: list[PhaseConfig] = []

    # Track PRNG key usage
    current_key = key

    for phase_idx, phase_spec in enumerate(phases):
        # Parse phase spec
        if len(phase_spec) == 2:
            n_steps, lr = phase_spec
            opt = adam(lr) if default_optimizer == "adam" else sgd(lr)
        else:
            n_steps, lr, opt = phase_spec
            if opt is None:
                opt = adam(lr) if default_optimizer == "adam" else sgd(lr)

        # Determine fix status for each parameter
        fix_updates: dict[str, bool] = {}
        val_updates: dict[str, Array] = {}

        for path, (free_phases, init_specs) in resolved_params.items():
            # Set fix status: fixed = True means NOT free
            fix_updates[path] = phase_idx not in free_phases

            # Handle initializations
            if phase_idx in init_specs:
                init_spec = init_specs[phase_idx]
                param_shape = _get_param_shape(model, path)

                if init_spec.kind == "value":
                    if init_spec.value is not None:
                        val_updates[path] = jnp.full(param_shape, init_spec.value)
                elif init_spec.kind == "normal":
                    if current_key is None:
                        raise ValueError(
                            f"PRNG key required for init_normal on '{path}'. "
                            "Pass key=jr.key(seed) to build_schedule."
                        )
                    current_key, subkey = jr.split(current_key)
                    val_updates[path] = (
                        init_spec.mean + init_spec.std * jr.normal(subkey, param_shape)
                    )
                elif init_spec.kind == "uniform":
                    if current_key is None:
                        raise ValueError(
                            f"PRNG key required for init_uniform on '{path}'. "
                            "Pass key=jr.key(seed) to build_schedule."
                        )
                    current_key, subkey = jr.split(current_key)
                    val_updates[path] = jr.uniform(
                        subkey, param_shape, minval=init_spec.low, maxval=init_spec.high
                    )

        phase_configs.append(
            PhaseConfig(
                n_steps=n_steps,
                optimiser=opt,
                fix_status_updates=fix_updates,
                param_val_updates=val_updates,
            )
        )

    # Create the schedule
    if managed:
        return ManagedOptimiserSchedule(
            model, loss_fn, phase_configs, get_filter_spec_fn
        )
    else:
        return OptimiserSchedule(model, loss_fn, phase_configs, get_filter_spec_fn)


def _get_param_shape(model: ShareModule, path: str) -> tuple[int, ...]:
    """Get the shape of a parameter by path."""
    from spectracles.tree.path_utils import str_to_leafpath, use_path_get_leaf

    leaf_path = str_to_leafpath(path)
    param = use_path_get_leaf(model, leaf_path)
    return param.val.shape
