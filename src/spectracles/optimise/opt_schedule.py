"""opt_schedule.py - OptimiserSchedule encapsulates many OptimiserFrames in a sequence for more control over how the optimisation proceeds."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Optional, Union, get_args

import jax.numpy as jnp
from jaxtyping import Array
from optax import GradientTransformation  # type: ignore[import]

from spectracles.model.share_module import ShareModule
from spectracles.optimise.opt_frame import OptimiserFrame, get_opt_filter_spec

ExitStrategy = Literal[None, "placeholder"]


@dataclass(frozen=True)
class PhaseConfig:
    n_steps: int
    optimiser: GradientTransformation
    # exit_strategy: ExitStrategy = field(default=None)
    Δloss_criterion: float = field(default=1e2)
    fix_status_updates: dict[str, bool] = field(default_factory=dict)
    param_val_updates: dict[str, Array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_phase_config()

    def _validate_phase_config(self) -> None:
        # Check n_steps ∈ Z^+
        if not isinstance(self.n_steps, int):
            raise TypeError("n_steps must be int.")
        # TODO: Check optimiser is okay somehow?
        ...
        # Check fix updates are bool
        for fix in self.fix_status_updates.values():
            if not isinstance(fix, bool):
                raise TypeError("All values in fix_status_updates must be bool.")
        # Check val updates are Array
        for val in self.param_val_updates.values():
            if not isinstance(val, Array):
                raise TypeError("All values in param_val_updates must be jax Arrays.")
        # Check exit strategy exists
        # if self.exit_strategy not in get_args(ExitStrategy):
        # raise ValueError(f"Unkown exit strategy: {self.exit_strategy}")


@dataclass
class Phase:
    config: PhaseConfig
    frame: OptimiserFrame

    def __post_init__(self) -> None:
        self._validate_phase(self.config, self.frame.model)

    @staticmethod
    def _validate_phase(config: PhaseConfig, model: ShareModule) -> None:
        # Try and check for failures.
        # We'll just use whatever exceptions are raised by methods instead of reraising.
        model.set_fixed_status(
            list(config.fix_status_updates.keys()),
            list(config.fix_status_updates.values()),
        )
        model.set(
            list(config.param_val_updates.keys()),
            list(config.param_val_updates.values()),
        )


class OptimiserSchedule:
    def __init__(
        self,
        model: ShareModule,
        loss_fn: Callable[..., float],
        phase_configs: list[PhaseConfig],
        get_filter_spec_fn: Callable[[ShareModule], Callable] = get_opt_filter_spec,
    ):
        self.model_history = [model]

        # Assemble the Phases
        self.phases = []
        for config in phase_configs:
            self.phases.append(
                Phase(
                    config,
                    OptimiserFrame(
                        model,
                        loss_fn,
                        config.optimiser,
                        get_filter_spec_fn,
                        config.Δloss_criterion,
                    ),
                )
            )

    def run_all(self, *loss_args, **loss_kwargs) -> None:
        """Run all phases in the schedule."""
        for phase in self.phases:
            self.run_phase(phase, *loss_args, **loss_kwargs)

    def run_phase(self, phase: Phase, *loss_args, **loss_kwargs) -> None:
        """Run a single phase in the schedule."""
        # Grab the most recent model from history
        recent_model = self.model_history[-1]
        # Apply the phase updates
        recent_model = recent_model.set_fixed_status(
            list(phase.config.fix_status_updates.keys()),
            list(phase.config.fix_status_updates.values()),
        )
        recent_model = recent_model.set(
            list(phase.config.param_val_updates.keys()),
            list(phase.config.param_val_updates.values()),
        )
        # Now we have the model for this phase, we can reinitialise the state of the frame
        phase.frame._set_opt_state(recent_model)
        # Run the optimiser with the phase's frame
        updated_model = phase.frame.run(phase.config.n_steps, *loss_args, **loss_kwargs)
        # Update the model history
        self.model_history.append(updated_model)

    @property
    def loss_histories(self) -> list[Array]:
        """Get the total loss history from all phases."""
        return [jnp.array(phase.frame.loss_history) for phase in self.phases]

    @property
    def loss_history(self) -> Array:
        """Get the total loss history from all phases."""
        return jnp.concatenate(self.loss_histories)


# TODO: Check the expanded implementation from Claude below that is more general
# (I think it is mostly fine but we should check all the implemented logic actually works)


class PhaseState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class OptimiserScheduleUnsafe:
    def __init__(
        self,
        model: ShareModule,
        loss_fn: Callable[..., float],
        phase_configs: list[PhaseConfig],
        get_filter_spec_fn: Callable[[ShareModule], Callable] = get_opt_filter_spec,
    ):
        self.model_history = [model]
        self.initial_model = model
        self.loss_fn = loss_fn
        self.phase_configs = phase_configs
        self.get_filter_spec_fn = get_filter_spec_fn

        # Create initial phases
        self.phases = []
        self._create_phases()

        # Track phase execution state
        self.phase_states = [PhaseState.PENDING] * len(self.phases)
        self.current_phase_index = 0

    def _create_phases(self):
        """Create fresh Phase objects with new OptimiserFrame instances."""
        self.phases = []
        for config in self.phase_configs:
            self.phases.append(
                Phase(
                    config,
                    OptimiserFrame(
                        self.initial_model,
                        self.loss_fn,
                        config.optimiser,
                        self.get_filter_spec_fn,
                    ),
                )
            )

    def run_all(self, *loss_args, **loss_kwargs) -> None:
        """Run all remaining phases in the schedule."""
        while self.current_phase_index < len(self.phases):
            self.run_next_phase(*loss_args, **loss_kwargs)

    def run_next_phase(self, *loss_args, **loss_kwargs) -> bool:
        """Run the next pending phase. Returns True if a phase was run, False if all complete."""
        if self.current_phase_index >= len(self.phases):
            return False

        phase_idx = self.current_phase_index
        self.run_phase_by_index(phase_idx, *loss_args, **loss_kwargs)
        return True

    def run_phase_by_index(self, phase_idx: int, *loss_args, **loss_kwargs) -> None:
        """Run a specific phase by index, with validation."""
        if phase_idx < 0 or phase_idx >= len(self.phases):
            raise ValueError(
                f"Phase index {phase_idx} out of range [0, {len(self.phases) - 1}]"
            )

        # Validation: can't run phases out of order
        if phase_idx > self.current_phase_index:
            raise RuntimeError(
                f"Cannot run phase {phase_idx} - must complete phases in order. "
                f"Next phase to run is {self.current_phase_index}"
            )

        # Validation: can't re-run completed phases
        if self.phase_states[phase_idx] == PhaseState.COMPLETED:
            raise RuntimeError(f"Phase {phase_idx} has already been completed")

        # Mark as running
        self.phase_states[phase_idx] = PhaseState.RUNNING

        try:
            phase = self.phases[phase_idx]
            self.run_phase(phase, *loss_args, **loss_kwargs)

            # Mark as completed and advance current phase
            self.phase_states[phase_idx] = PhaseState.COMPLETED
            if phase_idx == self.current_phase_index:
                self.current_phase_index += 1

        except Exception as e:
            # Reset state on failure
            self.phase_states[phase_idx] = PhaseState.PENDING
            raise e

    def run_phase(self, phase: Phase, *loss_args, **loss_kwargs) -> None:
        """Run a single phase in the schedule."""
        # Grab the most recent model from history
        recent_model = self.model_history[-1]
        # Apply the phase updates
        recent_model = recent_model.set_fixed_status(
            list(phase.config.fix_status_updates.keys()),
            list(phase.config.fix_status_updates.values()),
        )
        recent_model = recent_model.set(
            list(phase.config.param_val_updates.keys()),
            list(phase.config.param_val_updates.values()),
        )
        # Now we have the model for this phase, we can reinitialise the state of the frame
        phase.frame._set_opt_state(recent_model)
        # Run the optimiser with the phase's frame
        updated_model = phase.frame.run(phase.config.n_steps, *loss_args, **loss_kwargs)
        # Update the model history
        self.model_history.append(updated_model)

    def run_phases(
        self, phase_indices: Union[int, list[int]], *loss_args, **loss_kwargs
    ) -> None:
        """Run multiple phases by index."""
        if isinstance(phase_indices, int):
            phase_indices = [phase_indices]

        for idx in phase_indices:
            self.run_phase_by_index(idx, *loss_args, **loss_kwargs)

    def skip_phase(self, phase_idx: int) -> None:
        """Skip a phase (mark as completed without running)."""
        if phase_idx < 0 or phase_idx >= len(self.phases):
            raise ValueError(f"Phase index {phase_idx} out of range")

        if phase_idx > self.current_phase_index:
            raise RuntimeError(
                f"Cannot skip phase {phase_idx} - must process phases in order"
            )

        if self.phase_states[phase_idx] == PhaseState.COMPLETED:
            raise RuntimeError(f"Phase {phase_idx} has already been completed")

        self.phase_states[phase_idx] = PhaseState.SKIPPED
        if phase_idx == self.current_phase_index:
            self.current_phase_index += 1

    def reset(self) -> None:
        """Reset the schedule to initial state with fresh optimizer frames."""
        self.model_history = [self.initial_model]
        self.phase_states = [PhaseState.PENDING] * len(self.phases)
        self.current_phase_index = 0
        # Create completely fresh Phase objects with new OptimiserFrame instances
        self._create_phases()

    def reset_from_phase(self, phase_idx: int) -> None:
        """Reset schedule from a specific phase onwards with fresh optimizer frames."""
        if phase_idx < 0 or phase_idx >= len(self.phases):
            raise ValueError(f"Phase index {phase_idx} out of range")

        # Reset states for phases from phase_idx onwards
        for i in range(phase_idx, len(self.phases)):
            self.phase_states[i] = PhaseState.PENDING

        # Truncate model history to the point just before phase_idx
        self.model_history = self.model_history[: phase_idx + 1]
        self.current_phase_index = phase_idx

        # Create fresh Phase objects for affected phases
        model_at_phase = self.model_history[phase_idx]
        for i in range(phase_idx, len(self.phases)):
            config = self.phase_configs[i]
            self.phases[i] = Phase(
                config,
                OptimiserFrame(
                    model_at_phase,
                    self.loss_fn,
                    config.optimiser,
                    self.get_filter_spec_fn,
                ),
            )

    # Status and inspection methods
    def get_phase_status(self) -> dict:
        """Get detailed status of all phases."""
        return {
            "current_phase": self.current_phase_index,
            "total_phases": len(self.phases),
            "completed_phases": sum(
                1 for state in self.phase_states if state == PhaseState.COMPLETED
            ),
            "phase_states": [state.value for state in self.phase_states],
            "is_complete": self.is_complete(),
        }

    def is_complete(self) -> bool:
        """Check if all phases are completed."""
        return self.current_phase_index >= len(self.phases)

    def get_next_phase_index(self) -> Optional[int]:
        """Get the index of the next phase to run, or None if complete."""
        return self.current_phase_index if not self.is_complete() else None

    def get_completed_phases(self) -> list[int]:
        """Get indices of all completed phases."""
        return [
            i
            for i, state in enumerate(self.phase_states)
            if state == PhaseState.COMPLETED
        ]

    def get_pending_phases(self) -> list[int]:
        """Get indices of all pending phases."""
        return [
            i
            for i, state in enumerate(self.phase_states)
            if state == PhaseState.PENDING
        ]

    @property
    def loss_histories(self) -> list[Array]:
        """Get the loss history from completed phases only."""
        completed_indices = self.get_completed_phases()
        return [jnp.array(self.phases[i].frame.loss_history) for i in completed_indices]

    @property
    def loss_history(self) -> Array:
        """Get the total loss history from completed phases."""
        histories = self.loss_histories
        return jnp.concatenate(histories) if histories else jnp.array([])
