"""opt_schedule.py - OptimiserSchedule encapsulates many OptimiserFrames in a sequence for more control over how the optimisation proceeds."""

from dataclasses import dataclass, field
from typing import Callable, Literal, get_args

from jaxtyping import Array
from optax import GradientTransformation  # type: ignore[import]

from modelling_lib.model.share_module import ShareModule
from modelling_lib.optimise.opt_frame import OptimiserFrame, get_opt_filter_spec

ExitStrategy = Literal[None, "placeholder"]


@dataclass(frozen=True)
class PhaseConfig:
    n_steps: int
    optimiser: GradientTransformation
    exit_strategy: ExitStrategy = field(default=None)
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
        if self.exit_strategy not in get_args(ExitStrategy):
            raise ValueError(f"Unkown exit strategy: {self.exit_strategy}")


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
                    ),
                )
            )
