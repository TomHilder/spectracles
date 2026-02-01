# Optimization API

## Schedule Classes

::: spectracles.optimise.opt_schedule.OptimiserSchedule
    options:
      members:
        - run_all
        - run_phase
        - loss_history
        - loss_histories

::: spectracles.optimise.opt_schedule.ManagedOptimiserSchedule
    options:
      members:
        - run_all
        - run_next_phase
        - run_phase_by_index
        - run_phases
        - skip_phase
        - reset
        - reset_from_phase
        - get_phase_status
        - is_complete
        - get_next_phase_index
        - get_completed_phases
        - get_pending_phases
        - loss_history
        - loss_histories

## Configuration

::: spectracles.optimise.opt_schedule.PhaseConfig

::: spectracles.optimise.opt_schedule.PhaseState
