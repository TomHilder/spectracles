# Schedule Builder API

A declarative, parameter-centric API for building optimization schedules with minimal boilerplate.

## Quick Start

```python
from spectracles import build_schedule, free_in, free_after, init_normal
import jax.random as jr

schedule = build_schedule(
    model, loss_fn,
    phases=[
        (100, 0.1),   # 100 steps, lr=0.1
        (50, 0.01),   # 50 steps, lr=0.01
    ],
    params={
        "gp.coefficients": free_in(0, 1),
        "gp.kernel.*": free_after(1),
        "line.amplitude": free_in(1) | init_normal(0, std=0.1),
    },
    key=jr.key(42),
)

schedule.run_all(x=data_x, y=data_y)
```

## Specifying Free/Fixed Status

### `free_in(*phases)`
Parameter is free (optimized) only in the specified phases.

```python
"param": free_in(0, 2, 4)  # Free in phases 0, 2, and 4
```

### `free_after(phase)`
Parameter is free from the specified phase onwards.

```python
"param": free_after(1)  # Fixed in phase 0, free in phases 1, 2, 3, ...
```

### `free_until(phase)`
Parameter is free up to and including the specified phase.

```python
"param": free_until(1)  # Free in phases 0 and 1, fixed afterwards
```

### `fixed_in(*phases)`
Parameter is fixed only in the specified phases, free elsewhere.

```python
"param": fixed_in(0)  # Fixed in phase 0, free in all other phases
```

## Initialization Helpers

Initialize parameters at the start of a phase.

### `init_value(phase, value)`
Set parameter to a specific value.

```python
"param": free_after(1) | init_value(1, 2.5)  # Set to 2.5 at start of phase 1
```

### `init_normal(phase, mean=0.0, std=1.0)`
Initialize with random normal values. Requires passing `key` to `build_schedule`.

```python
"param": free_in(0) | init_normal(0, mean=0.0, std=0.1)
```

### `init_uniform(phase, low=0.0, high=1.0)`
Initialize with random uniform values. Requires passing `key` to `build_schedule`.

```python
"param": free_in(0) | init_uniform(0, low=-1.0, high=1.0)
```

## Combining Specs

Use `|` to combine free/fixed specs with initialization:

```python
"param": free_in(0, 1, 2) | init_normal(0) | init_value(2, 1.0)
```

## Pattern Matching

Glob-style patterns match parameter paths:

| Pattern | Matches |
|---------|---------|
| `"param"` | Exact match |
| `"inner.param"` | Exact nested match |
| `"*.param"` | `inner1.param`, `inner2.param`, etc. |
| `"gp.kernel.*"` | All attributes of `gp.kernel` |
| `"gp.*.lengthscale"` | `gp.kernel.lengthscale`, `gp.mean.lengthscale`, etc. |
| `"**.lengthscale"` | Any `lengthscale` at any depth |

## Phase Specification

Phases are tuples of `(n_steps, learning_rate)` or `(n_steps, learning_rate, optimizer)`:

```python
phases=[
    (100, 0.1),                    # Uses default optimizer (adam)
    (50, 0.01),
    (50, 0.001, optax.sgd(0.001)), # Custom optimizer
]
```

## Full API

```python
schedule = build_schedule(
    model,                          # ShareModule to optimize
    loss_fn,                        # Loss function
    phases,                         # List of phase specs
    params,                         # Dict of pattern -> ParamSpec
    default_optimizer="adam",       # "adam" or "sgd"
    managed=False,                  # Return ManagedOptimiserSchedule if True
    key=None,                       # PRNG key for random initialization
)
```

## Shared Parameters

When parameters are shared, only specify the **parent** path (not the shared copies). The builder validates this and will error if you try to specify a shared path directly.

```python
# If 'b' is shared from 'a':
params={
    "a": free_in(0),  # Correct - use parent path
    # "b": free_in(0),  # Error - 'b' is a shared path
}
```

## Example: Multi-Phase Optimization

```python
schedule = build_schedule(
    model, loss_fn,
    phases=[
        (200, 0.1),    # Phase 0: Warm up coefficients
        (100, 0.05),   # Phase 1: Refine with kernel
        (50, 0.01),    # Phase 2: Fine tune everything
    ],
    params={
        # Coefficients: always free
        "*.coefficients": free_in(0, 1, 2),

        # Kernel params: free after warm-up
        "*.kernel.*": free_after(1),

        # Amplitude: reinitialize at phase 1
        "line.amplitude": free_after(1) | init_normal(1, std=0.5),
    },
    key=jr.key(0),
)

schedule.run_all(x=x, y=y)
final_model = schedule.model_history[-1]
```
