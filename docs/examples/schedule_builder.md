# Schedule Builder Examples

Practical examples of using the schedule builder API.

## Basic: Single Parameter

Optimize a simple model with one parameter:

```python
import spectracles as sp
import jax.numpy as jnp

# Build model
model = sp.build_model(MyModel, value=1.0)

# Loss function
def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

# Single phase, single parameter
schedule = sp.build_schedule(
    model, loss_fn,
    phases=[(100, 0.1)],
    params={
        "param": sp.free_in(0),
    },
)

schedule.run_all(x=x_data, y=y_data)
final_model = schedule.model_history[-1]
```

## Staged Optimization

Common pattern: warm up some parameters first, then refine others.

```python
schedule = sp.build_schedule(
    model, loss_fn,
    phases=[
        (200, 0.1),   # Phase 0: High LR warm-up
        (100, 0.01),  # Phase 1: Lower LR refinement
        (50, 0.001),  # Phase 2: Fine-tuning
    ],
    params={
        # Coefficients: optimize throughout
        "*.coefficients": sp.free_in(0, 1, 2),

        # Kernel parameters: freeze during warm-up
        "*.kernel.lengthscale": sp.free_after(1),
        "*.kernel.variance": sp.free_after(1),

        # Noise: only fine-tune at the end
        "noise": sp.free_in(2),
    },
)
```

## Reinitializing Parameters

Sometimes you want to reset a parameter partway through:

```python
import jax.random as jr

schedule = sp.build_schedule(
    model, loss_fn,
    phases=[
        (100, 0.1),  # Phase 0
        (100, 0.1),  # Phase 1: restart amplitude
        (50, 0.01),  # Phase 2
    ],
    params={
        "gp.coefficients": sp.free_in(0, 1, 2),

        # Reset amplitude to random values at phase 1
        "line.amplitude": sp.free_in(0, 1, 2) | sp.init_normal(1, mean=0.0, std=0.5),

        # Set width to specific value at phase 1
        "line.width": sp.free_after(1) | sp.init_value(1, 1.0),
    },
    key=jr.key(42),  # Required for init_normal
)
```

## Wildcard Patterns

Match multiple parameters with glob patterns:

```python
schedule = sp.build_schedule(
    model, loss_fn,
    phases=[(100, 0.1), (50, 0.01)],
    params={
        # All coefficients in any submodule
        "*.coefficients": sp.free_in(0, 1),

        # All kernel parameters (lengthscale, variance, etc.)
        "*.kernel.*": sp.free_after(1),

        # Any parameter named 'noise' at any depth
        "**.noise": sp.free_in(1),
    },
)
```

### Pattern Reference

| Pattern | Matches |
|---------|---------|
| `"param"` | Exact match |
| `"gp.kernel.lengthscale"` | Exact nested path |
| `"*.param"` | `inner1.param`, `inner2.param` |
| `"gp.*"` | `gp.coefficients`, `gp.kernel`, etc. |
| `"gp.*.lengthscale"` | `gp.kernel.lengthscale`, `gp.spatial.lengthscale` |
| `"**.lengthscale"` | Any `lengthscale` at any depth |

## Using Different Optimizers

Override the default optimizer per-phase:

```python
import optax

schedule = sp.build_schedule(
    model, loss_fn,
    phases=[
        (100, 0.1),                      # Uses default (adam)
        (50, 0.01, optax.sgd(0.01)),     # Explicit SGD
        (50, 0.001, optax.adamw(0.001)), # AdamW with weight decay
    ],
    params={
        "*.coefficients": sp.free_in(0, 1, 2),
    },
    default_optimizer="adam",  # Default for phases without explicit optimizer
)
```

## Managed Schedule (Interactive)

Use `ManagedOptimiserSchedule` for more control:

```python
schedule = sp.build_schedule(
    model, loss_fn,
    phases=[
        (100, 0.1),
        (50, 0.01),
        (50, 0.001),
    ],
    params={
        "*.coefficients": sp.free_in(0, 1, 2),
        "*.kernel.*": sp.free_after(1),
    },
    managed=True,  # Returns ManagedOptimiserSchedule
)

# Run phases one at a time
schedule.run_next_phase(x=x_data, y=y_data)
print(f"Phase 0 done, loss: {schedule.loss_history[-1]:.4f}")

# Check status
print(schedule.get_phase_status())

# Skip a phase if needed
schedule.skip_phase(1)

# Run remaining
schedule.run_all(x=x_data, y=y_data)

# Reset and try again with different data
schedule.reset()
schedule.run_all(x=new_x, y=new_y)
```

## Shared Parameters

When parameters are shared, only specify the parent path:

```python
# If model has sharing: line_2.amplitude -> line_1.amplitude
# Only specify the parent path

schedule = sp.build_schedule(
    model, loss_fn,
    phases=[(100, 0.1)],
    params={
        "line_1.amplitude": sp.free_in(0),  # Parent path
        # "line_2.amplitude": ...           # Don't specify - it's shared!
    },
)

# Check available paths if unsure:
print(model.get_parameter_paths(show_shared=False))  # Parent paths only
print(model.get_parameter_paths(show_shared=True))   # All paths
```

## Full Example: Spectral Line Fitting

```python
import spectracles as sp
import jax.numpy as jnp
import jax.random as jr

# Build a spectral-spatial model
model = sp.build_model(
    MySpectralModel,
    n_spatial=64,
    n_spectral=100,
)

def loss_fn(model, data):
    pred = model(data)
    return jnp.mean((pred - data.flux) ** 2)

# Three-phase optimization strategy
schedule = sp.build_schedule(
    model, loss_fn,
    phases=[
        (500, 0.05),   # Phase 0: Establish spatial structure
        (200, 0.01),   # Phase 1: Refine line parameters
        (100, 0.001),  # Phase 2: Polish everything
    ],
    params={
        # Spatial coefficients: always free
        "spatial.*.coefficients": sp.free_in(0, 1, 2),

        # GP kernel: freeze initially, then optimize
        "spatial.*.kernel.*": sp.free_after(1),

        # Line amplitudes: reinitialize at phase 1
        "lines.*.amplitude": sp.free_in(0, 1, 2) | sp.init_normal(1, std=0.1),

        # Line positions: fix until final polish
        "lines.*.center": sp.free_in(2),

        # Line widths: optimize from phase 1
        "lines.*.width": sp.free_after(1) | sp.init_value(1, 1.0),

        # Continuum: always free
        "continuum.*": sp.free_in(0, 1, 2),
    },
    key=jr.key(0),
)

# Run optimization
schedule.run_all(data=my_data)

# Get result
final_model = schedule.model_history[-1]
```
