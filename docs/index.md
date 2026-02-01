# spectracles

Spectrospatial modelling with JAX and Equinox.

## Installation

```bash
pip install spectracles
```

## Quick Start

```python
import spectracles as sp

# Build a model
model = sp.build_model(MyModel, ...)

# Create an optimization schedule
schedule = sp.build_schedule(
    model, loss_fn,
    phases=[(100, 0.1), (50, 0.01)],
    params={
        "*.coefficients": sp.free_in(0, 1),
        "*.kernel.*": sp.free_after(1),
    },
)

# Run optimization
schedule.run_all(x=data_x, y=data_y)
```
