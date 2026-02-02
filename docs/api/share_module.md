# ShareModule

`ShareModule` is a wrapper around JAX/Equinox models that enables parameter sharing and provides utilities for model inspection and manipulation.

## Building Models

Use `build_model()` to wrap your Equinox model in a ShareModule:

```python
import spectracles as sp

# Your Equinox model with shared components
model = MyModel(shared_component=inner, ...)

# Wrap in ShareModule to enable sharing utilities
wrapped = sp.build_model(model)
```

## Inspecting the Model

### Parameter Paths

```python
# Get all unique parameter paths (excludes shared duplicates and Known)
paths = model.get_parameter_paths()

# Include shared parameters
paths = model.get_parameter_paths(show_shared=True)

# Include Known parameters
paths = model.get_parameter_paths(show_knowns=True)
```

### Parameter Summary

```python
# Print a rich table of all parameters
model.get_parameter_summary()
```

Shows path, shape, bounds, free/fixed status, and sharing info for each parameter.

### Sharing Summary

```python
# Component-level: which modules are the same object
model.get_sharing_summary(level="component")
# Returns: {'line_1.v': ['line_2.v']}

# Parameter-level: which parameter values are shared
model.get_sharing_summary(level="parameter")
# Returns: {'line_1.v.coefficients': ['line_2.v.coefficients']}
```

### Model Tree

```python
# Print the model structure
model.print_model_tree()

# Include sharing relationships
model.print_model_tree(show_sharing=True)
```

### Model Graph

```python
# Plot the model as a directed graph
model.plot_model_graph()

# Component-level view (shared modules point to same node)
model.plot_model_graph(sharing_level="component")

# Parameter-level view (shows all paths)
model.plot_model_graph(sharing_level="parameter")
```

## Modifying Parameters

### Setting Values

```python
# Set a single parameter
model = model.set("line_1.A.coefficients", new_value)

# Set multiple parameters
model = model.set(
    ["path.to.param1", "path.to.param2"],
    [value1, value2]
)
```

### Fixing/Freeing Parameters

```python
# Fix parameters (won't be optimized)
model = model.set_fixed_status(["path.to.param"], [True])

# Free parameters
model = model.set_fixed_status(["path.to.param"], [False])

# Fix all parameters
model = model.fix_all()

# Free all parameters
model = model.free_all()
```

## Validation

```python
# Check sharing structure is valid
info = model.validate_sharing()
```

## API Reference

::: spectracles.ShareModule
    options:
      members:
        - get_parameter_paths
        - get_parameter_summary
        - get_sharing_summary
        - print_model_tree
        - plot_model_graph
        - set
        - set_fixed_status
        - fix_all
        - free_all
        - validate_sharing
        - copy
        - get_locked_model
        - debug_repr

::: spectracles.build_model
