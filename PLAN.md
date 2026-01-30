# User Experience Improvements Plan

## Development Environment

This project uses `uv` for dependency management. To set up:

```bash
# Sync the environment (uses Python 3.12+)
uv sync

# Run tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/test_leaf_sharing.py -v
```

**Note**: The project requires Python 3.12+ due to the `type` statement syntax in `path_utils.py`.

## Completed Features

All planned UX improvements have been implemented:

1. ✅ **Better `Shared` repr with parent path** - Shows `Shared → a.val` instead of memory address
2. ✅ **`get_sharing_summary()` method** - Returns dict mapping parent paths to shared child paths
3. ✅ **`get_parameter_paths(show_shared=False)` method** - Lists parameter paths for use with `set()`
4. ✅ **Clearer sub-component access error** - Helpful error message when calling sub-components with Shared values
5. ✅ **`validate_sharing()` method** - Validates sharing structure, returns diagnostic info
6. ✅ **`print_model_tree(show_sharing=True)` option** - Shows sharing relationships after tree
7. ✅ **Gradient diagnostics in OptimiserFrame** - `get_gradient_summary()` and `print_gradient_summary()`
8. ✅ **`get_shared_components()` method** - Detects module-level (branch) sharing for visualization

## Future Work (Deferred)

- **Python 3.10/3.11 support**: The `type LeafPath = ...` syntax in `path_utils.py` requires Python 3.12+. To support older versions, this could be replaced with `from typing import TypeAlias` and `LeafPath: TypeAlias = tuple[GetAttrKey, ...]`.
- **GitHub Actions CI**: Set up workflows to run tests on different Python versions/environments.
- **Static type checking**: Ensure mypy/pyright coverage is comprehensive across the codebase.
- **Docstrings**: Add docstrings for all user-facing functionality (public methods, classes, modules).
- **Alternative to `build_model()`**: Consider a class decorator (`@shareable`) or base class (`ShareableModule`) to avoid the need for `build_model(MyModel, ...)`. The decorator approach is cleanest - no inheritance required, opt-in per class. Main challenge is ensuring equinox's PyTree machinery still works.
- **Accept Parameter objects in `set()`**: Stretch goal - allow passing Parameter objects directly instead of just path strings.

## Documentation

- Update README.md with relevant usage examples as features are added
- Keep documentation minimal but useful
