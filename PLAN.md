# User Experience Improvements Plan

## Development Environment

This project uses `uv` for dependency management. To set up:

```bash
# Sync the environment (uses Python 3.10+)
uv sync

# Run tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/test_leaf_sharing.py -v
```

**Note**: The project requires Python 3.10+.

## Completed Features

### Phase 1: Core UX Improvements

1. ✅ **Better `Shared` repr with parent path** - Shows `Shared → a.val` instead of memory address
2. ✅ **`get_sharing_summary(level=...)` method** - Returns dict mapping parent paths to shared child paths
   - `level='component'`: Module-level sharing (default)
   - `level='parameter'`: Leaf-level parameter sharing
   - Deprecated `get_shared_components()` in favor of `get_sharing_summary(level='component')`
3. ✅ **`get_parameter_paths(show_shared, show_knowns)` method** - Lists parameter paths for use with `set()`
   - `show_shared=False`: Excludes shared/duplicate paths (default)
   - `show_knowns=False`: Excludes Known parameters (default)
4. ✅ **Clearer sub-component access error** - Helpful error message when calling sub-components with Shared values
5. ✅ **`validate_sharing()` method** - Validates sharing structure, returns diagnostic info
6. ✅ **`print_model_tree(show_sharing=True)` option** - Shows sharing relationships after tree with improved formatting
7. ✅ **Gradient diagnostics in OptimiserFrame** - `get_gradient_summary()` and `print_gradient_summary()`
8. ✅ **Python 3.10+ support** - Replaced `type` statement with TypeAlias, lowered networkx to >=3.4
9. ✅ **`parameter_summary()` method** - Rich table showing path, shape, bounds, status, sharing info
10. ✅ **Known parameter protections** - `set()` and `set_fixed_status()` block modifications by default
11. ✅ **`plot_model_graph(sharing_level=...)` option** - Visualize at component or parameter level
12. ✅ **`debug_repr()` fix** - Clean output without extra whitespace

### Phase 2: Schedule Builder API

1. ✅ **Parameter-centric schedule builder** - Declarative API to specify optimization schedules
   - `build_schedule()` function generates PhaseConfig objects from parameter specs
   - Helpers: `free_in()`, `free_after()`, `free_until()`, `fixed_in()`
   - Initialization: `init_normal()`, `init_value()`, `init_uniform()`
   - Glob-style pattern matching for parameter paths (`*`, `**`)
   - Validates shared parameter paths

2. ✅ **ManagedOptimiserSchedule** - Renamed from OptimiserScheduleUnsafe, fully tested
   - State tracking (PENDING, RUNNING, COMPLETED, SKIPPED)
   - `run_next_phase()`, `skip_phase()`, `reset()`, `reset_from_phase()`
   - Status inspection: `get_phase_status()`, `is_complete()`, etc.

3. ✅ **`fix_all()` / `free_all()` methods** - Freeze or unfreeze all parameters at once

### Phase 3: Documentation & CI

1. ✅ **MkDocs Material setup** - Dark/light mode toggle, code highlighting
2. ✅ **mkdocstrings integration** - Auto-generated API docs from docstrings
3. ✅ **Schedule Builder docs** - Guide, examples, and API reference
4. ✅ **Optimization API docs** - OptimiserSchedule, ManagedOptimiserSchedule, PhaseConfig
5. ✅ **GitHub Actions for docs** - Build and deploy docs on release
6. ✅ **GitHub Actions CI** - Run tests on Python 3.10, 3.11, 3.12 with coverage

## Test Coverage

- **Total tests:** 298 passing
- **Overall coverage:** 82% (excluding lvm_models)
- Key modules at 90%+: opt_schedule (97%), schedule_builder (92%), formatting (100%), graph (99%)

See `TEST_COVERAGE_REPORT.md` for details.

## Future Work (Deferred)

- **Static type checking**: Ensure mypy/pyright coverage is comprehensive across the codebase.
- **More docstrings**: Add docstrings for remaining user-facing functionality.
- **Alternative to `build_model()`**: Consider a class decorator (`@shareable`) or base class.
- **Accept Parameter objects in `set()`**: Allow passing Parameter objects directly.
- **Support lists/tuples/dicts of Parameters**: Currently ShareModule only supports attribute-based paths. See GitHub issue for details.

## Documentation

Docs are built with MkDocs Material. To preview locally:

```bash
pip install mkdocs-material mkdocstrings[python]
pip install -e .
mkdocs serve
```

Docs are auto-deployed to GitHub Pages on release.
