# Test Coverage Report

**Date:** February 2026
**Total Tests:** 298 passing
**Overall Coverage (excluding lvm_models):** 82%

## Summary

This report analyzes test coverage for the spectracles library, excluding the `lvm_models/` directory which is not targeted for testing.

## Coverage by Module

| Module | Statements | Missed | Coverage | Status |
|--------|-----------|--------|----------|--------|
| `__init__.py` | 15 | 2 | 87% | Good |
| `_version.py` | 13 | 0 | 100% | Complete |
| `model/data.py` | 17 | 0 | 100% | Complete |
| `model/formatting.py` | 103 | 0 | 100% | Complete |
| `model/graph.py` | 78 | 1 | 99% | Excellent |
| `model/io.py` | 28 | 3 | 89% | Good |
| `model/kernels.py` | 50 | 1 | 98% | Excellent |
| `model/parameter.py` | 154 | 6 | 96% | Excellent |
| `model/share_module.py` | 532 | 131 | 75% | Good |
| `model/spatial.py` | 122 | 27 | 78% | Good |
| `model/spectral.py` | 27 | 1 | 96% | Excellent |
| `optimise/opt_frame.py` | 113 | 12 | 89% | Good |
| `optimise/opt_schedule.py` | 161 | 5 | 97% | Excellent |
| `optimise/schedule_builder.py` | 168 | 14 | 92% | Excellent |
| `tree/path_utils.py` | 37 | 0 | 100% | Complete |

## Modules with Complete Coverage (100%)

### `model/data.py` - 100%
All data structures and conversion utilities are fully tested.

### `model/formatting.py` - 100%
All formatting utilities for Rich-based pretty printing are covered by smoke tests.

### `tree/path_utils.py` - 100%
Path utilities for PyTree traversal are thoroughly tested.

## Modules with Excellent Coverage (>90%)

### `optimise/opt_schedule.py` - 97%
Both `OptimiserSchedule` and `ManagedOptimiserSchedule` are thoroughly tested including:
- Phase state management (PENDING, RUNNING, COMPLETED, SKIPPED)
- Sequential execution with `run_all()`, `run_next_phase()`, `run_phase_by_index()`
- Skip and reset functionality
- Status inspection methods
- Loss history tracking

### `optimise/schedule_builder.py` - 92%
The parameter-centric schedule builder API is well tested including:
- Helper functions (`free_in`, `free_after`, `free_until`, `fixed_in`)
- Initialization helpers (`init_normal`, `init_value`, `init_uniform`)
- Pattern matching with wildcards (`*`, `**`)
- Shared parameter validation
- Full schedule building and execution

### `model/graph.py` - 99%
Graph visualization utilities including `print_graph()` and `layered_hierarchy_pos()` are well tested.

### `model/kernels.py` - 98%
Kernel implementations (Matern12, Matern32, Matern52, SquaredExponential) are well tested.

### `model/parameter.py` - 96%
Parameter classes including repr methods and log parameterization are thoroughly tested.

### `model/spectral.py` - 96%
Spectral models (Constant, Gaussian) are well covered.

## Modules with Good Coverage (70-90%)

### `optimise/opt_frame.py` - 89%
The OptimiserFrame class is well tested including:
- Initialization with valid/invalid inputs
- Optimization with shared parameters
- Gradient diagnostics

**Untested areas:**
- Early convergence exit path (lines 141-149)

### `model/io.py` - 89%
Save/load functionality is well tested.

### `model/spatial.py` - 78%
Spatial models including FourierGP, FourierBasis, and PerSpaxel are tested.

**Untested areas:**
- Some branches in conjugate symmetry handling
- Edge cases in dimension checks

### `model/share_module.py` - 75%
Core ShareModule functionality is tested including parameter sharing, validation, model building, `fix_all()`/`free_all()` methods, and Known parameter handling.

**Untested areas:**
- `plot_model_graph()` visualization
- Some branches in `get_digraph()` graph building
- `debug_repr()` formatting

## Test File Mapping

| Test File | Primary Module | Tests |
|-----------|---------------|-------|
| `test_data.py` | model/data.py | 8 |
| `test_formatting.py` | model/formatting.py | 27 |
| `test_graph.py` | model/graph.py | 13 |
| `test_io.py` | model/io.py | 8 |
| `test_kernels.py` | model/kernels.py | 16 |
| `test_leaf_sharing.py` | model/share_module.py, tree/path_utils.py | 63 |
| `test_opt_schedule.py` | optimise/opt_schedule.py | 37 |
| `test_optimise.py` | optimise/opt_frame.py | 13 |
| `test_parameter.py` | model/parameter.py | 36 |
| `test_schedule_builder.py` | optimise/schedule_builder.py | 45 |
| `test_spatial.py` | model/spatial.py | 22 |
| `test_spectral.py` | model/spectral.py | 8 |

## Coverage History

| Date | Tests | Coverage | Key Changes |
|------|-------|----------|-------------|
| Initial | 133 | 48% | Baseline |
| Update 1 | 183 | 70% | +formatting, +fix_all/free_all, +opt_schedule |
| Update 2 | 218 | 78% | +graph, +FourierBasis, +parameter repr/log |
| Update 3 | 285 | 82% | +ManagedOptimiserSchedule (97%), +schedule_builder (92%) |
| Update 4 | 298 | 82% | +Known parameter handling, +sharing levels, +parameter_summary |

## Recent Improvements

### Update 4 (Current)
1. **`model/share_module.py`**: Added 132 statements for new features
   - `parameter_summary()` method with Rich table output
   - `get_sharing_summary(level='component'|'parameter')` consolidated API
   - `get_parameter_paths(show_knowns=False)` excludes Known by default
   - `set(allow_set_knowns=False)` protects Known parameters
   - `set_fixed_status()` blocks unfixing Known parameters
   - `plot_model_graph(sharing_level=...)` option
   - Improved `print_model_tree` sharing display format
   - `debug_repr()` fix for cleaner output

2. **`test_leaf_sharing.py`**: 52 → 63 tests (+11)
   - Added `TestKnownParameterHandling` test class with 11 tests
   - Covers `allow_set_knowns`, `show_knowns`, Known unfixing protection

### Update 3
1. **`optimise/opt_schedule.py`**: 52% → 97% (+45%)
   - Renamed OptimiserScheduleUnsafe to ManagedOptimiserSchedule
   - Added 22 comprehensive tests for state management, skip/reset, status inspection
   - Added docstrings to all classes

2. **`optimise/schedule_builder.py`**: New module at 92%
   - Parameter-centric API for building schedules
   - 45 tests covering helpers, pattern matching, building, and shared validation

### Update 2
1. **`model/graph.py`**: 45% → 99% (+54%)
2. **`model/parameter.py`**: 77% → 96% (+19%)
3. **`model/spatial.py`**: 70% → 78% (+8%)

### Update 1
1. **`model/formatting.py`**: 36% → 100% (+64%)
2. **`model/share_module.py`**: 64% → 74% (+10%)
3. **`optimise/opt_schedule.py`**: 36% → 52% (+16%)

## Excluded from Coverage

The following directories are intentionally excluded:
- `lvm_models/` - Domain-specific models not targeted for unit testing

## Running Coverage

To regenerate this report:

```bash
python -m pytest tests/ --cov=src/spectracles --cov-report=term-missing --cov-report=html
```

HTML report is available at `htmlcov/index.html`.
