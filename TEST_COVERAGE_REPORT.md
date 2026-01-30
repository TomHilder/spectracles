# Test Coverage Report

**Date:** January 2026
**Total Tests:** 133 passing
**Overall Coverage (excluding lvm_models):** 64%

## Summary

This report analyzes test coverage for the spectracles library, excluding the `lvm_models/` directory which is not targeted for testing.

## Coverage by Module

| Module | Statements | Missed | Coverage | Status |
|--------|-----------|--------|----------|--------|
| `__init__.py` | 14 | 2 | 86% | Good |
| `_version.py` | 13 | 0 | 100% | Complete |
| `model/data.py` | 17 | 0 | 100% | Complete |
| `model/formatting.py` | 103 | 66 | 36% | Needs Work |
| `model/graph.py` | 76 | 42 | 45% | Needs Work |
| `model/io.py` | 28 | 3 | 89% | Good |
| `model/kernels.py` | 50 | 1 | 98% | Excellent |
| `model/parameter.py` | 154 | 36 | 77% | Good |
| `model/share_module.py` | 400 | 143 | 64% | Moderate |
| `model/spatial.py` | 122 | 36 | 70% | Moderate |
| `model/spectral.py` | 27 | 1 | 96% | Excellent |
| `optimise/opt_frame.py` | 113 | 13 | 88% | Good |
| `optimise/opt_schedule.py` | 161 | 103 | 36% | Needs Work |
| `tree/path_utils.py` | 37 | 1 | 97% | Excellent |

## Modules with Excellent Coverage (>90%)

### `model/data.py` - 100%
All data structures and conversion utilities are fully tested.

### `model/kernels.py` - 98%
Kernel implementations (Matern12, Matern32, Matern52, SquaredExponential) are well tested. Only missing coverage is one branch in the kernel normalization.

### `tree/path_utils.py` - 97%
Path utilities for PyTree traversal are thoroughly tested.

### `model/spectral.py` - 96%
Spectral models (Constant, Gaussian) are well covered.

## Modules with Good Coverage (70-90%)

### `optimise/opt_frame.py` - 88%
The OptimiserFrame class is well tested including:
- Initialization with valid/invalid inputs
- Optimization with shared parameters
- Gradient diagnostics

**Untested areas:**
- Early convergence exit path (lines 140-149)
- Some edge cases in gradient summary

### `model/io.py` - 89%
Save/load functionality is well tested.

**Untested areas:**
- Error handling when file already exists without overwrite flag

### `model/parameter.py` - 77%
Parameter and ConstrainedParameter classes are tested for core functionality.

**Untested areas:**
- `__repr__` methods with Rich formatting (lines 56-71)
- Log parameterization branches (lines 108-114, 140, 150)
- Some edge cases in inverse transforms

### `model/spatial.py` - 70%
Spatial models are tested for basic operations.

**Untested areas:**
- FourierBasis class (lines 88-105)
- Some branches in conjugate symmetry handling
- Error handling in dimension checks

### `model/share_module.py` - 64%
Core ShareModule functionality is tested including parameter sharing, validation, and model building.

**Untested areas:**
- `fix_all()` and `free_all()` methods (lines 415-427)
- `print_graph()` visualization (lines 809-883)
- Some branches in sharing detection (lines 291-332)
- Error paths in sharing validation

## Modules Needing Attention (<70%)

### `model/formatting.py` - 36%
This is a new module for Rich-based pretty printing. Many formatting functions are untested.

**Recommendation:** The formatting module is primarily for display purposes. Direct tests may have limited value, but integration tests through repr methods would help.

### `model/graph.py` - 45%
Graph visualization utilities have limited direct testing.

**Untested areas:**
- `print_graph()` function
- `layered_hierarchy_pos()` layout algorithm
- Matplotlib integration

**Recommendation:** These are visualization utilities. Consider adding smoke tests to ensure they don't crash.

### `optimise/opt_schedule.py` - 36%
The multi-phase optimization scheduler has minimal testing.

**Untested areas:**
- `PhaseConfig` validation
- `Phase` class methods
- `OptimiserSchedule` orchestration
- Parameter fixing/unfixing between phases

**Recommendation:** This module needs comprehensive tests for multi-phase optimization workflows.

## Test File Mapping

| Test File | Primary Module | Tests |
|-----------|---------------|-------|
| `test_data.py` | model/data.py | 8 |
| `test_io.py` | model/io.py | 8 |
| `test_kernels.py` | model/kernels.py | 16 |
| `test_leaf_sharing.py` | model/share_module.py, tree/path_utils.py | 44 |
| `test_optimise.py` | optimise/opt_frame.py | 13 |
| `test_parameter.py` | model/parameter.py | 20 |
| `test_spatial.py` | model/spatial.py | 16 |
| `test_spectral.py` | model/spectral.py | 8 |

## Recommendations

### High Priority
1. **`optimise/opt_schedule.py`**: Add tests for multi-phase optimization schedules. This is core functionality that needs coverage.
2. **`model/share_module.py`**: Add tests for `fix_all()`, `free_all()`, and visualization methods.

### Medium Priority
3. **`model/spatial.py`**: Add tests for FourierBasis class and edge cases in dimension handling.
4. **`model/parameter.py`**: Add tests for log parameterization and edge cases in bounded transforms.

### Low Priority
5. **`model/formatting.py`**: Consider adding smoke tests for formatting utilities.
6. **`model/graph.py`**: Consider adding smoke tests for graph visualization.

## Excluded from Coverage

The following directories are intentionally excluded:
- `lvm_models/` - Domain-specific models not targeted for unit testing

## Running Coverage

To regenerate this report:

```bash
python -m pytest tests/ --cov=src/spectracles --cov-report=term-missing --cov-report=html
```

HTML report is available at `htmlcov/index.html`.
