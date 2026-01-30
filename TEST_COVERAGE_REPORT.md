# Test Coverage Report

**Date:** January 2026
**Total Tests:** 218 passing
**Overall Coverage (excluding lvm_models):** 78%

## Summary

This report analyzes test coverage for the spectracles library, excluding the `lvm_models/` directory which is not targeted for testing.

## Coverage by Module

| Module | Statements | Missed | Coverage | Status |
|--------|-----------|--------|----------|--------|
| `__init__.py` | 14 | 2 | 86% | Good |
| `_version.py` | 13 | 0 | 100% | Complete |
| `model/data.py` | 17 | 0 | 100% | Complete |
| `model/formatting.py` | 103 | 0 | 100% | Complete |
| `model/graph.py` | 76 | 1 | 99% | Excellent |
| `model/io.py` | 28 | 3 | 89% | Good |
| `model/kernels.py` | 50 | 1 | 98% | Excellent |
| `model/parameter.py` | 154 | 6 | 96% | Excellent |
| `model/share_module.py` | 400 | 103 | 74% | Good |
| `model/spatial.py` | 122 | 27 | 78% | Good |
| `model/spectral.py` | 27 | 1 | 96% | Excellent |
| `optimise/opt_frame.py` | 113 | 12 | 89% | Good |
| `optimise/opt_schedule.py` | 161 | 77 | 52% | Moderate |
| `tree/path_utils.py` | 37 | 0 | 100% | Complete |

## Modules with Complete Coverage (100%)

### `model/data.py` - 100%
All data structures and conversion utilities are fully tested.

### `model/formatting.py` - 100%
All formatting utilities for Rich-based pretty printing are covered by smoke tests.

### `tree/path_utils.py` - 100%
Path utilities for PyTree traversal are thoroughly tested.

## Modules with Excellent Coverage (>90%)

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

### `model/share_module.py` - 74%
Core ShareModule functionality is tested including parameter sharing, validation, model building, and `fix_all()`/`free_all()` methods.

**Untested areas:**
- `print_graph()` visualization (lines 809-883)
- Some branches in sharing detection (lines 291-332)

## Modules Needing Attention (<70%)

### `optimise/opt_schedule.py` - 52%
The multi-phase optimization scheduler has moderate testing coverage.

**Tested areas:**
- PhaseConfig initialization and validation
- Phase creation
- OptimiserSchedule creation and execution
- Loss history tracking

**Untested areas:**
- `OptimiserScheduleUnsafe` class (experimental)
- Phase reset and skip functionality

## Test File Mapping

| Test File | Primary Module | Tests |
|-----------|---------------|-------|
| `test_data.py` | model/data.py | 8 |
| `test_formatting.py` | model/formatting.py | 27 |
| `test_graph.py` | model/graph.py | 13 |
| `test_io.py` | model/io.py | 8 |
| `test_kernels.py` | model/kernels.py | 16 |
| `test_leaf_sharing.py` | model/share_module.py, tree/path_utils.py | 52 |
| `test_opt_schedule.py` | optimise/opt_schedule.py | 15 |
| `test_optimise.py` | optimise/opt_frame.py | 13 |
| `test_parameter.py` | model/parameter.py | 36 |
| `test_spatial.py` | model/spatial.py | 22 |
| `test_spectral.py` | model/spectral.py | 8 |

## Coverage History

| Date | Tests | Coverage | Key Changes |
|------|-------|----------|-------------|
| Initial | 133 | 48% | Baseline |
| Update 1 | 183 | 70% | +formatting, +fix_all/free_all, +opt_schedule |
| Update 2 | 218 | 78% | +graph, +FourierBasis, +parameter repr/log |

## Recent Improvements

### Update 2
1. **`model/graph.py`**: 45% → 99% (+54%)
   - Added 13 tests for print_graph and layered_hierarchy_pos

2. **`model/parameter.py`**: 77% → 96% (+19%)
   - Added 5 tests for log parameterization
   - Added 11 tests for repr methods

3. **`model/spatial.py`**: 70% → 78% (+8%)
   - Added 6 tests for FourierBasis class

### Update 1
1. **`model/formatting.py`**: 36% → 100% (+64%)
2. **`model/share_module.py`**: 64% → 74% (+10%)
3. **`optimise/opt_schedule.py`**: 36% → 52% (+16%)
4. **`tree/path_utils.py`**: 97% → 100% (+3%)

## Recommendations

### Low Priority
1. **`optimise/opt_schedule.py`**: Test experimental `OptimiserScheduleUnsafe` class if it will be used.
2. **`model/share_module.py`**: Add tests for `print_graph()` visualization method.

## Excluded from Coverage

The following directories are intentionally excluded:
- `lvm_models/` - Domain-specific models not targeted for unit testing

## Running Coverage

To regenerate this report:

```bash
python -m pytest tests/ --cov=src/spectracles --cov-report=term-missing --cov-report=html
```

HTML report is available at `htmlcov/index.html`.
