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

## Future Work (Deferred)

- **Python 3.10/3.11 support**: The `type LeafPath = ...` syntax in `path_utils.py` requires Python 3.12+. To support older versions, this could be replaced with `from typing import TypeAlias` and `LeafPath: TypeAlias = tuple[GetAttrKey, ...]`.
- **GitHub Actions CI**: Set up workflows to run tests on different Python versions/environments.
- **Static type checking**: Ensure mypy/pyright coverage is comprehensive across the codebase.
- **Docstrings**: Add docstrings for all user-facing functionality (public methods, classes, modules).

## Scope

Improvements to make designing, implementing, and debugging models easier without reducing flexibility.

**In scope:**
1. Better `Shared` repr with parent path
2. `get_sharing_summary()` method on ShareModule
3. `get_parameter_paths(show_shared=False)` method on ShareModule
4. Clearer error when accessing sub-components with Shared values on unlocked models
5. Optional shape validation for shared parameters in ShareModule.__init__
6. `print_model_tree(show_sharing=True)` option
7. Gradient diagnostics method in OptimiserFrame

**Out of scope (stretch goal):**
- Accept Parameter objects directly in `set()` instead of just strings

## Implementation Order

### Phase 1: Core Introspection (share_module.py)

**1.1 Better `Shared` repr**
- Store `parent_path` in `Shared` class
- Update `__repr__` to show path instead of memory address
- Files: `share_module.py`
- Tests: Update `test_leaf_sharing.py::TestSharedClass`

**1.2 `get_sharing_summary()` method**
- Returns `dict[str, list[str]]` mapping parent paths to list of shared child paths
- Uses existing `_parent_leaf_paths` and `_dupl_leaf_paths`
- Files: `share_module.py`
- Tests: New test class in `test_leaf_sharing.py`

**1.3 `get_parameter_paths(show_shared=False)` method**
- Returns list of parameter path strings
- Default: only parent/unique parameters
- `show_shared=True`: includes shared (duplicate) paths
- Files: `share_module.py`
- Tests: New test class in `test_leaf_sharing.py`

### Phase 2: Error Handling & Validation

**2.1 Clearer sub-component access error**
- Modify `__getattr__` to detect when returned attribute contains `Shared` objects
- Provide helpful error message suggesting `get_locked_model()`
- Must not break existing functionality (only error on actual problematic access)
- Files: `share_module.py`
- Tests: New test in `test_leaf_sharing.py`

**2.2 Optional shape validation**
- Add `validate_shapes: bool = False` parameter to `ShareModule.__init__`
- Validate that shared parameters have compatible shapes
- Files: `share_module.py`
- Tests: New test class

### Phase 3: Visualization

**3.1 `print_model_tree(show_sharing=True)`**
- Add `show_sharing` parameter (default False for backwards compat)
- When True, annotate shared parameters with `→ parent_path`
- Files: `share_module.py`, `graph.py`
- Tests: Visual inspection + basic test

### Phase 4: Optimization Debugging

**4.1 Gradient diagnostics in OptimiserFrame**
- Add `get_gradient_summary(*loss_args, **loss_kwargs) -> dict[str, float]`
- Returns mapping of parameter paths to gradient norms
- Files: `opt_frame.py`
- Tests: `test_optimise.py`

## Test Strategy

- Add tests alongside each implementation
- Consider restructuring tests if needed for clarity
- Test both positive cases and edge cases
- Ensure backwards compatibility

## Documentation

- Update README.md with relevant usage examples as features are added
- Keep documentation minimal but useful

## Commit Strategy

Each phase should be a separate commit with clear message:
- `feat: improve Shared repr to show parent path`
- `feat: add get_sharing_summary() method`
- `feat: add get_parameter_paths() method`
- `feat: add clearer error for sub-component access on unlocked models`
- `feat: add optional shape validation for shared parameters`
- `feat: add show_sharing option to print_model_tree`
- `feat: add gradient diagnostics to OptimiserFrame`
