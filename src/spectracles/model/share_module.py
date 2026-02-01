"""share_module.py - Models built on equinox and jax that can have parameters that are shared between leaves of the model tree. Module refers to equinox Modules not Python modules."""

from typing import Any, Callable, Dict

from typing_extensions import Self

import matplotlib.pyplot as plt
from equinox import Module, filter, is_inexact_array, tree_at
from jax.tree import leaves_with_path
from jax.tree_util import tree_map
from jaxtyping import Array, PyTree
from matplotlib.axes import Axes
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels, draw_networkx_nodes

from spectracles.model.graph import (
    DEFAULT_NX_KWDS,
    layered_hierarchy_pos,
    print_graph,
    temporarily_disable_tex,
)
from spectracles.model.parameter import AnyParameter, is_constrained, is_parameter
from spectracles.tree.path_utils import (
    GetAttrKey,
    LeafPath,
    get_duplicated_leaves,
    leafpath_to_str,
    str_to_leafpath,
    use_path_get_leaf,
    use_paths_get_leaves,
)


def id_from_path(dictionary, value):
    return next((key for key, val in dictionary.items() if val == value), None)


def get_indices(lst, target):
    return [i for i, x in enumerate(lst) if x == target]


def get_duplicated_parameters(
    tree: PyTree,
) -> tuple[list[int], list[LeafPath], dict[int, LeafPath]]:
    filter_specs = [
        tree_map(
            is_parameter,
            tree,
            is_leaf=is_parameter,
        ),
        tree_map(lambda x: is_inexact_array(x), tree),
    ]
    return get_duplicated_leaves(tree, filter_specs)


def _detect_shared_modules(model: Module) -> dict[str, list[str]]:
    """Detect which Module-level components are shared (same object identity).

    This detects when entire sub-modules are the same Python object, which
    is different from parameter-level sharing (where different modules may
    share the same Parameter object).

    Args:
        model: The model to analyze (BEFORE ShareModule processing).

    Returns:
        Dictionary mapping parent paths to list of duplicate paths.
        E.g., {"branch_a": ["branch_b"]} means branch_b is the same object as branch_a.
    """
    seen_ids: dict[int, str] = {}
    shared: dict[str, list[str]] = {}

    def traverse(obj: Any, path: str) -> None:
        if not isinstance(obj, Module):
            return

        obj_id = id(obj)

        # If we've seen this object before, it's a shared component
        if obj_id in seen_ids:
            parent_path = seen_ids[obj_id]
            if parent_path not in shared:
                shared[parent_path] = []
            shared[parent_path].append(path)
            return  # Don't traverse children of duplicate modules

        # Record this object
        seen_ids[obj_id] = path

        # Traverse child modules
        # Get all public attributes that are Modules
        for attr_name in vars(obj).keys():
            if attr_name.startswith("_"):
                continue
            try:
                child = getattr(obj, attr_name)
                if isinstance(child, Module):
                    child_path = f"{path}.{attr_name}" if path else attr_name
                    traverse(child, child_path)
            except (AttributeError, TypeError):
                pass

    traverse(model, "")
    return shared


class Shared:
    """A sentinel object used to indicate a parameter is shared.

    Attributes:
        id: The Python object id of the original (parent) parameter value.
        parent_path: String representation of the path to the parent parameter.
    """

    id: int
    parent_path: str

    def __init__(self, id: int, parent_path: str = ""):
        self.id = id
        self.parent_path = parent_path

    def __repr__(self) -> str:
        from spectracles.model.formatting import shared_repr_parts, render_to_string

        if self.parent_path:
            text = shared_repr_parts(self.parent_path)
            return render_to_string(text)
        return f"Shared({self.id})"

    def __str__(self) -> str:
        return self.__repr__()


def is_shared(x: Any) -> bool:
    return isinstance(x, Shared)


def contains_shared(obj: Any) -> bool:
    """
    Check if an object (e.g., a model sub-component) contains any Shared sentinel values.

    This is useful for checking whether a sub-component of an unlocked ShareModule
    can be called directly. Sub-components containing Shared values cannot be called
    because the Shared sentinels are not valid JAX values.

    Args:
        obj: Any object to check (typically an equinox Module).

    Returns:
        True if the object contains any Shared values, False otherwise.

    Example:
        >>> model = build_model(MyModel, ...)
        >>> if contains_shared(model.line_1):
        ...     # Use locked model for this sub-component
        ...     locked = model.get_locked_model()
        ...     result = locked.line_1(data)
        ... else:
        ...     result = model.line_1(data)
    """
    found_shared = [False]

    def check_leaf(x):
        if is_shared(x):
            found_shared[0] = True
        return x

    tree_map(check_leaf, obj, is_leaf=is_shared)
    return found_shared[0]


class _SharedSubcomponentProxy:
    """Proxy for sub-components that contain Shared values.

    Delegates attribute access to the underlying object, but raises a helpful
    error when the user tries to call the sub-component.
    """

    def __init__(self, obj: Any, attr_name: str):
        # Use object.__setattr__ to avoid triggering our __setattr__
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_attr_name", attr_name)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot call '{self._attr_name}' directly - it contains shared parameters "
            f"that have been replaced with Shared sentinel objects.\n\n"
            f"To call sub-components of the model, use get_locked_model() first:\n"
            f"    locked_model = model.get_locked_model()\n"
            f"    result = locked_model.{self._attr_name}(...)\n\n"
            f"Note: Locked models cannot be optimised, so get_locked_model() is "
            f"intended for inference/prediction only."
        )

    def __repr__(self):
        return repr(self._obj)

    def __getattr__(self, name):
        # Delegate attribute access to the underlying object
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        # Delegate attribute setting to the underlying object
        setattr(self._obj, name, value)


class ShareModule(Module):
    model: Module

    # Sharing metadata
    _dupl_leaf_ids: list[int]
    _dupl_leaf_paths: list[LeafPath]
    _parent_leaf_paths: Dict[int, LeafPath]
    _shared_components: dict[str, list[str]]

    # Is this instance locked?
    _locked: bool = False

    # Keep track of attributes to avoid recursion
    _attr_names = {"model", "_dupl_leaf_ids", "_dupl_leaf_paths", "_parent_leaf_paths", "_shared_components", "_locked"}

    def __init__(self, model: Module, locked: bool = False):
        # Detect module-level sharing BEFORE any tree manipulation
        # (tree_at operations can break object identity)
        self._shared_components = _detect_shared_modules(model)

        # Save the parameter-level sharing info
        (
            self._dupl_leaf_ids,
            self._dupl_leaf_paths,
            self._parent_leaf_paths,
        ) = get_duplicated_parameters(model)
        # Other metadata
        self._locked = locked

        # Remove leaves that are coupled to other leaves
        def replace_fn(leaf):
            leaf_id = id(leaf)
            parent_path = self._parent_leaf_paths.get(leaf_id)
            if parent_path is not None:
                parent_path_str = leafpath_to_str(parent_path)
            else:
                parent_path_str = ""
            return Shared(leaf_id, parent_path_str)

        # If locked, we don't want Shared() objects because all sub-models need to be callable
        # and if we replace some leaves with Shared() objects, they won't be
        if locked:
            self.model = model
        # Otherwise, replace the leaves with Shared objects
        else:
            self.model = tree_at(self._where, model, replace_fn=replace_fn)

    def __getattr__(self, name):
        # Use the class attribute instead of instance attribute
        if name in self._attr_names or name.startswith("__"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")

        # Safe delegation to model
        if self.model is None:
            raise AttributeError(f"The model attribute is None, cannot access {name}")

        try:
            attr = getattr(self.model, name)
        except AttributeError:
            raise AttributeError(
                f"Neither {type(self).__name__} nor {type(self.model).__name__} has attribute {name}"
            )

        # If the model is not locked and the attribute is callable and contains
        # Shared values, wrap it in a proxy that gives a helpful error on __call__
        if not self._locked and callable(attr) and contains_shared(attr):
            return _SharedSubcomponentProxy(attr, name)

        return attr

    def __getstate__(self):
        # Make sure we don't include any computed properties that might cause recursion
        return {
            "model": self.model,
            "_dupl_leaf_ids": self._dupl_leaf_ids,
            "_dupl_leaf_paths": self._dupl_leaf_paths,
            "_parent_leaf_paths": self._parent_leaf_paths,
            "_shared_components": self._shared_components,
            "_locked": self._locked,
        }

    def __setstate__(self, state):
        # When dealing with frozen instances, we need to use object.__setattr__
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        from spectracles.model.formatting import (
            create_model_tree,
            add_tree_node,
            render_to_string,
            styled_type,
            styled_free,
            styled_fixed,
        )
        from rich.text import Text

        # Count parameters
        n_unique = len(self._parent_leaf_paths)
        n_shared = len(self._dupl_leaf_ids)

        # Build header
        header = Text()
        header.append_text(styled_type("ShareModule"))
        header.append(" (", style="dim")
        header.append(f"{n_unique} params", style="value")
        if n_shared > 0:
            header.append(", ", style="dim")
            header.append(f"{n_shared} shared", style="shared")
        header.append(", ", style="dim")
        if self._locked:
            header.append_text(styled_fixed())
        else:
            header.append_text(styled_free())
        header.append(")", style="dim")

        # Create tree
        tree = create_model_tree("model", type(self.model).__name__)

        # Add model attributes that are parameters or modules
        self._add_model_attrs_to_tree(tree, self.model, "")

        # Combine header and tree
        result = Text()
        result.append_text(header)
        result.append("\n")
        result.append(render_to_string(tree))

        return render_to_string(result)

    def _add_model_attrs_to_tree(self, tree, obj, path_prefix: str) -> None:
        """Recursively add model attributes to the tree."""
        from spectracles.model.formatting import add_tree_node
        from rich.text import Text

        for attr_name in vars(obj).keys():
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(obj, attr_name)
            except (AttributeError, TypeError):
                continue

            full_path = f"{path_prefix}.{attr_name}" if path_prefix else attr_name

            # Check if it's a parameter
            if is_parameter(attr):
                # Get the repr of the parameter (already styled)
                node = tree.add(Text(f"{attr_name}: ", style="value") + Text(repr(attr)))
            # Check if it's a nested Module (but not a parameter)
            elif isinstance(attr, Module) and not is_parameter(attr):
                # Create a subtree for nested modules
                label = Text()
                label.append(attr_name, style="value")
                label.append(" (", style="dim")
                label.append(type(attr).__name__, style="type")
                label.append(")", style="dim")
                subtree = tree.add(label)
                self._add_model_attrs_to_tree(subtree, attr, full_path)

    def debug_repr(self) -> str:
        """Return the full equinox-style repr for debugging."""
        import equinox as eqx
        return eqx.tree_pformat(self)

    def __call__(self, *args, **kwargs) -> Any:
        # Replace nodes specified by `where` with the nodes specified by `get`
        # This places the deleted nodes back in the tree before calling the model
        restored_model = tree_at(self._where, self.model, self._get(self.model))
        return restored_model(*args, **kwargs)

    def _where(self, model) -> list[Any]:
        return use_paths_get_leaves(
            model,
            self._dupl_leaf_paths,
        )

    def _get(self, model) -> list[Any]:
        return use_paths_get_leaves(
            model,
            [self._parent_leaf_paths[id_val] for id_val in self._dupl_leaf_ids],
        )

    def _param_from_path(self, param_path: LeafPath) -> AnyParameter:
        leaf: AnyParameter = use_path_get_leaf(self.model, param_path)
        if not is_parameter(leaf):
            raise ValueError(f"Leaf at path '{leafpath_to_str(param_path)}' is not a Parameter.")
        return leaf

    def _get_val_and_attr(self, param_path: LeafPath) -> LeafPath:
        param_leaf = self._param_from_path(param_path)
        val_attr = "val" if not is_constrained(param_leaf) else "unconstrained_val"
        val = getattr(param_leaf, val_attr)
        return val, val_attr

    def _get_val_path(self, param_path: LeafPath) -> LeafPath:
        val, val_attr = self._get_val_and_attr(param_path)
        return (
            self._parent_leaf_paths[val.id]
            if isinstance(val, Shared)
            else param_path + (GetAttrKey(val_attr),)
        )

    def _get_fix_path(self, param_path: LeafPath) -> LeafPath:
        return param_path + (GetAttrKey("fix"),)

    def _get_val_paths(self, params: list[str]) -> list[LeafPath]:
        """
        Get the value paths for the given parameter path strings.
        """
        param_paths = [str_to_leafpath(p) for p in params]
        return [self._get_val_path(p) for p in param_paths]

    def _prepare_new_val(self, val_path: LeafPath, new_val: Array) -> Array:
        param_leaf = self._param_from_path(val_path[:-1])
        if param_leaf.val.shape != new_val.shape:
            raise ValueError(
                f"New value shape {new_val.shape} does not match parameter leaf shape {param_leaf.val.shape}."
            )
        if is_constrained(param_leaf):
            return param_leaf.backward_transform(new_val)  # type: ignore
        else:
            return new_val

    def _prepare_new_vals(self, val_paths: list[LeafPath], new_vals: list[Array]) -> list[Array]:
        return [self._prepare_new_val(p, v) for p, v in zip(val_paths, new_vals)]

    def get_locked_model(self) -> Self:
        """
        Get a locked model. Locked models do not properly track shared parameters and so can no longer be optimised. Since all model parameters contained their actual values instead of some containing Shared objects, any subcomponent of the model may be called to make predictions, which is the primary use case for a locked model. You can't convert a locked model back, but this function returns a copy anyway so it doesn't matter.
        """
        cls = type(self)
        return cls(tree_at(self._where, self.model, self._get(self.model)), locked=True)

    def copy(self) -> Self:
        """
        Create a proper copy of the ShareModule with correct sharing structure preserved and duplicated array data. It's like deepcopy but preserves the sharing structure.
        """
        # First, create a fresh restored model with all proper sharing
        restored_model = tree_at(self._where, self.model, self._get(self.model))

        # Create an ID map to keep track of which arrays we've already copied
        # This ensures we create exactly one copy of each unique array
        id_to_copy_map: Dict[int, Array] = {}

        def deep_copy_with_sharing(x):
            if is_inexact_array(x):
                # Get the ID of this array
                x_id = id(x)

                # If we've already copied this exact array, return the existing copy
                if x_id in id_to_copy_map:
                    return id_to_copy_map[x_id]

                # Otherwise, create a new copy and remember it
                x_copy = x.copy()
                id_to_copy_map[x_id] = x_copy
                return x_copy
            return x

        # Apply the deep copy to all leaves in the model, preserving sharing
        copied_model = tree_map(deep_copy_with_sharing, restored_model)

        # Return a new instance with the deep-copied model
        cls = type(self)
        return cls(copied_model, locked=self._locked)

    def get_shared_components(self) -> dict[str, list[str]]:
        """
        Get module-level components that are shared (same object).

        This is useful for visualization - it shows when entire sub-models
        are the same object, not just their leaf parameters. The internal
        sharing mechanism still operates at the parameter level.

        Note: Module sharing is detected during ShareModule initialization
        before any tree manipulation (which can break object identity).

        Returns:
            Dict mapping parent component paths to list of paths that share
            the same object. Only components that are actually shared are included.

        Example:
            >>> model.get_shared_components()
            {'branch_a': ['branch_b']}  # branch_b IS branch_a (same object)
        """
        return self._shared_components

    def get_sharing_summary(self) -> dict[str, list[str]]:
        """
        Get a summary of which parameters are shared with which.

        Returns:
            A dictionary mapping parent parameter paths (strings) to lists of
            child parameter paths that share the same value. Only parameters
            that are actually shared (have duplicates) are included.

        Example:
            >>> model.get_sharing_summary()
            {'line_1.v.coefficients.val': ['line_2.v.coefficients.val']}
        """
        from collections import defaultdict

        # Build a mapping from parent path to list of duplicate paths
        summary: dict[str, list[str]] = defaultdict(list)

        for dupl_id, dupl_path in zip(self._dupl_leaf_ids, self._dupl_leaf_paths):
            parent_path = self._parent_leaf_paths[dupl_id]
            parent_path_str = leafpath_to_str(parent_path)
            dupl_path_str = leafpath_to_str(dupl_path)
            summary[parent_path_str].append(dupl_path_str)

        return dict(summary)

    def get_parameter_paths(self, show_shared: bool = False) -> list[str]:
        """
        Get all parameter paths that can be used with the `set()` method.

        Args:
            show_shared: If False (default), only returns paths to unique/parent
                parameters. If True, also includes paths to shared (duplicate)
                parameters.

        Returns:
            A list of parameter path strings (e.g., ['line.A.coefficients',
            'line.v.coefficients']).

        Example:
            >>> model.get_parameter_paths()
            ['line_1.A.coefficients', 'line_1.v.coefficients', 'line_2.A.coefficients']
            >>> model.get_parameter_paths(show_shared=True)
            ['line_1.A.coefficients', 'line_1.v.coefficients', 'line_2.A.coefficients',
             'line_2.v.coefficients']  # line_2.v is shared with line_1.v
        """
        # Get all unique/parent parameter paths (strip the .val/.unconstrained_val suffix)
        parent_paths: list[str] = []
        for path in self._parent_leaf_paths.values():
            # Remove the last component (.val or .unconstrained_val) to get the Parameter path
            param_path = path[:-1]
            param_path_str = leafpath_to_str(param_path)
            if param_path_str not in parent_paths:
                parent_paths.append(param_path_str)

        if not show_shared:
            return parent_paths

        # Also include shared/duplicate paths
        all_paths = parent_paths.copy()
        for dupl_path in self._dupl_leaf_paths:
            # Remove the last component (.val or .unconstrained_val) to get the Parameter path
            param_path = dupl_path[:-1]
            param_path_str = leafpath_to_str(param_path)
            if param_path_str not in all_paths:
                all_paths.append(param_path_str)

        return all_paths

    def validate_sharing(self, raise_on_error: bool = True) -> dict[str, Any]:
        """
        Validate the sharing structure of the model.

        Checks that:
        1. All Shared objects reference valid parent parameters
        2. Parent parameters exist and have the expected shape
        3. No orphaned Shared objects exist

        Args:
            raise_on_error: If True (default), raises ValueError on validation
                failure. If False, returns validation results without raising.

        Returns:
            A dictionary containing validation results:
            - 'valid': bool - True if all checks passed
            - 'n_shared_params': int - Number of shared parameter relationships
            - 'n_unique_params': int - Number of unique parameters
            - 'errors': list[str] - List of error messages (empty if valid)

        Example:
            >>> result = model.validate_sharing(raise_on_error=False)
            >>> if not result['valid']:
            ...     print("Errors:", result['errors'])
        """
        errors: list[str] = []

        # Check that all parent paths are valid
        for leaf_id, parent_path in self._parent_leaf_paths.items():
            try:
                parent_leaf = use_path_get_leaf(self.model, parent_path)
                # If the model is unlocked, the parent should be an actual array
                # If locked, it should also be an array (not Shared)
                if is_shared(parent_leaf):
                    errors.append(
                        f"Parent at '{leafpath_to_str(parent_path)}' is itself a Shared "
                        f"object, which indicates a bug in sharing detection."
                    )
            except (AttributeError, KeyError) as e:
                errors.append(
                    f"Parent path '{leafpath_to_str(parent_path)}' is invalid: {e}"
                )

        # Check that all duplicate paths reference valid parents
        for dupl_id, dupl_path in zip(self._dupl_leaf_ids, self._dupl_leaf_paths):
            if dupl_id not in self._parent_leaf_paths:
                errors.append(
                    f"Duplicate at '{leafpath_to_str(dupl_path)}' references unknown "
                    f"parent id {dupl_id}."
                )

        # Check that shared parameters have matching fixed states
        for dupl_id, dupl_path in zip(self._dupl_leaf_ids, self._dupl_leaf_paths):
            parent_path = self._parent_leaf_paths.get(dupl_id)
            if parent_path is None:
                continue

            # Get the Parameter objects (strip .val/.unconstrained_val)
            try:
                parent_param = use_path_get_leaf(self.model, parent_path[:-1])
                dupl_param = use_path_get_leaf(self.model, dupl_path[:-1])

                if is_parameter(parent_param) and is_parameter(dupl_param):
                    if parent_param.fix != dupl_param.fix:
                        parent_path_str = leafpath_to_str(parent_path[:-1])
                        dupl_path_str = leafpath_to_str(dupl_path[:-1])
                        parent_status = "fixed" if parent_param.fix else "free"
                        dupl_status = "fixed" if dupl_param.fix else "free"
                        errors.append(
                            f"Shared parameters have mismatched fixed states: "
                            f"'{parent_path_str}' is {parent_status} but "
                            f"'{dupl_path_str}' is {dupl_status}."
                        )
            except (AttributeError, KeyError):
                pass  # Already caught by earlier checks

        result = {
            "valid": len(errors) == 0,
            "n_shared_params": len(self._dupl_leaf_ids),
            "n_unique_params": len(self._parent_leaf_paths),
            "errors": errors,
        }

        if raise_on_error and errors:
            raise ValueError(
                f"Sharing validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        return result

    def set(self, params: list[str], values: list[Array]) -> Self:
        """
        Return a new model with updated parameter values. Can only be used to update values of Parameters or ConstrainedParameters. The model must not be locked.
        """
        if self._locked:
            raise ValueError("Cannot set parameters on a locked model.")
        val_paths = self._get_val_paths(params)
        replace_vals = self._prepare_new_vals(val_paths, values)
        return tree_at(lambda x: use_paths_get_leaves(x, val_paths), self, replace_vals)  # type: ignore[no-any-return]

    def set_fixed_status(self, params: list[str], fix: list[bool]) -> Self:
        """
        Return a new model with parameters updated to be fixed or not based on provided paths and list of bools. Can only be used to update fixed statuses of Parameters or ConstrainedParameters. The model must not be locked.
        """
        # TODO: Refactor based on reused functionality
        if self._locked:
            raise ValueError("Cannot set parameters on a locked model.")
        # Convert the path strings to LeafPath
        param_paths = [str_to_leafpath(p) for p in params]
        # Iterate over all paths, find all shared copies of each, since we update them all
        fix_paths = []
        new_fix = []
        for pp, ff in zip(param_paths, fix):
            val, val_attr = self._get_val_and_attr(pp)
            if is_shared(val):
                p_id = val.id
            else:
                p_id = id_from_path(self._parent_leaf_paths, pp + (GetAttrKey(val_attr),))
            fix_paths.append(self._parent_leaf_paths[p_id][:-1] + (GetAttrKey("fix"),))
            new_fix.append(ff)
            inds = get_indices(self._dupl_leaf_ids, p_id)
            for ii in inds:
                fix_paths.append(self._dupl_leaf_paths[ii][:-1] + (GetAttrKey("fix"),))
                new_fix.append(ff)
        return tree_at(lambda x: use_paths_get_leaves(x, fix_paths), self, new_fix)  # type: ignore[no-any-return]

    def fix_all(self) -> Self:
        """
        Return a new model with all parameters set to fixed.

        This is useful for freezing all parameters, e.g., when you want to
        use the model for inference only without any further optimization.

        Returns:
            A new ShareModule with all parameters fixed.

        Example:
            >>> model = build_model(MyModel, ...)
            >>> frozen_model = model.fix_all()
            >>> # All parameters are now fixed
        """
        if self._locked:
            raise ValueError("Cannot modify parameters on a locked model.")

        # Get all parameter paths (including shared ones)
        all_paths = self.get_parameter_paths(show_shared=True)

        # Set all to fixed
        return self.set_fixed_status(all_paths, [True] * len(all_paths))

    def free_all(self) -> Self:
        """
        Return a new model with all parameters set to free (not fixed).

        This is useful for unfreezing all parameters for optimization.

        Returns:
            A new ShareModule with all parameters free.

        Example:
            >>> frozen_model = model.fix_all()
            >>> unfrozen_model = frozen_model.free_all()
            >>> # All parameters are now free
        """
        if self._locked:
            raise ValueError("Cannot modify parameters on a locked model.")

        # Get all parameter paths (including shared ones)
        all_paths = self.get_parameter_paths(show_shared=True)

        # Set all to free
        return self.set_fixed_status(all_paths, [False] * len(all_paths))

    def parameter_summary(
        self, show_shared: bool = True, show_knowns: bool = False
    ) -> None:
        """
        Print a summary table of all parameters with their status.

        Shows path, shape, bounds, free/fixed status, and sharing info.

        Args:
            show_shared: If True (default), include parameters that are shared
                from other parameters. If False, only show unique/parent parameters.
            show_knowns: If False (default), exclude Known parameters from the
                summary. If True, include them with status 'known'.

        Example:
            >>> model.parameter_summary()
            Parameter Summary
            ──────────────────────────────────────────────────────────────────
            Path                      Shape     Bounds       Status  Shared from
            ──────────────────────────────────────────────────────────────────
            line_1.A.coefficients     (10,)     (-∞, ∞)      free    -
            line_1.v.coefficients     (10,)     (-∞, ∞)      fixed   -
            line_2.v.coefficients     (10,)     (-∞, ∞)      fixed   line_1.v.coefficients
            ──────────────────────────────────────────────────────────────────
            3 parameters (2 unique, 1 shared) · 1 free, 2 fixed
        """
        from rich.table import Table
        from rich.text import Text

        from spectracles.model.formatting import get_console
        from spectracles.model.parameter import (
            ConstrainedParameter,
            Known,
            Parameter,
        )

        console = get_console()

        # Get the locked model for traversal
        locked = self.get_locked_model()

        # Build list of parameter info
        params_info = []
        sharing_summary = self.get_sharing_summary()

        # Create reverse lookup: shared_path -> parent_path
        shared_to_parent: Dict[str, str] = {}
        for parent_path, shared_paths in sharing_summary.items():
            # Strip .val/.unconstrained_val suffix
            parent_clean = (
                parent_path.rsplit(".", 1)[0] if "." in parent_path else parent_path
            )
            for shared_path in shared_paths:
                shared_clean = (
                    shared_path.rsplit(".", 1)[0] if "." in shared_path else shared_path
                )
                shared_to_parent[shared_clean] = parent_clean

        # Get unique paths and shared paths
        unique_paths = self.get_parameter_paths(show_shared=False)
        all_paths = self.get_parameter_paths(show_shared=True)
        shared_paths_set = set(all_paths) - set(unique_paths)

        for path in all_paths:
            # Skip shared params if not showing them
            is_shared = path in shared_paths_set
            if is_shared and not show_shared:
                continue

            # Get the parameter object from the locked model (has real values, not Shared)
            param = use_path_get_leaf(locked.model, str_to_leafpath(path))

            # Check if it's a Known
            is_known = isinstance(param, Known)
            if is_known and not show_knowns:
                continue

            # Get shape
            if isinstance(param, ConstrainedParameter):
                shape = param.unconstrained_val.shape
            else:
                shape = param.val.shape

            # Get bounds
            if isinstance(param, ConstrainedParameter):
                lower = param._lower
                upper = param._upper
                is_log = param._log
                if is_log:
                    bounds = "(0, ∞) log"
                elif lower is not None and upper is not None:
                    bounds = f"({lower}, {upper})"
                elif lower is not None:
                    bounds = f"({lower}, ∞)"
                elif upper is not None:
                    bounds = f"(-∞, {upper})"
                else:
                    bounds = "(-∞, ∞)"
            else:
                bounds = "(-∞, ∞)"

            # Get status
            if is_known:
                status = "known"
            elif param.fix:
                status = "fixed"
            else:
                status = "free"

            # Get parent (for shared params)
            parent = shared_to_parent.get(path, "-")

            params_info.append(
                {
                    "path": path,
                    "shape": str(shape),
                    "bounds": bounds,
                    "status": status,
                    "parent": parent,
                    "is_shared": is_shared,
                    "is_known": is_known,
                }
            )

        # Build the table
        table = Table(
            title="Parameter Summary",
            show_header=True,
            header_style="bold",
            border_style="dim",
            title_style="bold",
        )

        table.add_column("Path", style="white")
        table.add_column("Shape", style="dim")
        table.add_column("Bounds", style="dim")
        if show_shared:
            table.add_column("Status", justify="center")
            table.add_column("Shared from", style="#cba6f7")  # Mauve/shared color
        else:
            table.add_column("Status", justify="center")

        for info in params_info:
            # Style the status
            if info["status"] == "free":
                status_text = Text("free", style="#a6e3a1")  # Green
            elif info["status"] == "fixed":
                status_text = Text("fixed", style="#f9e2af")  # Yellow
            else:  # known
                status_text = Text("known", style="dim")

            if show_shared:
                table.add_row(
                    info["path"],
                    info["shape"],
                    info["bounds"],
                    status_text,
                    info["parent"],
                )
            else:
                table.add_row(
                    info["path"],
                    info["shape"],
                    info["bounds"],
                    status_text,
                )

        console.print(table)

        # Summary line
        total = len(params_info)
        n_unique = sum(1 for p in params_info if not p["is_shared"])
        n_shared = sum(1 for p in params_info if p["is_shared"])
        n_free = sum(1 for p in params_info if p["status"] == "free")
        n_fixed = sum(1 for p in params_info if p["status"] == "fixed")
        n_known = sum(1 for p in params_info if p["status"] == "known")

        summary = Text()
        if show_shared and n_shared > 0:
            summary.append(f"{total} parameters ", style="dim")
            summary.append(f"({n_unique} unique, {n_shared} shared)", style="dim")
        else:
            summary.append(f"{n_unique} parameters", style="dim")

        if show_knowns and n_known > 0:
            summary.append(f" + {n_known} known", style="dim")

        summary.append(" · ", style="dim")
        summary.append(f"{n_free} free", style="#a6e3a1")
        summary.append(", ", style="dim")
        summary.append(f"{n_fixed} fixed", style="#f9e2af")

        console.print(summary)

    def print_model_tree(self, show_sharing: bool = False) -> None:
        """
        Print the model tree in an easy to parse format.

        Args:
            show_sharing: If True, also print summaries of sharing relationships
                after the tree. Shows both component-level sharing (entire modules
                that are the same object) and parameter-level sharing.
                Default is False for backwards compatibility.

        Example with show_sharing=True:
            └── model (LineRatioModel)
                ├── line_1 (EmissionLine)
                │   └── v (GPField)
                └── line_2 (LinkedEmissionLine)
                    └── v (GPField)

            Shared components (same object):
              line_1.v  ←  line_2.v

            Shared parameters:
              line_1.v.coefficients  ←  line_2.v.coefficients
        """
        from rich.text import Text
        from spectracles.model.formatting import get_console

        console = get_console()

        graph, root_id = get_digraph(self)
        print_graph(graph, root_id)

        if show_sharing:
            # Show component-level sharing (entire modules that are the same object)
            shared_components = self.get_shared_components()
            if shared_components:
                header = Text("\nShared components ", style="dim")
                header.append("(same object)", style="dim")
                header.append(":", style="dim")
                console.print(header)
                for parent_path, shared_paths in shared_components.items():
                    for shared_path in shared_paths:
                        line = Text("  ")
                        line.append(parent_path, style="value")
                        line.append("  ←  ", style="shared")
                        line.append(shared_path, style="shared")
                        console.print(line)

            # Show parameter-level sharing
            if self._dupl_leaf_ids:
                header = Text("\nShared parameters:", style="dim")
                console.print(header)
                summary = self.get_sharing_summary()
                for parent_path, dupl_paths in summary.items():
                    # Strip .val/.unconstrained_val suffix for cleaner display
                    parent_display = parent_path.rsplit(".", 1)[0] if "." in parent_path else parent_path
                    for dupl_path in dupl_paths:
                        dupl_display = dupl_path.rsplit(".", 1)[0] if "." in dupl_path else dupl_path
                        line = Text("  ")
                        line.append(parent_display, style="value")
                        line.append("  ←  ", style="shared")
                        line.append(dupl_display, style="shared")
                        console.print(line)

    def plot_model_graph(
        self,
        ax: Axes | None = None,
        show: bool = True,
        label_func: Callable[[DiGraph], Dict[int, str]] | None = None,
        nx_draw_kwds: dict = DEFAULT_NX_KWDS,
    ) -> None:
        """
        Plot the model as a graph using networkx and matplotlib. The graph is directed towards parameters and accounts for the sharing structure.
        """
        with temporarily_disable_tex():  # TODO: implement this and check it works
            graph, root_id = get_digraph(self)
            pos = layered_hierarchy_pos(graph, root_id)
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 10), layout="compressed")
            if label_func is None:
                labels = {n: f"{d['name']}\n({d['type']})" for n, d in graph.nodes(data=True)}
            else:
                labels = label_func(graph)

            NODES = list(graph.nodes())
            NODE_SIZE = nx_draw_kwds.get("node_size", 8000)

            draw_networkx_nodes(
                graph,
                pos,
                ax=ax,
                node_size=nx_draw_kwds["node_size"],
                node_color=nx_draw_kwds["node_color"],
                edgecolors=nx_draw_kwds["edgecolors"],
                linewidths=nx_draw_kwds["linewidths"],
            )
            draw_networkx_labels(
                graph,
                pos,
                labels=labels,
                ax=ax,
                font_size=nx_draw_kwds.get("font_size", 8),
                font_color=nx_draw_kwds.get("font_color", "black"),
            )

            # split edges into cross-level vs same-level
            same_level, cross_level = [], []
            for u, v in graph.edges():
                if abs(pos[u][1] - pos[v][1]) < 1e-12:
                    same_level.append((u, v))
                else:
                    cross_level.append((u, v))

            # straight cross-level edges
            draw_networkx_edges(
                graph,
                pos,
                edgelist=cross_level,
                ax=ax,
                arrows=True,
                arrowsize=nx_draw_kwds.get("arrowsize", 20),
                width=1.5,
                nodelist=NODES,
                node_size=NODE_SIZE,
                min_target_margin=2.0,
                min_source_margin=2.0,
            )

            # curved same-level edges (stagger radii so parallel arcs don’t overlap)
            arc_rads = (0.25, 0.37, 0.49)
            for i, e in enumerate(same_level):
                draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=[e],
                    ax=ax,
                    arrows=True,
                    arrowsize=nx_draw_kwds.get("arrowsize", 20),
                    connectionstyle=f"arc3,rad={arc_rads[i % len(arc_rads)]}",
                    # style="dashed",
                    # alpha=0.85,
                    width=1.5,
                    nodelist=NODES,
                    node_size=NODE_SIZE,
                    min_target_margin=2.0,
                    min_source_margin=2.0,
                )
            if show:
                plt.show()


def parent_model(model) -> ShareModule:
    # Check if it is already wrapped
    if isinstance(model, ShareModule):
        return model
    return ShareModule(model)


def build_model(cls: Callable[..., Module], *args, **kwargs) -> ShareModule:
    return parent_model(cls(*args, **kwargs))


def get_digraph(module: ShareModule) -> tuple[DiGraph, int]:
    # Filter tree and extract leaves + paths
    filtered_tree = filter(module, is_parameter)
    leaves = leaves_with_path(filtered_tree.model, is_leaf=is_parameter)
    paths = [leaf[0] for leaf in leaves]
    leaves = [leaf[1] for leaf in leaves]
    # Start with root node
    root_id = id(module.model)
    graph: DiGraph = DiGraph()
    graph.add_node(
        root_id,
        name="model",
        type=module.model.__class__.__name__,
        is_param=False,
        is_root=True,
    )
    # Select one path to a leaf
    for p in paths:
        # Iterate from model down to leaf
        parent = module.model
        for entry in p:
            leaf = getattr(parent, entry.name)
            # Figure out what leaf we are adding, accounting for sharing
            if is_parameter(leaf):
                if is_constrained(leaf):
                    val_attr = "unconstrained_val"
                else:
                    val_attr = "val"
                val_leaf = getattr(leaf, val_attr)
                if is_shared(val_leaf):
                    parent_path = module._parent_leaf_paths[val_leaf.id][:-1]
                else:
                    parent_path = p
                leaf = use_path_get_leaf(module, parent_path)
            # If it was a shared leaf, then the following will actually do nothing, as desired
            graph.add_node(
                id(leaf),
                name=entry.name,
                type=leaf.__class__.__name__,
                is_param=is_parameter(leaf),
                is_root=False,
            )
            # Connect edges then set leaf to parent for next loop
            graph.add_edge(id(parent), id(leaf))
            parent = leaf
    return graph, root_id
