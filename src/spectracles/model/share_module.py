"""share_module.py - Models built on equinox and jax that can have parameters that are shared between leaves of the model tree. Module refers to equinox Modules not Python modules."""

from typing import Any, Callable, Dict, Self

import matplotlib.pyplot as plt
from equinox import Module, filter, is_inexact_array, tree_at
from jax.tree import leaves_with_path
from jax.tree_util import tree_map
from jaxtyping import Array, PyTree
from matplotlib.axes import Axes
from networkx import DiGraph, draw

from spectracles.model.graph import DEFAULT_NX_KWDS, layered_hierarchy_pos, print_graph
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


class Shared:
    """A sentinel object used to indicate a parameter is shared."""

    id: int

    def __init__(self, id: int):
        self.id = id

    def __repr__(self) -> str:
        return f"Shared({self.id})"

    def __str__(self) -> str:
        return f"Shared({self.id})"


def is_shared(x: Any) -> bool:
    return isinstance(x, Shared)


class ShareModule(Module):
    model: Module

    # Sharing metadata
    _dupl_leaf_ids: list[int]
    _dupl_leaf_paths: list[LeafPath]
    _parent_leaf_paths: Dict[int, LeafPath]

    # Is this instance locked?
    _locked: bool = False

    # Keep track of attributes to avoid recursion
    _attr_names = {"model", "_dupl_leaf_ids", "_dupl_leaf_paths", "_parent_leaf_paths", "_locked"}

    def __init__(self, model: Module, locked: bool = False):
        # Save the sharing info
        (
            self._dupl_leaf_ids,
            self._dupl_leaf_paths,
            self._parent_leaf_paths,
        ) = get_duplicated_parameters(model)
        # Other metadata
        self._locked = locked

        # Remove leaves that are coupled to other leaves
        def replace_fn(leaf):
            return Shared(id(leaf))

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
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(
                f"Neither {type(self).__name__} nor {type(self.model).__name__} has attribute {name}"
            )

    def __getstate__(self):
        # Make sure we don't include any computed properties that might cause recursion
        return {
            "model": self.model,
            "_dupl_leaf_ids": self._dupl_leaf_ids,
            "_dupl_leaf_paths": self._dupl_leaf_paths,
            "_parent_leaf_paths": self._parent_leaf_paths,
            "_locked": self._locked,
        }

    def __setstate__(self, state):
        # When dealing with frozen instances, we need to use object.__setattr__
        for key, value in state.items():
            object.__setattr__(self, key, value)

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

    def print_model_tree(self) -> None:
        """
        Print the model tree in an easy to parse format. This is a simple tree structure that shows the model and its parameters. It does not show the sharing structure.
        """
        print_graph(*get_digraph(self))

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
        # with temporarily_disable_tex(): # TODO: implement this and check it works
        graph, root_id = get_digraph(self)
        pos = layered_hierarchy_pos(graph, root_id)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10), layout="compressed")
        if label_func is None:
            labels = {n: f"{d['name']}\n({d['type']})" for n, d in graph.nodes(data=True)}
        else:
            labels = label_func(graph)
        draw(graph, pos, ax=ax, labels=labels, **nx_draw_kwds)
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
