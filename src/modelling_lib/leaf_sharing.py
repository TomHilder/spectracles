from typing import Any, Callable, Dict, Tuple

from equinox import Module, field, filter, is_inexact_array, tree_at
from jax.tree import leaves_with_path
from jax.tree_util import tree_map
from jaxlib.xla_extension.pytree import DictKey, GetAttrKey, SequenceKey
from jaxtyping import Array, PyTree

from .parameter import Parameter

LeafKey = SequenceKey | DictKey | GetAttrKey
LeafPath = Tuple[LeafKey, ...]


def use_path_get_leaf(tree: PyTree, path: LeafPath) -> Array | None:
    """
    Iterates through the path to find the leaf in the tree.
    """
    current_node = tree
    for key in path:
        current_node = getattr(current_node, key.name)
    return current_node


def use_paths_get_leaves(tree: PyTree, paths: list[LeafPath]) -> list[Any]:
    """
    Iterates through the paths to find the leaves in the tree. Returns a list of leaves.
    """
    leaves = []
    for path in paths:
        leaf = use_path_get_leaf(tree, path)
        if leaf is not None:
            leaves.append(leaf)
    return leaves


def get_duplicated_leaves(tree: PyTree) -> Tuple[list[int], list[LeafPath], dict[int, LeafPath]]:
    # Filter out leaves that are not Parameter
    filter_spec = tree_map(
        lambda x: isinstance(x, Parameter), tree, is_leaf=lambda x: isinstance(x, Parameter)
    )
    filtered_tree = filter(tree, filter_spec=filter_spec)
    # Filter out leaves that are not inexact arrays (avoids Parameter class's other attributes)
    filtered_tree = filter(
        filtered_tree, filter_spec=tree_map(lambda x: is_inexact_array(x), filtered_tree)
    )
    leaves = leaves_with_path(filtered_tree)
    # Create a dictionary to keep track of the parent leaves
    parent_leaf_paths: dict[int, LeafPath] = dict()
    dupl_leaf_paths = []
    dupl_leaf_ids = []
    # Go through all leaves and keep track of ids
    for path, leaf in leaves:
        leaf_id = id(leaf)
        # If id already seen:
        if leaf_id in parent_leaf_paths.keys():
            # Remember path to duplicated leaf
            dupl_leaf_paths.append(path)
            dupl_leaf_ids.append(leaf_id)
        # If not already seen:
        else:
            # Add path to dictionary with id as key
            parent_leaf_paths[leaf_id] = path
    return dupl_leaf_ids, dupl_leaf_paths, parent_leaf_paths


class Shared:
    """A sentinel object used to indicate a parameter is shared."""

    def __repr__(self):
        return "shared"

    def __str__(self):
        return "shared"


class ShareModule(Module):
    model: Module

    _dupl_leaf_ids: list[int] = field(static=True)
    _dupl_leaf_paths: list[LeafPath] = field(static=True)
    _parent_leaf_paths: Dict[int, LeafPath] = field(static=True)

    def __init__(self, model: Module):
        # Save the sharing info
        (
            self._dupl_leaf_ids,
            self._dupl_leaf_paths,
            self._parent_leaf_paths,
        ) = get_duplicated_leaves(model)
        # Remove leaves that are coupled to other leaves
        self.model = tree_at(self._where, model, replace_fn=lambda _: Shared())

    def __call__(self, *args, **kwargs):
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


def parent_model(model) -> ShareModule:
    # Check if it is already wrapped
    if isinstance(model, ShareModule):
        return model
    return ShareModule(model)


def build_model(cls: Callable[..., Module], *args, **kwargs) -> ShareModule:
    return parent_model(cls(*args, **kwargs))
