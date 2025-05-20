"""path_utils.py - utilities for handling paths in PyTree structures."""

from typing import Any

from equinox import filter
from jax.tree import leaves_with_path
from jax.tree_util import GetAttrKey
from jaxtyping import PyTree

type LeafPath = tuple[GetAttrKey, ...]  # type: ignore


def str_to_leafpath(str_path: str) -> LeafPath:
    """Convert a string path to a LeafPath."""
    return tuple(GetAttrKey(name) for name in str_path.split("."))


def leafpath_to_str(leaf_path: LeafPath) -> str:
    """Convert a LeafPath to a string path."""
    return ".".join([key.name for key in leaf_path])


def use_path_get_leaf(tree: PyTree, path: LeafPath) -> Any:
    """
    Iterates through the path to find the leaf in the tree. Doesn't work if leaves are sequences.
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


def get_duplicated_leaves(
    tree: PyTree,
    filter_specs: list[PyTree],
) -> tuple[list[int], list[LeafPath], dict[int, LeafPath]]:
    """
    Identifies duplicated leaves in a PyTree after applying a sequence of filters.

    Args:
        tree: The input PyTree to analyze
        filter_specs: A list of PyTree filters to apply sequentially. If None,
                     default filters for Parameters and inexact arrays will be used.

    Returns:
        A tuple containing:
        - List of IDs of duplicated leaves
        - List of paths to duplicated leaves
        - Dictionary mapping leaf IDs to their original paths
    """

    # Apply all filters sequentially
    filtered_tree = tree
    for filter_spec in filter_specs:
        filtered_tree = filter(filtered_tree, filter_spec=filter_spec)

    # Get all leaves with their paths
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
