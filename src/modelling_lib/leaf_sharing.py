from typing import Any, Callable, Dict, Tuple

# import equinox as eqx
from equinox import Module, field, tree_at
from jax.tree import leaves_with_path
from jaxlib.xla_extension.pytree import DictKey, GetAttrKey, SequenceKey
from jaxtyping import Array, PyTree

from .parameter import Param

LeafKey = SequenceKey | DictKey | GetAttrKey
LeafPath = Tuple[LeafKey, ...]


def use_path_get_leaf(tree: PyTree, path: LeafPath) -> Array | None:
    """
    Iterates through the path to find the leaf in the tree, returning the leaf value only if it is a Param. If the leaf is not a Param, returns None.
    """
    current_node = tree
    for key in path:
        if isinstance(current_node, Param):
            return getattr(current_node, key.name)
        if isinstance(key, GetAttrKey):
            current_node = getattr(current_node, key.name)
        elif isinstance(key, SequenceKey):
            current_node = current_node[key.idx]
        elif isinstance(key, DictKey):
            current_node = current_node[key.key]
        else:
            raise TypeError(
                f"Unsupported key type: {type(key)}. In practice if this happens you're either doing something crazy or JAX has added new types. Honestly, raise a GH issue if you see this, because it's probably the latter."
            )
    return None


def use_paths_get_leaves(tree: PyTree, paths: list[LeafPath]) -> list[Any]:
    leaves = []
    for path in paths:
        leaf = use_path_get_leaf(tree, path)
        if leaf is not None:
            print(path)
            print(type(leaf))
            leaves.append(leaf)
    return leaves


# def unwrap_leaf(leaf: Any) -> Any:
#     # Unwrap the leaf if it's an instance of _LeafWrapper
#     if isinstance(leaf, eqx._tree._LeafWrapper):
#         return leaf.value
#     return leaf


def get_duplicated_leaves(tree: PyTree) -> Tuple[list[int], list[LeafPath], dict[int, LeafPath]]:
    leaves = leaves_with_path(tree, is_leaf=lambda x: isinstance(x, Param))
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


class ShareParams(Module):
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
        self.model = tree_at(self._where, model, replace_fn=lambda _: 0)

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


def parent_model(model) -> ShareParams:
    # Check if it is already wrapped
    if isinstance(model, ShareParams):
        return model
    return ShareParams(model)


def build_model(cls: Callable[..., Module], *args, **kwargs) -> ShareParams:
    return parent_model(cls(*args, **kwargs))
