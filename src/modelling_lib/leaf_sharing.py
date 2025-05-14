from typing import Any, Callable, Dict, Self, Tuple, TypeAlias

from equinox import Module, filter, is_inexact_array, tree_at
from jax.tree import leaves, leaves_with_path
from jax.tree_util import tree_map
from jaxlib.xla_extension.pytree import GetAttrKey
from jaxtyping import PyTree

from .parameter import AnyParameter

LeafPath: TypeAlias = Tuple[GetAttrKey, ...]


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


def get_duplicated_leaves(tree: PyTree) -> Tuple[list[int], list[LeafPath], dict[int, LeafPath]]:
    # Filter out leaves that are not Parameter
    filter_spec = tree_map(
        lambda x: isinstance(x, AnyParameter), tree, is_leaf=lambda x: isinstance(x, AnyParameter)
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

    id: int

    def __init__(self, id: int):
        self.id = id

    def __repr__(self) -> str:
        return f"Shared({self.id})"

    def __str__(self) -> str:
        return f"Shared({self.id})"


class ShareModule(Module):
    # None only support since we intialise to None to avoid recursion problems with the custom __getattr__
    model: None | Module

    # Sharing metadata
    _dupl_leaf_ids: list[int]
    _dupl_leaf_paths: list[LeafPath]
    _parent_leaf_paths: Dict[int, LeafPath]

    # Is this instance locked?
    _locked: bool = False

    def __init__(self, model: Module, locked: bool = False):
        # Save the sharing info
        (
            self._dupl_leaf_ids,
            self._dupl_leaf_paths,
            self._parent_leaf_paths,
        ) = get_duplicated_leaves(model)
        # Other metadata
        self._locked = locked

        # Remove leaves that are coupled to other leaves
        def replace_fn(leaf):
            return Shared(id(leaf))

        # Initialise the model to None to avoid recursion issues
        self.model = None

        # If locked, we don't want Shared() objects because all sub-models need to be callable
        # and if we replace some leaves with Shared() objects, they won't be
        if locked:
            self.model = model
        # Otherwise, replace the leaves with Shared objects
        else:
            self.model = tree_at(self._where, model, replace_fn=replace_fn)

    def __getattr__(self, name):
        # Delegate unknown attributes to the model
        return getattr(self.model, name)

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

    def get_locked_model(self) -> Self:
        """
        Get a locked model. Locked models do not properly track shared parameters and so can no longer be optimised. Since all model parameters contained their actual values instead of some containing Shared objects, any subcomponent of the model may be called to make predictions, which is the primary use case for a locked model. You can't convert a locked model back, but this function returns a copy anyway so it doesn't matter.
        """
        cls = type(self)
        return cls(tree_at(self._where, self.model, self._get(self.model)), locked=True)

    def copy(self) -> Self:
        """
        NOTE: BROKEN since it the memory ids in the Shared objects are not updated
        Return a copy of this model this is bad don't use it. (I can use it, I know how it's bad)
        """
        return tree_at(leaves, self, replace_fn=lambda x: x)  # type: ignore[no-any-return]

    def rebuild(self) -> Self:
        """
        Return a rebuilt copy of this model. Rebuilding is useful in case you have replaced any Parameters using tree surgery, since it will re-calculate the sharing structure. Changing Parameters of a built model without rebuilding can have unintended consequences and so is not recommended.
        """
        raise NotImplementedError


def parent_model(model) -> ShareModule:
    # Check if it is already wrapped
    if isinstance(model, ShareModule):
        return model
    return ShareModule(model)


def build_model(cls: Callable[..., Module], *args, **kwargs) -> ShareModule:
    return parent_model(cls(*args, **kwargs))
