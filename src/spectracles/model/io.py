"""io.py - Model object serialisation and deserialisation."""

import sys
from pathlib import Path
from pickle import PicklingError
from types import CodeType
from typing import Any
from warnings import catch_warnings, filterwarnings

from dill import Pickler, PicklingWarning, load  # type: ignore[import]
from dill import settings as dill_settings  # type: ignore[import]
from dill._dill import MetaCatchingDict, _create_code  # type: ignore[import]

from spectracles.model.share_module import ShareModule

MODELFILE_EXT = ".model"


class _ModelPickler(Pickler):
    """dill's Pickler, with code objects saved on every supported Python.

    dill (0.4.1) reads `co_lnotab` when saving a code object, and Python 3.15
    removed it. dill stores it only for Pythons before 3.10 and drops it when
    loading, so on 3.15 and later an empty one is stored in its place.
    """

    dispatch = MetaCatchingDict(Pickler.dispatch.copy())


def _save_code(pickler: Any, obj: CodeType) -> None:
    args = (
        b"",  # co_lnotab
        obj.co_argcount,
        obj.co_posonlyargcount,
        obj.co_kwonlyargcount,
        obj.co_nlocals,
        obj.co_stacksize,
        obj.co_flags,
        obj.co_code,
        obj.co_consts,
        obj.co_names,
        obj.co_varnames,
        obj.co_filename,
        obj.co_name,
        obj.co_qualname,
        obj.co_firstlineno,
        obj.co_linetable,
        obj.co_exceptiontable,
        obj.co_freevars,
        obj.co_cellvars,
    )
    pickler.save_reduce(_create_code, args, obj=obj)


if sys.version_info >= (3, 15):
    _ModelPickler.dispatch[CodeType] = _save_code


def _dump(obj, file, protocol=None, byref=None, fmode=None, recurse=None, **kwds) -> None:
    """`dill.dump`, with `_ModelPickler`."""
    protocol = dill_settings["protocol"] if protocol is None else int(protocol)
    kwds.update(byref=byref, fmode=fmode, recurse=recurse)
    _ModelPickler(file, protocol, **kwds).dump(obj)


def save_model(model: ShareModule, file: Path, overwrite: bool = False, **dump_kwargs):
    """
    Save a model to a file.

    Args:
        model (ShareModule): The model to save.
        file (Path): The file to save the model to.
        overwrite (bool): Whether to overwrite the file if it exists. Defaults to False.
        **dump_kwargs (Any): Additional arguments to pass to dill.dump.

    Raises:
        FileExistsError: If the file already exists and overwrite is False.
        TypeError: If the model is not of type ShareModule.
        Exception: If there is a warning when saving the model.
    """
    # Ensure always the same file extension
    file = file.with_suffix(MODELFILE_EXT)
    # Check that the file doesn't exist already
    if file.exists() and not overwrite:
        raise FileExistsError("File already exists. Overwrite with overwrite=True.")
    # Check that model has the right type
    if not isinstance(model, ShareModule):
        raise TypeError(
            "model must be type ShareModule. Saving sub-components of a model is not currenly supported. Saving a model not instantiated via build_model will never be supported."
        )
    # We want dill warnings to be exceptions
    with catch_warnings():
        filterwarnings("error")
        # Open file in write and bytes mode
        with open(file, "wb") as f:
            try:
                _dump(model, f, **dump_kwargs)
            except PicklingWarning:
                raise PicklingError(
                    "Above warning raised by dill when saving. Likely, you need to move your model class into it's own file and import it."
                )


def load_model(file: Path, **load_kwargs) -> ShareModule:
    """
    Load a model from a file.

    Args:
        file (Path): The file to load the model from.
        **load_kwargs (Any): Additional arguments to pass to dill.load.

    Returns:
        ShareModule: The loaded model.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the loaded model is not of type ShareModule.
    """
    # Ensure always the same file extension
    file = file.with_suffix(MODELFILE_EXT)
    # Check that the file exists
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist.")
    # Open file in read and bytes mode
    with open(file, "rb") as f:
        model = load(f, **load_kwargs)
    # Check that the model is of the right type
    if not isinstance(model, ShareModule):
        raise TypeError("Loaded from file successfully, but model is not of type ShareModule.")
    return model
