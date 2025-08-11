"""io.py - Model object serialisation and deserialisation."""

from pathlib import Path
from warnings import catch_warnings, filterwarnings

from dill import PicklingWarning, dump, load  # type: ignore[import]

from spectracles.model.share_module import ShareModule

MODELFILE_EXT = ".model"


def save_model(model: ShareModule, file: Path, overwrite: bool = False, **dump_kwargs):
    """
    Save a model to a file.

    Args:
        model (ShareModule): The model to save.
        file (Path): The file to save the model to.
        overwrite (bool): Whether to overwrite the file if it exists. Defaults to False.
        dump_kwargs: Additional arguments to pass to dill.dump.

    Raises:
        FileExistsError: If the file already exists and overwrite is False.
        TypeError: If the model is not of type ShareModule.
        Exception: If there is a warning when saving the model.
    """
    # Ensure always the same file extension
    file = file.with_suffix(MODELFILE_EXT)
    # Check that the file doesn't exist already
    if file.exists():
        if not overwrite:
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
                dump(model, f, **dump_kwargs)
            except PicklingWarning:
                raise Exception(
                    "Above warning raised by dill when saving. Likely, you need to move your model class into it's own file and import it."
                )


def load_model(file: Path, **load_kwargs) -> ShareModule:
    """
    Load a model from a file.

    Args:
        file (Path): The file to load the model from.
        load_kwargs: Additional arguments to pass to dill.load.

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
