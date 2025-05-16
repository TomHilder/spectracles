"""Tests for modelling_lib.io module."""

import pytest
from modelling_lib.io import MODELFILE_EXT, load_model, save_model
from modelling_lib.leaf_sharing import ShareModule, build_model

from .test_models import ComplexSharedModel, SimpleModel


class TestSaveModel:
    def test_basic_save(self, temp_dir):
        # Create a model
        model = build_model(SimpleModel, value=1.0)

        # Save the model
        file_path = temp_dir / "simple_model"
        save_model(model, file_path)

        # Check that the file exists with correct extension
        assert (file_path.with_suffix(MODELFILE_EXT)).exists()

    def test_save_with_custom_suffix(self, temp_dir):
        # Create a model
        model = build_model(SimpleModel, value=1.0)

        # Save the model with custom suffix
        file_path = temp_dir / "simple_model.custom"
        save_model(model, file_path)

        # Check that the file exists with correct extension (should override custom suffix)
        assert (temp_dir / f"simple_model{MODELFILE_EXT}").exists()
        assert not (temp_dir / "simple_model.custom").exists()

    def test_save_with_overwrite(self, temp_dir):
        # Create a model
        model1 = build_model(SimpleModel, value=1.0)
        model2 = build_model(SimpleModel, value=2.0)

        # Save the first model
        file_path = temp_dir / "model"
        save_model(model1, file_path)

        # Save the second model without overwrite (should fail)
        with pytest.raises(FileExistsError):
            save_model(model2, file_path)

        # Save the second model with overwrite
        save_model(model2, file_path, overwrite=True)

    def test_save_non_sharemodule(self, temp_dir):
        # Create a model that's not wrapped with ShareModule
        model = SimpleModel(value=1.0)

        # Save the model
        file_path = temp_dir / "model"

        # Should raise a TypeError
        with pytest.raises(TypeError):
            save_model(model, file_path)


class TestLoadModel:
    def test_basic_load(self, temp_dir):
        # Create and save a model
        original_model = build_model(SimpleModel, value=1.0)
        file_path = temp_dir / "model"
        save_model(original_model, file_path)

        # Load the model
        loaded_model = load_model(file_path)

        # Check that it's a ShareModule
        assert isinstance(loaded_model, ShareModule)

        # Check that the parameter value is correct
        assert loaded_model.param.val == original_model.param.val

    def test_load_nonexistent_file(self, temp_dir):
        # Try to load a nonexistent file
        file_path = temp_dir / "nonexistent"

        # Should raise a FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_model(file_path)

    def test_load_with_custom_suffix(self, temp_dir):
        # Create and save a model
        original_model = build_model(SimpleModel, value=1.0)
        file_path = temp_dir / "model.custom"
        save_model(original_model, file_path)

        # Load the model with correct extension
        loaded_model = load_model(temp_dir / "model")

        # Check that it's loaded correctly
        assert isinstance(loaded_model, ShareModule)

    def test_save_and_load_complex_model(self, temp_dir):
        # Create a complex model with shared parameters
        original_model = build_model(ComplexSharedModel, value=2.0)

        # Save the model
        file_path = temp_dir / "complex_model"
        save_model(original_model, file_path)

        # Load the model
        loaded_model = load_model(file_path)

        # Use the sharing structure to removed Shared sentinels
        locked_loaded_model = loaded_model.get_locked_model()

        # Check that the model structure is preserved
        # inner1.param, param, inner2.param, and inner3.inner1.param should all be the same
        param_val = locked_loaded_model.inner1.param.val
        assert locked_loaded_model.param.val == param_val
        assert locked_loaded_model.inner2.param.val == param_val
        assert locked_loaded_model.inner3.inner1.param.val == param_val

        # Check that calling the model works
        x = 2.0
        assert loaded_model(x) == original_model(x)
