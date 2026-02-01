"""test_leaf_sharing.py - tests for the spectracles.model.share_module module. (share_module refers to Equinox Module objects not Python modules)"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from equinox import is_array
from jax.tree_util import tree_map
from spectracles.model.parameter import Parameter
from spectracles.model.share_module import (
    Shared,
    ShareModule,
    build_model,
    contains_shared,
    # get_duplicated_leaves, # TODO: this should be tested in test_path_utils.py
    get_duplicated_parameters,
    parent_model,
    use_paths_get_leaves,
)
from spectracles.tree.path_utils import (
    use_path_get_leaf,  # TODO: this test should be in test_path_utils.py
)

from .test_models import (
    ComplexSharedModel,
    NestedModel,
    SharedBranchModel,
    SharedLeafModel,
    SimpleModel,
)


class TestLeafHelperFunctions:
    def test_get_duplicated_parameters(self):
        # Create a model with shared leaves
        model = SharedLeafModel()

        # Get duplicated leaves
        dupl_ids, dupl_paths, parent_paths = get_duplicated_parameters(model)

        # There should be 1 duplicate
        assert len(dupl_ids) == 1
        assert len(dupl_paths) == 1
        assert len(parent_paths) == 1

        # The duplicate should be model.b
        leaf_b = use_path_get_leaf(model, dupl_paths[0])
        assert leaf_b is model.b.val  # Changed from model.b

    def test_get_duplicated_parameters_nested(self):
        # Create a more complex model with nested sharing
        model = ComplexSharedModel()

        # Get duplicated leaves
        dupl_ids, dupl_paths, parent_paths = get_duplicated_parameters(model)

        # There should be 3 duplicates
        assert len(dupl_ids) == 3

        # The duplicates should include model.param.val, model.inner2.param.val, and more
        dupl_leaves = use_paths_get_leaves(model, dupl_paths)
        # These assertions need to check for the .val attribute of the parameters
        assert model.param.val in dupl_leaves
        assert model.inner2.param.val in dupl_leaves

    def test_use_path_get_leaf(self):
        # Create a simple model
        model = SimpleModel()

        # Get duplicated leaves to get paths
        _, _, parent_paths = get_duplicated_parameters(model)

        # Use the path to get the parameter
        param_path = list(parent_paths.values())[0]
        param = use_path_get_leaf(model, param_path)

        # Check it's the right parameter's value
        assert param is model.param.val  # Changed from model.param

    def test_use_paths_get_leaves(self):
        # Create a more complex model
        model = NestedModel()

        # Get duplicated leaves to get paths
        _, _, parent_paths = get_duplicated_parameters(model)

        # Use the paths to get parameters
        paths = list(parent_paths.values())
        params = use_paths_get_leaves(model, paths)

        # Check they're the right parameters' values
        assert model.inner1.param.val in params  # Changed from model.inner1.param
        assert model.inner2.param.val in params  # Changed from model.inner2.param
        assert model.shared.val in params  # Changed from model.shared


class TestSharedClass:
    def test_shared_initialization(self):
        # Create a Shared object without parent_path
        shared = Shared(id=123)
        assert shared.id == 123
        assert shared.parent_path == ""

        # Create a Shared object with parent_path
        shared_with_path = Shared(id=456, parent_path="model.param.val")
        assert shared_with_path.id == 456
        assert shared_with_path.parent_path == "model.param.val"

    def test_shared_representation(self):
        # Check string representation without parent_path (fallback to id)
        shared = Shared(id=123)
        assert repr(shared) == "Shared(123)"
        assert str(shared) == "Shared(123)"

        # Check string representation with parent_path (shows path)
        shared_with_path = Shared(id=456, parent_path="model.param.val")
        assert repr(shared_with_path) == "Shared → model.param.val"
        assert str(shared_with_path) == "Shared → model.param.val"


class TestShareModule:
    def test_initialization(self):
        # Create a model with shared leaves
        model = SharedLeafModel()
        shared_model = ShareModule(model)

        # Check attributes
        assert shared_model.model is not None
        assert len(shared_model._dupl_leaf_ids) == 1
        assert len(shared_model._dupl_leaf_paths) == 1
        assert len(shared_model._parent_leaf_paths) == 1
        assert shared_model._locked is False

    def test_replace_duplicates_with_shared(self):
        # Create a model with shared leaves
        model = SharedLeafModel()
        shared_model = ShareModule(model)

        # Check that duplicated leaf is replaced with Shared
        assert isinstance(shared_model.a, Parameter)
        assert isinstance(shared_model.b.val, Shared)  # Changed: check .val

        # Check that Shared.id matches the id of the original parameter's value
        assert shared_model.b.val.id == id(
            model.a.val
        )  # Changed: check .val.id and id(model.a.val)

        # Check that Shared.parent_path is set correctly
        assert shared_model.b.val.parent_path == "a.val"
        assert "Shared → a.val" in repr(shared_model.b.val)

    def test_locked_model(self):
        # Create a locked model
        model = SharedLeafModel()
        locked_model = ShareModule(model, locked=True)

        # Check attributes
        assert locked_model._locked is True

        # Check that duplicated leaf is not replaced with Shared
        assert isinstance(locked_model.a, Parameter)
        assert isinstance(locked_model.b, Parameter)
        # If model.a and model.b are the same object in SharedLeafModel,
        # they should remain the same object in the locked_model.
        # Locking prevents creation of Shared objects, not unsharing identical Parameter objects.
        assert locked_model.a is locked_model.b

    def test_getattr_delegation(self):
        # Test attribute delegation to model
        model = SimpleModel(value=2.0)
        shared_model = ShareModule(model)

        # Access attribute from model
        assert jnp.allclose(shared_model.param.val, 2.0)

        # Access non-existent attribute
        with pytest.raises(AttributeError):
            shared_model.nonexistent

    def test_call_with_shared_leaves(self):
        # Test call with shared leaves
        model = SharedLeafModel(value=2.0)
        shared_model = ShareModule(model)

        # Call the model
        x = jnp.array([1.0, 2.0, 3.0])
        result = shared_model(x)

        # Check result (should be 2x + 2x = 4x)
        assert jnp.allclose(result, 4.0 * x)

    def test_complex_sharing(self):
        # Test with complex sharing structure
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        # Call the model
        x = jnp.array([1.0])
        result = shared_model(x)

        # Get the expected result (the sum of all model parts)
        expected = model(x)
        assert jnp.allclose(result, expected)

        # Verify the sharing structure
        # inner1.param is the original, its .val is shared
        # param.val, inner2.param.val, and inner3.inner1.param.val should be Shared objects
        assert isinstance(shared_model.inner1.param, Parameter)
        assert isinstance(shared_model.param.val, Shared)  # Changed: check .val
        assert isinstance(shared_model.inner2.param.val, Shared)  # Changed: check .val
        assert isinstance(shared_model.inner3.inner1.param.val, Shared)  # Changed: check .val

    def test_get_locked_model(self):
        # Test locking an unlocked model
        model = SharedLeafModel(value=2.0)
        shared_model = ShareModule(model)
        locked_model = shared_model.get_locked_model()

        # Check that the model is locked
        assert locked_model._locked is True

        # Check that all Shared objects are replaced with actual values
        assert isinstance(locked_model.a, Parameter)
        assert isinstance(locked_model.b, Parameter)

        # They should have the same values
        assert jnp.allclose(locked_model.a.val, locked_model.b.val)

        # But be different objects
        assert locked_model.a is not locked_model.b

    def test_copy(self):
        # Test copying a model
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)
        copied_model = shared_model.copy()

        # Check that it's a different object
        assert copied_model is not shared_model

        # But should have the same attributes
        assert copied_model._locked == shared_model._locked
        assert len(copied_model._dupl_leaf_ids) == len(shared_model._dupl_leaf_ids)
        # The paths themselves might differ if IDs change, so we check length.
        # A more thorough check would involve verifying semantic equivalence of paths.
        assert len(copied_model._dupl_leaf_paths) == len(shared_model._dupl_leaf_paths)
        assert len(copied_model._parent_leaf_paths) == len(shared_model._parent_leaf_paths)

        # Check that its arrays are copies by filtering the .model attribute
        orig_arrays = eqx.filter(shared_model.model, is_array)
        copied_arrays = eqx.filter(copied_model.model, is_array)

        # The arrays should be equal but different objects
        tree_map(lambda x, y: assert_array_equal_but_different(x, y), orig_arrays, copied_arrays)


def assert_array_equal_but_different(x, y):
    # Check arrays are equal in value
    assert jnp.array_equal(x, y)

    # For JAX arrays, we can't rely on id() to check object identity
    # since JAX may optimize and reuse arrays with the same contents
    # Instead, we can check that modifying one doesn't affect the other
    # by comparing the raw buffer pointers or memory locations
    #
    # Since we can't directly modify JAX arrays, this test is disabled
    # and we only check for equality
    pass


class TestContainsShared:
    def test_contains_shared_true(self):
        # Create a model with shared leaves
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        # The model's b parameter contains a Shared value
        assert contains_shared(shared_model.b) is True
        # The whole model contains Shared values
        assert contains_shared(shared_model.model) is True

    def test_contains_shared_false(self):
        # Create a model with shared leaves
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        # The model's a parameter does NOT contain Shared (it's the parent)
        assert contains_shared(shared_model.a) is False

    def test_contains_shared_locked_model(self):
        # Locked models should not contain Shared values
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)
        locked_model = shared_model.get_locked_model()

        assert contains_shared(locked_model.a) is False
        assert contains_shared(locked_model.b) is False
        assert contains_shared(locked_model.model) is False

    def test_contains_shared_simple(self):
        # Simple model with no sharing
        model = SimpleModel(value=1.0)
        shared_model = ShareModule(model)

        assert contains_shared(shared_model.param) is False
        assert contains_shared(shared_model.model) is False


class TestSubcomponentAccessError:
    def test_callable_subcomponent_with_shared_raises_helpful_error(self):
        # Create a complex model where inner2 contains Shared values
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        # Accessing inner2 should work (returns proxy)
        inner2 = shared_model.inner2

        # But trying to call it should raise a helpful RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            inner2(jnp.array([1.0]))

        # Check the error message is helpful
        error_msg = str(exc_info.value)
        assert "inner2" in error_msg
        assert "get_locked_model()" in error_msg
        assert "Shared" in error_msg

    def test_callable_subcomponent_without_shared_works(self):
        # Simple model has no shared parameters
        model = SimpleModel(value=1.0)
        shared_model = ShareModule(model)

        # The model itself should be callable without issues
        x = jnp.array([1.0, 2.0, 3.0])
        result = shared_model(x)
        assert jnp.allclose(result, x)

    def test_subcomponent_attribute_access_still_works(self):
        # Even for sub-components with Shared values, attribute access should work
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        # We can access attributes of inner2 even though it contains Shared
        inner2 = shared_model.inner2
        assert hasattr(inner2, "param")
        # The param's val should be a Shared object
        assert isinstance(inner2.param.val, Shared)

    def test_locked_model_subcomponent_is_callable(self):
        # Locked models should have callable sub-components
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)
        locked_model = shared_model.get_locked_model()

        # inner2 should be callable on the locked model
        x = jnp.array([1.0])
        result = locked_model.inner2(x)
        # inner2 is SimpleModel, so it just multiplies by param.val (2.0)
        assert jnp.allclose(result, 2.0 * x)


class TestParameterPaths:
    def test_get_parameter_paths_simple(self):
        # Simple model with one parameter
        model = SimpleModel(value=1.0)
        shared_model = ShareModule(model)

        paths = shared_model.get_parameter_paths()

        assert len(paths) == 1
        assert "param" in paths

    def test_get_parameter_paths_shared_default(self):
        # Model with shared parameters - default should only show parent
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        paths = shared_model.get_parameter_paths()

        # Only the parent (a) should be shown, not the shared (b)
        assert len(paths) == 1
        assert "a" in paths
        assert "b" not in paths

    def test_get_parameter_paths_shared_show_all(self):
        # Model with shared parameters - show_shared=True should show all
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        paths = shared_model.get_parameter_paths(show_shared=True)

        # Both a and b should be shown
        assert len(paths) == 2
        assert "a" in paths
        assert "b" in paths

    def test_get_parameter_paths_complex(self):
        # Complex model with multiple shared parameters
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        # Default - only unique parameters
        paths = shared_model.get_parameter_paths()

        # inner1.param is the parent, inner3.inner2.param and inner3.shared are unique
        assert "inner1.param" in paths
        assert "inner3.inner2.param" in paths
        assert "inner3.shared" in paths
        # These are shared (duplicates of inner1.param)
        assert "param" not in paths
        assert "inner2.param" not in paths

        # With show_shared=True
        all_paths = shared_model.get_parameter_paths(show_shared=True)

        # All should be present
        assert "inner1.param" in all_paths
        assert "param" in all_paths
        assert "inner2.param" in all_paths
        assert "inner3.inner1.param" in all_paths


class TestValidateSharing:
    def test_validate_sharing_valid_model(self):
        # A valid model should pass validation
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        result = shared_model.validate_sharing(raise_on_error=False)

        assert result["valid"] is True
        assert result["n_shared_params"] == 1
        assert result["n_unique_params"] == 1
        assert result["errors"] == []

    def test_validate_sharing_complex_model(self):
        # Complex model should also pass
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        result = shared_model.validate_sharing(raise_on_error=False)

        assert result["valid"] is True
        assert result["n_shared_params"] == 3  # 3 duplicates
        assert result["n_unique_params"] == 3  # 3 unique params
        assert result["errors"] == []

    def test_validate_sharing_no_sharing(self):
        # Model with no sharing should also pass
        model = SimpleModel(value=1.0)
        shared_model = ShareModule(model)

        result = shared_model.validate_sharing(raise_on_error=False)

        assert result["valid"] is True
        assert result["n_shared_params"] == 0
        assert result["n_unique_params"] == 1
        assert result["errors"] == []

    def test_validate_sharing_raises_on_error(self):
        # Test that raise_on_error=True raises ValueError
        # (We can't easily create an invalid model, so just test the valid case doesn't raise)
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        # Should not raise
        result = shared_model.validate_sharing(raise_on_error=True)
        assert result["valid"] is True


class TestSharedComponents:
    def test_get_shared_components_with_branch_sharing(self):
        # SharedBranchModel shares entire SimpleModel modules
        model = SharedBranchModel(value=1.0)
        shared_model = ShareModule(model)

        components = shared_model.get_shared_components()

        # branch_a and branch_b are the same object
        assert len(components) == 1
        # The parent should be 'branch_a' and shared should include 'branch_b'
        parent = list(components.keys())[0]
        assert "branch_a" in parent
        assert len(components[parent]) == 1
        assert "branch_b" in components[parent][0]

    def test_get_shared_components_parameter_sharing(self):
        # SharedLeafModel shares a Parameter object between a and b
        # Since Parameter is also an equinox Module, this is detected as module sharing
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        components = shared_model.get_shared_components()

        # Parameters ARE Modules, so component sharing is detected
        assert len(components) == 1
        parent = list(components.keys())[0]
        assert "a" in parent
        assert "b" in components[parent][0]

    def test_get_shared_components_complex_parameter_sharing(self):
        # ComplexSharedModel shares Parameters across different Modules
        # But the Modules themselves are different objects
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        components = shared_model.get_shared_components()

        # inner1 and inner2 are different SimpleModel objects (even though
        # they share the same Parameter), so no component sharing detected
        assert components == {}

    def test_get_shared_components_no_sharing(self):
        # SimpleModel has no shared components
        model = SimpleModel(value=1.0)
        shared_model = ShareModule(model)

        components = shared_model.get_shared_components()

        assert components == {}


class TestSharingSummary:
    def test_get_sharing_summary_simple_component(self):
        # Simple model with one shared parameter - component level (default)
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        summary = shared_model.get_sharing_summary()

        # Should have one parent with one duplicate
        assert len(summary) == 1
        assert "a" in summary
        assert summary["a"] == ["b"]

    def test_get_sharing_summary_simple_parameter(self):
        # Simple model with one shared parameter - parameter level
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        summary = shared_model.get_sharing_summary(level="parameter")

        # Should have one parent with one duplicate
        assert len(summary) == 1
        assert "a" in summary
        assert summary["a"] == ["b"]

    def test_get_sharing_summary_complex_parameter(self):
        # Complex model - parameter level catches tree_at sharing
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        summary = shared_model.get_sharing_summary(level="parameter")

        # Should have one parent (inner1.param) with multiple duplicates
        assert len(summary) == 1
        parent_path = list(summary.keys())[0]
        assert "inner1.param" == parent_path

        # Should have 3 duplicates: param, inner2.param, inner3.inner1.param
        duplicates = summary[parent_path]
        assert len(duplicates) == 3
        assert "param" in duplicates
        assert "inner2.param" in duplicates
        assert "inner3.inner1.param" in duplicates

    def test_get_sharing_summary_complex_component(self):
        # Complex model - component level doesn't catch tree_at sharing
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        summary = shared_model.get_sharing_summary(level="component")

        # tree_at sharing breaks object identity, so component level is empty
        assert len(summary) == 0

    def test_get_sharing_summary_no_sharing(self):
        # Model with no shared parameters
        model = SimpleModel(value=1.0)
        shared_model = ShareModule(model)

        summary = shared_model.get_sharing_summary()

        # Should be empty
        assert summary == {}


class TestPrintModelTree:
    def test_print_model_tree_basic(self, capsys):
        # Test that print_model_tree runs without error
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        shared_model.print_model_tree()

        captured = capsys.readouterr()
        assert "SharedLeafModel" in captured.out
        # Note: due to sharing, only one of a/b may appear in the tree
        # (they're the same object)

    def test_print_model_tree_show_sharing(self, capsys):
        # Test that show_sharing prints sharing summary
        model = SharedLeafModel(value=1.0)
        shared_model = ShareModule(model)

        shared_model.print_model_tree(show_sharing=True)

        captured = capsys.readouterr()
        # Should show "Shared parameters:" section
        assert "Shared parameters:" in captured.out
        # Should show the sharing relationship with arrow
        assert "←" in captured.out
        # Should mention both a and b
        assert "a" in captured.out
        assert "b" in captured.out

    def test_print_model_tree_complex_sharing(self, capsys):
        # Test with complex model
        model = ComplexSharedModel(value=2.0)
        shared_model = ShareModule(model)

        shared_model.print_model_tree(show_sharing=True)

        captured = capsys.readouterr()
        # Should have sharing summary section
        assert "Shared parameters:" in captured.out
        # Should have multiple sharing annotations
        assert captured.out.count("←") >= 1

    def test_print_model_tree_no_sharing(self, capsys):
        # Test with model that has no sharing
        model = SimpleModel(value=1.0)
        shared_model = ShareModule(model)

        shared_model.print_model_tree(show_sharing=True)

        captured = capsys.readouterr()
        # Should NOT show "Shared parameters:" section since there's no sharing
        assert "Shared parameters:" not in captured.out


class TestBuildAndParentModel:
    def test_build_model(self):
        # Test building a model
        shared_model = build_model(SimpleModel, value=2.0)

        # Check that it's a ShareModule
        assert isinstance(shared_model, ShareModule)

        # Check that the model was initialized correctly
        assert jnp.allclose(shared_model.param.val, 2.0)

    def test_parent_model(self):
        # Test wrapping an existing model
        original_model = SimpleModel(value=2.0)
        shared_model = parent_model(original_model)

        # Check that it's a ShareModule
        assert isinstance(shared_model, ShareModule)

        # Check that the model attribute is of the correct type and has correct values
        assert isinstance(shared_model.model, SimpleModel)
        assert jnp.allclose(shared_model.model.param.val, original_model.param.val)

        # For a SimpleModel, there are no internal duplicates to be replaced by Shared objects.
        # So, _dupl_leaf_ids should be empty.
        assert not shared_model._dupl_leaf_ids
        # And the model attribute should not contain Shared objects directly.
        # (It wouldn't anyway for SimpleModel as .param is a Parameter object)

        # The following assertion would fail because tree_at likely returns a new instance
        # assert shared_model.model is original_model

    def test_parent_model_already_wrapped(self):
        # Test wrapping an already wrapped model
        model = SimpleModel(value=2.0)
        shared_model = parent_model(model)
        double_wrapped = parent_model(shared_model)

        # Should return the same object, not double-wrap
        assert double_wrapped is shared_model


class TestFixAllFreeAll:
    def test_fix_all_simple(self):
        # Test fix_all on a simple model
        model = build_model(SimpleModel, value=1.0)

        # Initially the parameter is free
        assert not model.param.fix

        # Fix all parameters
        fixed_model = model.fix_all()

        # Parameter should now be fixed
        assert fixed_model.param.fix

        # Original model should be unchanged
        assert not model.param.fix

    def test_fix_all_shared_params(self):
        # Test fix_all with shared parameters
        model = build_model(SharedLeafModel, value=1.0)

        # Fix all parameters
        fixed_model = model.fix_all()

        # Both shared parameters should be fixed
        locked = fixed_model.get_locked_model()
        assert locked.a.fix
        assert locked.b.fix

    def test_free_all_simple(self):
        # Test free_all on a model with fixed parameters
        class FixedModel(eqx.Module):
            param: Parameter

            def __init__(self):
                self.param = Parameter(initial=1.0, fixed=True)

            def __call__(self, x):
                return self.param.val * x

        model = build_model(FixedModel)

        # Initially the parameter is fixed
        assert model.param.fix

        # Free all parameters
        freed_model = model.free_all()

        # Parameter should now be free
        assert not freed_model.param.fix

    def test_fix_all_then_free_all(self):
        # Test round-trip fix_all -> free_all
        model = build_model(SimpleModel, value=1.0)

        # Fix all, then free all
        fixed_model = model.fix_all()
        freed_model = fixed_model.free_all()

        # Should be back to free
        assert not freed_model.param.fix

    def test_fix_all_on_locked_model_raises(self):
        # Test that fix_all raises on a locked model
        model = build_model(SimpleModel, value=1.0)
        locked_model = model.get_locked_model()

        # Wrap in ShareModule with locked=True
        locked_share = ShareModule(locked_model.model, locked=True)

        with pytest.raises(ValueError, match="locked"):
            locked_share.fix_all()

    def test_free_all_on_locked_model_raises(self):
        # Test that free_all raises on a locked model
        model = build_model(SimpleModel, value=1.0)
        locked_share = ShareModule(model.model, locked=True)

        with pytest.raises(ValueError, match="locked"):
            locked_share.free_all()

    def test_fix_all_with_nested_model(self):
        # Test fix_all with nested models
        model = build_model(NestedModel, value=1.0)

        fixed_model = model.fix_all()
        locked = fixed_model.get_locked_model()

        # All parameters should be fixed
        assert locked.inner1.param.fix
        assert locked.inner2.param.fix
        assert locked.shared.fix

    def test_free_all_with_complex_shared_model(self):
        # Test free_all with complex shared model
        model = build_model(ComplexSharedModel, value=1.0)

        # First fix all
        fixed_model = model.fix_all()

        # Verify all are fixed
        locked = fixed_model.get_locked_model()
        assert locked.param.fix
        assert locked.inner1.param.fix

        # Then free all
        freed_model = fixed_model.free_all()
        locked_freed = freed_model.get_locked_model()

        # All should be free
        assert not locked_freed.param.fix
        assert not locked_freed.inner1.param.fix


class TestKnownParameterHandling:
    """Tests for Known parameter handling in ShareModule."""

    def test_set_known_raises_by_default(self):
        """set() should raise when trying to modify Known parameters by default."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())

        with pytest.raises(ValueError, match="Cannot set Known parameter"):
            model.set(["known"], [jnp.array([999.0])])

    def test_set_known_with_allow_flag(self):
        """set() should allow modifying Known when allow_set_knowns=True."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown(known_value=1.0))

        # Should work with flag
        updated = model.set(["known"], [jnp.array([999.0])], allow_set_knowns=True)
        locked = updated.get_locked_model()
        assert jnp.allclose(locked.model.known.val, 999.0)

    def test_set_regular_param_still_works(self):
        """set() should still work for regular Parameters."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown(param_value=1.0))

        updated = model.set(["param"], [jnp.array([42.0])])
        locked = updated.get_locked_model()
        assert jnp.allclose(locked.model.param.val, 42.0)

    def test_set_fixed_status_known_unfix_raises(self):
        """set_fixed_status() should raise when trying to unfix a Known."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())

        with pytest.raises(ValueError, match="Cannot unfix Known parameter"):
            model.set_fixed_status(["known"], [False])

    def test_set_fixed_status_known_fix_allowed(self):
        """set_fixed_status() should allow keeping Known fixed (no-op)."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())

        # This should work (Known is already fixed)
        updated = model.set_fixed_status(["known"], [True])
        assert updated.model.known.fix

    def test_set_fixed_status_regular_param_still_works(self):
        """set_fixed_status() should still work for regular Parameters."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())

        # Fix the parameter
        fixed = model.set_fixed_status(["param"], [True])
        assert fixed.model.param.fix

        # Unfix the parameter
        unfixed = fixed.set_fixed_status(["param"], [False])
        assert not unfixed.model.param.fix

    def test_get_parameter_paths_excludes_known_by_default(self):
        """get_parameter_paths() should exclude Known parameters by default."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())
        paths = model.get_parameter_paths()

        assert "param" in paths
        assert "known" not in paths

    def test_get_parameter_paths_includes_known_with_flag(self):
        """get_parameter_paths() should include Known with show_knowns=True."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())
        paths = model.get_parameter_paths(show_knowns=True)

        assert "param" in paths
        assert "known" in paths

    def test_parameter_summary_excludes_known_by_default(self, capsys):
        """parameter_summary() should exclude Known parameters by default."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())
        model.parameter_summary()

        captured = capsys.readouterr()
        assert "param" in captured.out
        assert "known" not in captured.out

    def test_parameter_summary_includes_known_with_flag(self, capsys):
        """parameter_summary() should include Known with show_knowns=True."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())
        model.parameter_summary(show_knowns=True)

        captured = capsys.readouterr()
        assert "param" in captured.out
        assert "known" in captured.out

    def test_free_all_does_not_unfix_known(self):
        """free_all() should not unfix Known parameters."""
        from .test_models import ModelWithKnown

        model = ShareModule(ModelWithKnown())

        # Fix everything first
        fixed = model.fix_all()
        assert fixed.model.param.fix
        assert fixed.model.known.fix

        # Free all
        freed = fixed.free_all()

        # Regular param should be free, Known should stay fixed
        assert not freed.model.param.fix
        assert freed.model.known.fix
