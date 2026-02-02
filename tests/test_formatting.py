"""test_formatting.py - smoke tests for the spectracles.model.formatting module."""

import jax.numpy as jnp
import pytest
from spectracles.model.formatting import (
    format_array_value,
    format_constraint,
    get_console,
    render_to_string,
    styled_type,
    styled_value,
    styled_free,
    styled_fixed,
    styled_shared,
    styled_dim,
    parameter_repr_parts,
    known_repr_parts,
    shared_repr_parts,
    create_gradient_table,
    create_model_tree,
    add_tree_node,
)


class TestFormatArrayValue:
    def test_scalar(self):
        # Single element array
        arr = jnp.array([1.5])
        result = format_array_value(arr)
        assert "1.5" in result

    def test_small_array(self):
        # Small array shows values
        arr = jnp.array([1.0, 2.0, 3.0])
        result = format_array_value(arr)
        assert "[" in result
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_large_array(self):
        # Large array shows dtype and shape
        arr = jnp.ones((10, 10))
        result = format_array_value(arr)
        assert "10" in result  # shape
        assert "f" in result.lower()  # dtype (float)

    def test_none(self):
        result = format_array_value(None)
        assert result == "None"


class TestFormatConstraint:
    def test_log_constraint(self):
        result = format_constraint(None, None, log=True)
        assert result == "log"

    def test_lower_bound(self):
        result = format_constraint(lower=0.0, upper=None, log=False)
        assert ">" in result
        assert "0" in result

    def test_upper_bound(self):
        result = format_constraint(lower=None, upper=10.0, log=False)
        assert "<" in result
        assert "10" in result

    def test_both_bounds(self):
        result = format_constraint(lower=0.0, upper=10.0, log=False)
        assert "0" in result
        assert "10" in result
        assert "<" in result

    def test_no_bounds(self):
        result = format_constraint(lower=None, upper=None, log=False)
        assert result == ""


class TestStyledFunctions:
    def test_styled_type(self):
        text = styled_type("Parameter")
        assert "Parameter" in str(text)

    def test_styled_value(self):
        text = styled_value("1.5")
        assert "1.5" in str(text)

    def test_styled_free(self):
        text = styled_free()
        assert "free" in str(text)

    def test_styled_fixed(self):
        text = styled_fixed()
        assert "fixed" in str(text)

    def test_styled_shared(self):
        text = styled_shared("a.val")
        assert "a.val" in str(text)

    def test_styled_dim(self):
        text = styled_dim("(")
        assert "(" in str(text)


class TestReprParts:
    def test_parameter_repr_parts_free(self):
        text = parameter_repr_parts("Parameter", "1.5", is_fixed=False)
        result = str(text)
        assert "Parameter" in result
        assert "1.5" in result
        assert "free" in result

    def test_parameter_repr_parts_fixed(self):
        text = parameter_repr_parts("Parameter", "1.5", is_fixed=True)
        result = str(text)
        assert "fixed" in result

    def test_parameter_repr_parts_with_constraint(self):
        text = parameter_repr_parts(
            "ConstrainedParameter", "1.5", is_fixed=False, constraint_str="x > 0"
        )
        result = str(text)
        assert "ConstrainedParameter" in result
        assert "x > 0" in result

    def test_parameter_repr_parts_with_shared_path(self):
        text = parameter_repr_parts(
            "Parameter", "", is_fixed=False, shared_path="a.val"
        )
        result = str(text)
        assert "a.val" in result

    def test_known_repr_parts(self):
        text = known_repr_parts("2.5")
        result = str(text)
        assert "Known" in result
        assert "2.5" in result

    def test_shared_repr_parts(self):
        text = shared_repr_parts("parent.param.val")
        result = str(text)
        assert "Shared" in result
        assert "parent.param.val" in result


class TestConsoleAndRender:
    def test_get_console(self):
        console = get_console()
        assert console is not None

    def test_render_to_string(self):
        text = styled_type("Test")
        result = render_to_string(text)
        assert isinstance(result, str)
        assert "Test" in result


class TestTableAndTree:
    def test_create_gradient_table(self):
        table = create_gradient_table()
        assert table is not None
        assert table.title == "Gradient Summary"

    def test_create_model_tree(self):
        tree = create_model_tree("model", "SimpleModel")
        assert tree is not None

    def test_add_tree_node_with_string(self):
        tree = create_model_tree("model", "SimpleModel")
        node = add_tree_node(tree, "param", "Parameter(1.5)")
        assert node is not None

    def test_add_tree_node_with_text(self):
        tree = create_model_tree("model", "SimpleModel")
        text = styled_type("Parameter")
        node = add_tree_node(tree, "param", text)
        assert node is not None
