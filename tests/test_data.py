"""test_data.py - tests for the modelling_lib.model.data module."""

import jax.numpy as jnp
import numpy as np
from modelling_lib.model.data import SpatialDataGeneric, convert_to_flat_array


class TestConvertToFlatArray:
    def test_scalar(self):
        result = convert_to_flat_array(1.0)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (1,)
        assert jnp.allclose(result, jnp.array([1.0]))

    def test_1d_array(self):
        # Test with jax array
        arr = jnp.array([1.0, 2.0, 3.0])
        result = convert_to_flat_array(arr)
        assert result.shape == (3,)
        assert jnp.allclose(result, arr)

        # Test with numpy array
        arr = np.array([1.0, 2.0, 3.0])
        result = convert_to_flat_array(arr)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (3,)
        assert jnp.allclose(result, arr)

        # Test with list
        arr = [1.0, 2.0, 3.0]
        result = convert_to_flat_array(arr)
        assert result.shape == (3,)
        assert jnp.allclose(result, jnp.array(arr))

    def test_2d_array(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = convert_to_flat_array(arr)
        assert result.shape == (4,)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0, 4.0]))

    def test_3d_array(self):
        arr = jnp.ones((2, 2, 2))
        result = convert_to_flat_array(arr)
        assert result.shape == (8,)
        assert jnp.allclose(result, jnp.ones(8))


class TestSpatialData:
    def test_initialization(self):
        # Test basic initialization
        data = SpatialDataGeneric(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]), idx=jnp.array([0, 1])
        )
        assert jnp.allclose(data.x, jnp.array([1.0, 2.0]))
        assert jnp.allclose(data.y, jnp.array([3.0, 4.0]))
        assert jnp.allclose(data.idx, jnp.array([0, 1]))

    def test_auto_conversion(self):
        # Test with lists
        data = SpatialDataGeneric(x=[1.0, 2.0], y=[3.0, 4.0], idx=[0, 1])
        assert data.x.shape == (2,)
        assert data.y.shape == (2,)
        assert data.idx.shape == (2,)

        # Test with numpy arrays
        data = SpatialDataGeneric(
            x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), idx=np.array([0, 1])
        )
        assert isinstance(data.x, jnp.ndarray)
        assert isinstance(data.y, jnp.ndarray)
        assert isinstance(data.idx, jnp.ndarray)

        # Test with 2D arrays
        data = SpatialDataGeneric(
            x=np.array([[1.0], [2.0]]), y=np.array([[3.0], [4.0]]), idx=np.array([[0], [1]])
        )
        assert data.x.shape == (2,)
        assert data.y.shape == (2,)
        assert data.idx.shape == (2,)

    def test_empty_data(self):
        # Test with empty arrays
        data = SpatialDataGeneric(x=jnp.array([]), y=jnp.array([]), idx=jnp.array([]))
        assert data.x.shape == (0,)
        assert data.y.shape == (0,)
        assert data.idx.shape == (0,)

    def test_dimension_mismatch_handling(self):
        # The module doesn't explicitly check for dimension matching between x, y, and idx
        # This is more of a "what happens" test than a correctness test
        data = SpatialDataGeneric(
            x=jnp.array([1.0, 2.0]), y=jnp.array([3.0]), idx=jnp.array([0, 1, 2])
        )
        assert data.x.shape == (2,)
        assert data.y.shape == (1,)
        assert data.idx.shape == (3,)
