"""test_parameters.py - tests for the modelling_lib.model.parameter module."""

import jax.numpy as jnp
import numpy as np
import pytest
from modelling_lib.model.parameter import (
    BoundsError,
    ConstrainedParameter,
    Parameter,
    init_parameter,
    l_bounded,
    l_bounded_inv,
    lu_bounded,
    lu_bounded_inv,
    softplus,
    softplus_frac_inv,
    softplus_inv,
    u_bounded,
    u_bounded_inv,
)


class TestParameter:
    def test_basic_initialization(self):
        # Test scalar initialization
        p = Parameter(initial=1.0)
        assert p.val.shape == (1,)
        assert jnp.allclose(p.val, 1.0)
        assert p.fix is False

        # Test array initialization
        arr = jnp.array([1.0, 2.0, 3.0])
        p = Parameter(initial=arr)
        assert p.val.shape == (3,)
        assert jnp.allclose(p.val, arr)

        # Test dimension specification
        p = Parameter(dims=5)
        assert p.val.shape == (5,)
        assert jnp.allclose(p.val, jnp.zeros(5))

        # Test fixed parameter
        p = Parameter(initial=1.0, fixed=True)
        assert p.fix is True

    def test_dimension_mismatch(self):
        # Test dimension mismatch
        with pytest.raises(ValueError):
            Parameter(dims=3, initial=jnp.array([1.0, 2.0]))

    def test_numpy_array_conversion(self):
        # Test numpy array gets converted to jax array
        arr = np.array([1.0, 2.0, 3.0])
        p = Parameter(initial=arr)
        assert isinstance(p.val, jnp.ndarray)

    def test_tuple_dimensions(self):
        # Test tuple dimensions
        p = Parameter(dims=(2, 3))
        assert p.val.shape == (2, 3)

        arr = jnp.ones((2, 3))
        p = Parameter(dims=(2, 3), initial=arr)
        assert p.val.shape == (2, 3)


class TestConstrainedParameter:
    def test_lower_bound(self):
        # Test initialization with value inside bounds
        c = ConstrainedParameter(initial=1.0, lower=0.0)
        assert jnp.all(c.val >= 0.0)
        assert jnp.allclose(c.val, 1.0)

        # Test initialization with value near bound - not exactly at bound
        # since that would require infinite unconstrained values
        c = ConstrainedParameter(initial=0.001, lower=0.0)
        assert jnp.all(c.val > 0.0)
        assert jnp.allclose(c.val, 0.001, atol=1e-3)

        # Test initialization with value outside bounds
        with pytest.raises(BoundsError):
            ConstrainedParameter(initial=-1.0, lower=0.0)

        # Test auto-initialization with zeros outside bounds
        with pytest.raises(BoundsError):
            ConstrainedParameter(lower=1.0)

    def test_upper_bound(self):
        # Test initialization with value inside bounds
        c = ConstrainedParameter(initial=0.0, upper=1.0)
        assert jnp.all(c.val <= 1.0)
        assert jnp.allclose(c.val, 0.0)

        # Test initialization with value near bound - not exactly at bound
        c = ConstrainedParameter(initial=0.999, upper=1.0)
        assert jnp.all(c.val < 1.0)
        assert jnp.allclose(c.val, 0.999, atol=1e-3)

        # Test initialization with value outside bounds
        with pytest.raises(BoundsError):
            ConstrainedParameter(initial=2.0, upper=1.0)

    def test_both_bounds(self):
        # Test initialization with value inside bounds
        c = ConstrainedParameter(initial=0.5, lower=0.0, upper=1.0)
        assert jnp.all((c.val >= 0.0) & (c.val <= 1.0))
        assert jnp.allclose(c.val, 0.5)

        # Test initialization with values close to bounds (not exactly at bounds)
        c = ConstrainedParameter(initial=0.001, lower=0.0, upper=1.0)
        assert jnp.all(c.val > 0.0)
        assert jnp.allclose(c.val, 0.001, atol=1e-3)

        c = ConstrainedParameter(initial=0.9, lower=0.0, upper=1.0)
        assert jnp.all(c.val < 1.0)
        assert jnp.allclose(c.val, 0.9, atol=1e-2)

        # Test initialization with value outside bounds
        with pytest.raises(BoundsError):
            ConstrainedParameter(initial=1.5, lower=0.0, upper=1.0)

        with pytest.raises(BoundsError):
            ConstrainedParameter(initial=-0.5, lower=0.0, upper=1.0)

    def test_no_bounds_error(self):
        # Test initialization with no bounds
        with pytest.raises(ValueError):
            ConstrainedParameter(initial=1.0)

    def test_array_values(self):
        # Test array values
        arr = jnp.array([0.1, 0.2, 0.3])
        c = ConstrainedParameter(initial=arr, lower=0.0, upper=1.0)
        assert jnp.all((c.val >= 0.0) & (c.val <= 1.0))
        assert jnp.allclose(c.val, arr)

        # Test array with values outside bounds
        with pytest.raises(BoundsError):
            ConstrainedParameter(initial=jnp.array([-0.1, 0.2, 0.3]), lower=0.0, upper=1.0)

    def test_fixed_parameter(self):
        # Test fixed parameter
        c = ConstrainedParameter(initial=0.5, lower=0.0, upper=1.0, fixed=True)
        assert c.fix is True
        assert jnp.allclose(c.val, 0.5)


class TestBoundFunctions:
    def test_softplus(self):
        x = jnp.array([-10.0, 0.0, 10.0])
        fx = softplus(x)
        assert jnp.all(fx > 0.0)  # Always positive
        assert jnp.allclose(fx, jnp.array([4.5399931e-05, 6.9314718e-01, 1.0000045e01]), atol=1e-6)

    def test_softplus_inv(self):
        fx = jnp.array([0.5, 1.0, 5.0])
        x = softplus_inv(fx)
        assert jnp.allclose(softplus(x), fx, atol=1e-6)

        # Test values <= 0 cause error
        with pytest.raises(ValueError):
            softplus_inv(jnp.array([0.0]))

        with pytest.raises(ValueError):
            softplus_inv(jnp.array([-1.0]))

    def test_softplus_frac_inv(self):
        # Test within allowed range
        x = jnp.array([0.1, 0.5, 0.9])
        y = softplus_frac_inv(x)
        assert y.shape == x.shape

        # Test clipping at boundaries - should be finite
        assert jnp.isfinite(softplus_frac_inv(jnp.array([0.0])))

        # For not too close to 1.0, result should be finite
        assert jnp.isfinite(softplus_frac_inv(jnp.array([0.95])))

    def test_l_bounded(self):
        x = jnp.array([-10.0, 0.0, 10.0])
        lower = 1.0
        fx = l_bounded(x, lower)
        assert jnp.all(fx >= lower)

    def test_l_bounded_inv(self):
        fx = jnp.array([1.5, 2.0, 10.0])
        lower = 1.0
        x = l_bounded_inv(fx, lower)
        recovered = l_bounded(x, lower)
        assert jnp.allclose(recovered, fx, atol=1e-6)

        # Test values below bound cause error
        with pytest.raises(BoundsError):
            l_bounded_inv(jnp.array([0.5]), lower)

    def test_u_bounded(self):
        x = jnp.array([-10.0, 0.0, 10.0])
        upper = 5.0
        fx = u_bounded(x, upper)
        assert jnp.all(fx <= upper)

    def test_u_bounded_inv(self):
        fx = jnp.array([1.0, 3.0, 4.9])
        upper = 5.0
        x = u_bounded_inv(fx, upper)
        recovered = u_bounded(x, upper)
        assert jnp.allclose(recovered, fx, atol=1e-6)

        # Test values above bound cause error
        with pytest.raises(BoundsError):
            u_bounded_inv(jnp.array([5.1]), upper)

    def test_lu_bounded(self):
        x = jnp.array([-10.0, 0.0, 10.0])
        lower, upper = 1.0, 5.0
        fx = lu_bounded(x, lower, upper)
        assert jnp.all((fx >= lower) & (fx <= upper))

    def test_lu_bounded_inv(self):
        fx = jnp.array([1.1, 3.0, 4.9])
        lower, upper = 1.0, 5.0
        x = lu_bounded_inv(fx, lower, upper)
        recovered = lu_bounded(x, lower, upper)
        assert jnp.allclose(recovered, fx, atol=1e-6)

        # Test values outside bounds cause error
        with pytest.raises(BoundsError):
            lu_bounded_inv(jnp.array([0.9]), lower, upper)

        with pytest.raises(BoundsError):
            lu_bounded_inv(jnp.array([5.1]), lower, upper)


def test_init_parameter():
    # Test with None
    p = init_parameter(None, initial=1.0)
    assert isinstance(p, Parameter)
    assert jnp.allclose(p.val, 1.0)

    # Test with existing parameter
    existing = Parameter(initial=2.0)
    p = init_parameter(existing, initial=1.0)
    assert p is existing
    assert jnp.allclose(p.val, 2.0)  # Should keep original value
