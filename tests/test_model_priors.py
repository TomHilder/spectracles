"""test_model_priors.py - Priors on parameters and models (proposal_tests.md section B)."""

import equinox as eqx
import jax
import pytest

from spectracles import ConstrainedParameter, Known, Parameter, ShareModule
from spectracles.priors import LogNormal, Normal, Prior


class TwoParams(eqx.Module):
    a: Parameter
    b: Parameter

    def __call__(self):
        return self.a.val + self.b.val


# ---- B1: storing the prior ----


def test_prior_is_stored_and_static():
    prior = LogNormal(1.0, 0.5)
    param = Parameter(1.0, prior=prior)
    assert param.prior is prior
    assert not any(isinstance(leaf, Prior) for leaf in jax.tree.leaves(param))
    # Hyperparameters are not array leaves either
    assert len(jax.tree.leaves(param)) == len(jax.tree.leaves(Parameter(1.0)))


def test_default_is_no_prior():
    assert Parameter(1.0).prior is None
    assert Known(1.0).prior is None


def test_prior_must_be_a_prior():
    with pytest.raises(TypeError):
        Parameter(1.0, prior="LogNormal")  # type: ignore[arg-type]


# ---- B2: forbidden combinations ----


def test_known_cannot_have_prior():
    with pytest.raises(ValueError, match="Known"):
        Known(1.0, prior=Normal(0.0, 1.0))


def test_constrained_parameter_cannot_have_prior():
    with pytest.raises(ValueError, match="prior"):
        ConstrainedParameter(1.0, lower=0.0, prior=LogNormal(1.0, 0.5))


# ---- B3: sharing detection isn't fooled by a shared Prior instance ----


def test_shared_prior_instance_is_not_parameter_sharing():
    prior = Normal(0.0, 1.0)
    model = ShareModule(TwoParams(a=Parameter(1.0, prior=prior), b=Parameter(2.0, prior=prior)))
    assert model.get_sharing_summary(level="parameter") == {}
    assert model.get_parameter_paths() == ["a", "b"]
    assert model() == 3.0


def test_shared_parameter_still_detected_with_prior():
    shared = Parameter(1.0, prior=Normal(0.0, 1.0))
    model = ShareModule(TwoParams(a=shared, b=shared))
    assert model.get_sharing_summary(level="parameter") == {"a": ["b"]}


# ---- repr (part of C3) ----


def test_repr_includes_prior():
    assert "~LogNormal(0.3, 0.15)" in repr(Parameter(0.3, prior=LogNormal(0.3, 0.15)))
    assert "~" not in repr(Parameter(0.3))
