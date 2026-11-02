"""test_spatial_data.py - SpatialData base class and lvm_models smoke test (proposal_tests.md H)."""

import importlib
import pkgutil

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

import spectracles.lvm_models
from spectracles import (
    FourierGP,
    Known,
    Matern32,
    Parameter,
    PerSpaxel,
    SpatialData,
    SpatialDataGeneric,
    SpatialDataLVM,
    build_model,
)
from spectracles.model.data import convert_to_flat_array
from spectracles.model.spectral import Gaussian


class MySpatialData(SpatialData):
    x: Array = eqx.field(converter=convert_to_flat_array)
    y: Array = eqx.field(converter=convert_to_flat_array)
    idx: Array = eqx.field(converter=convert_to_flat_array)
    exposure_idx: Array = eqx.field(converter=convert_to_flat_array)


def make_my_data(n=12):
    xy = jnp.linspace(-2.0, 2.0, 2 * n).reshape(2, n)
    return MySpatialData(x=xy[0], y=xy[1], idx=jnp.arange(n), exposure_idx=jnp.arange(n) % 2)


# ---- H1: SpatialData is a base class ----


def test_builtin_data_classes_subclass_spatial_data():
    assert issubclass(SpatialDataGeneric, SpatialData)
    assert issubclass(SpatialDataLVM, SpatialData)


def test_spatial_data_itself_is_abstract():
    with pytest.raises(TypeError):
        SpatialData()  # type: ignore[abstract]


def test_converter_still_flattens():
    data = SpatialDataGeneric(
        x=jnp.ones((2, 3)), y=jnp.ones((2, 3)), idx=jnp.arange(6).reshape(2, 3)
    )
    assert data.x.shape == (6,) and data.idx.shape == (6,)


def test_user_defined_spatial_data_works_with_models():
    data = make_my_data()
    assert isinstance(data, SpatialData)
    kernel = Matern32(length_scale=Parameter(1.0), variance=Parameter(1.0))
    gp = FourierGP(n_modes=(5, 5), kernel=kernel)
    per_spaxel = PerSpaxel(12, spaxel_values=Parameter(jnp.arange(12.0)))
    spectral = Gaussian(
        A=per_spaxel,
        λ0=PerSpaxel(12, Parameter(jnp.zeros(12))),
        σ=PerSpaxel(12, Parameter(jnp.ones(12))),
    )

    out = eqx.filter_jit(gp)(data)
    assert out.shape == (12,)
    np.testing.assert_array_equal(eqx.filter_jit(per_spaxel)(data), jnp.arange(12.0))
    λ = jnp.linspace(-1.0, 1.0, 5)
    spectra = eqx.filter_jit(lambda m, l, d: jax.vmap(m, in_axes=(0, None))(l, d))(
        spectral, λ, data
    )
    assert spectra.shape == (5, 12) and jnp.all(jnp.isfinite(spectra))

    # vmap over a batch of data objects
    batch = jax.tree.map(lambda *a: jnp.stack(a), data, data)
    assert jax.vmap(gp)(batch).shape == (2, 12)


# ---- H2: old code is unaffected ----


def test_spatial_data_still_importable_from_top_level():
    import spectracles

    assert spectracles.SpatialData is SpatialData


# ---- H3: lvm_models still imports and runs (experimental, not removed) ----


def test_every_lvm_models_module_imports():
    # lvm_models needs astropy (lvm_models/constants.py), which spectracles doesn't
    # declare; it's in the dev group so this smoke test can run
    pytest.importorskip("astropy")
    for info in pkgutil.iter_modules(spectracles.lvm_models.__path__):
        importlib.import_module(f"spectracles.lvm_models.{info.name}")


def test_lvm_models_marked_experimental():
    assert "Experimental" in spectracles.lvm_models.__doc__


def test_lvm_single_line_model_evaluates():
    pytest.importorskip("astropy")
    from spectracles.lvm_models.line_single import LVMModelSingle

    n_spaxels, n_tiles = 8, 2

    def kernel():
        return Matern32(length_scale=Parameter(1.0), variance=Parameter(1.0))

    model = build_model(
        LVMModelSingle,
        n_tiles=n_tiles,
        n_spaxels=n_spaxels,
        n_modes=(5, 5),
        μ=Known(6562.8),
        A_kernel=kernel(),
        v_kernel=kernel(),
        vσ_kernel=kernel(),
        σ_lsf=Parameter(jnp.full(n_spaxels, 0.8), fixed=True),
        v_bary=Parameter(jnp.zeros(n_spaxels), fixed=True),
        v_syst=Parameter(0.0),
        C_v_cal=Parameter(jnp.zeros(2)),
    )
    xy = jnp.linspace(-1.0, 1.0, 2 * n_spaxels).reshape(2, n_spaxels)
    data = SpatialDataLVM(
        x=xy[0],
        y=xy[1],
        idx=jnp.arange(n_spaxels),
        tile_idx=jnp.arange(n_spaxels) % n_tiles,
        ifu_idx=jnp.arange(n_spaxels) % 3,
        fib_idx=jnp.arange(n_spaxels),
    )
    λ = jnp.linspace(6555.0, 6570.0, 7)
    out = jax.vmap(model, in_axes=(0, None))(λ, data)
    assert out.shape == (7, n_spaxels) and jnp.all(jnp.isfinite(out))
