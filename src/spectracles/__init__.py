from spectracles.model.data import SpatialData, SpatialDataGeneric, SpatialDataLVM
from spectracles.model.io import load_model, save_model
from spectracles.model.kernels import Kernel, Matern12, Matern32, Matern52, SquaredExponential
from spectracles.model.parameter import (
    AnyParameter,
    ConstrainedParameter,
    Known,
    Parameter,
    l_bounded,
)
from spectracles.model.share_module import ShareModule, build_model, contains_shared
from spectracles.model.spatial import FourierGP, PerSpaxel, SpatialModel
from spectracles.model.spectral import Constant, Gaussian, SpectralSpatialModel, WindowConstant
from spectracles.optimise.opt_frame import OptimiserFrame
from spectracles.optimise.opt_schedule import (
    ManagedOptimiserSchedule,
    OptimiserSchedule,
    PhaseConfig,
)
from spectracles.optimise.schedule_builder import (
    build_schedule,
    fixed_in,
    free_after,
    free_in,
    free_until,
    init_normal,
    init_uniform,
    init_value,
)

__all__ = [
    "AnyParameter",
    "Constant",
    "ConstrainedParameter",
    "FourierGP",
    "Gaussian",
    "Kernel",
    "Known",
    "ManagedOptimiserSchedule",
    "Matern12",
    "Matern32",
    "Matern52",
    "OptimiserFrame",
    "OptimiserSchedule",
    "Parameter",
    "PerSpaxel",
    "PhaseConfig",
    "ShareModule",
    "SpatialData",
    "SpatialDataGeneric",
    "SpatialDataLVM",
    "SpatialModel",
    "SpectralSpatialModel",
    "SquaredExponential",
    "WindowConstant",
    "build_model",
    "build_schedule",
    "contains_shared",
    "fixed_in",
    "free_after",
    "free_in",
    "free_until",
    "init_normal",
    "init_uniform",
    "init_value",
    "l_bounded",
    "load_model",
    "save_model",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"
