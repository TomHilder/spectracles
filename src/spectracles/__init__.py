from spectracles.model.data import SpatialDataGeneric, SpatialDataLVM
from spectracles.model.io import load_model, save_model
from spectracles.model.kernels import Kernel, Matern12, Matern32, Matern52, SquaredExponential
from spectracles.model.parameter import AnyParameter, ConstrainedParameter, Parameter, l_bounded
from spectracles.model.share_module import build_model
from spectracles.model.spatial import FourierGP, PerSpaxel, SpatialModel
from spectracles.model.spectral import Constant, Gaussian, SpectralSpatialModel
from spectracles.optimise.opt_frame import OptimiserFrame
from spectracles.optimise.opt_schedule import OptimiserSchedule, PhaseConfig

__all__ = [
    "FourierGP",
    "Gaussian",
    "SpatialDataGeneric",
    "SpatialDataLVM",
    "PerSpaxel",
    "Constant",
    "Kernel",
    "Matern12",
    "Matern32",
    "Matern52",
    "SquaredExponential",
    "build_model",
    "SpatialModel",
    "SpectralSpatialModel",
    "Parameter",
    "ConstrainedParameter",
    "AnyParameter",
    "l_bounded",
    "OptimiserFrame",
    "save_model",
    "load_model",
    "PhaseConfig",
    "OptimiserSchedule",
]

try:
    from ._version import __version__
except Exception:
    __version__ = "0+unknown"
