"""ratios.py - Spectrospatial models joint over multiple emission lines where the spatial fields of the lines are modelled in terms of line ratio field(s)."""

from jaxtyping import Array

from spectracles import AnyParameter, Parameter, PerSpaxel, SpatialDataLVM, SpatialModel
from spectracles.model.kernels import Kernel

from .fields import FieldFromRatio, GPField, PositiveGPField
from .line_single import EmissionLine, LinkedEmissionLine


class LineRatioModel(SpatialModel):
    """Model of two linked emission lines where the flux spatial field of the weak line is represented via a line ratio field multiplied by the flux spatial field of the strong line. line_1 is the strong line (denominator), line_2 is the weak line (numerator)."""

    # Emission lines
    line_1: EmissionLine
    line_2: LinkedEmissionLine

    # Continuum level (same all wavelengths TODO make local to each line?)
    cont: PerSpaxel

    def __init__(
        self,
        n_spaxels: int,
        n_modes: tuple[int, int],
        μ_1: AnyParameter,
        μ_2: AnyParameter,
        σ_lsf_1: AnyParameter,
        σ_lsf_2: AnyParameter,
        v_bary: AnyParameter,
        v_syst: AnyParameter,
        A_kernel: Kernel,
        v_kernel: Kernel,
        vσ_kernel: Kernel,
        r_kernel: Kernel,
    ):
        # Emission lines
        self.line_1 = EmissionLine(
            μ=μ_1,
            A=PositiveGPField(kernel=A_kernel, n_modes=n_modes),
            v=GPField(kernel=v_kernel, n_modes=n_modes),
            vσ=PositiveGPField(kernel=vσ_kernel, n_modes=n_modes),
            σ_lsf=σ_lsf_1,
            v_bary=v_bary,
            v_syst=v_syst,
        )
        self.line_2 = EmissionLine(
            μ=μ_2,
            A=FieldFromRatio(
                base_field=self.line_1.A,
                log10_ratio_field=GPField(kernel=r_kernel, n_modes=n_modes),
            ),
            v=self.line_1.v,
            vσ=self.line_1.vσ,
            σ_lsf=σ_lsf_2,
            v_bary=v_bary,
            v_syst=v_syst,
        )
        # Local continuum to each line
        self.cont = PerSpaxel(n_spaxels=n_spaxels)

    # Convenience function
    def log10_ratio(self, spatial_data: SpatialDataLVM):
        return self.line_2.A.log10_ratio_field(spatial_data)

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> tuple[Array, Array]:
        """Return the model flux for both lines at the given wavelengths and spatial data."""
        return (
            self.line_1(λ, spatial_data) + self.line_2(λ, spatial_data) + self.cont(spatial_data)
        )
