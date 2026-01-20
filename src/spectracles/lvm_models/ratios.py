"""ratios.py - Spectrospatial models joint over multiple emission lines where the spatial fields of the lines are modelled in terms of line ratio field(s)."""

from jaxtyping import Array

from spectracles import Parameter, PerSpaxel, SpatialDataLVM, SpatialModel
from spectracles.model.spatial import PerTile

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
        μ_1: Parameter,
        μ_2: Parameter,
        σ_lsf_1: Parameter,
        σ_lsf_2: Parameter,
        v_bary: Parameter,
        v_syst: Parameter,
    ):
        # Emission lines
        self.line_1 = EmissionLine(
            μ=μ_1,
            A=PositiveGPField(...),
            v=GPField(...),
            vσ=PositiveGPField(...),
            σ_lsf=σ_lsf_1,
            v_bary=v_bary,
            v_syst=v_syst,
        )
        # self.line_2 = LinkedEmissionLine(
        #     μ=μ_2,
        #     A=FieldFromRatio(
        #         base_field=self.line_1.A,
        #         log10_ratio_field=GPField(...),
        #     ),
        #     parent_line=self.line_1,
        #     σ_lsf=σ_lsf_2,
        #     v_bary=v_bary,
        #     v_syst=v_syst,
        # )
        self.line_2 = EmissionLine(
            μ=μ_2,
            A=FieldFromRatio(
                base_field=self.line_1.A,
                log10_ratio_field=GPField(...),
            ),
            v=self.line_1.v,
            vσ=self.line_1.vσ,
            σ_lsf=σ_lsf_2,
            v_bary=v_bary,
            v_syst=v_syst,
        )
        # Local continuum to each line
        self.cont = PerSpaxel(...)

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> tuple[Array, Array]:
        """Return the model flux for both lines at the given wavelengths and spatial data."""
        return (
            self.line_1(λ, spatial_data) + self.line_2(λ, spatial_data) + self.cont(spatial_data)
        )
