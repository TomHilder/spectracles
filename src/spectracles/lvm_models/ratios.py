"""ratios.py - Spectrospatial models joint over multiple emission lines where the spatial fields of the lines are modelled in terms of line ratio field(s)."""

import jax
from jaxtyping import Array

from .. import (
    AnyParameter,
    Kernel,
    PerSpaxel,
    SpatialDataLVM,
    SpectralSpatialModel,
    WindowConstant,
)
from .fields import FieldFromRatio, GPField, PositiveGPField
from .likelihood import ln_likelihood
from .line_single import EmissionLineVCal, WaveCalVelocity


class LineRatioModel(SpectralSpatialModel):
    """Model of two linked emission lines where the flux spatial field of the weak line is represented via a line ratio field multiplied by the flux spatial field of the strong line. line_1 is the strong line (denominator), line_2 is the weak line (numerator)."""

    # Emission lines
    line_1: EmissionLineVCal
    line_2: EmissionLineVCal

    # Continuum level
    cont_1: WindowConstant
    cont_2: WindowConstant

    def __init__(
        self,
        n_spaxels: int,
        n_modes: tuple[int, int],
        μ_1: AnyParameter,
        μ_2: AnyParameter,
        σ_lsf_1: AnyParameter,
        σ_lsf_2: AnyParameter,
        v_bary: AnyParameter,
        v_syst_1: AnyParameter,
        v_syst_2: AnyParameter,
        A_kernel: Kernel,
        v_kernel: Kernel,
        vσ_kernel: Kernel,
        r_kernel: Kernel,
        r_mean_log10: AnyParameter,
        line_1_λ_window: tuple[float, float],
        line_2_λ_window: tuple[float, float],
        C_v_cal_1: AnyParameter,  # MUST be 2 values i.e. shape is (2,)
        C_v_cal_2: AnyParameter,  # MUST be 2 values i.e. shape is (2,)
    ):
        # Barycentric correction and LSF as per-spaxel sub-models
        # Very likely these will be fixed (σ_lsf_1, σ_lsf_2v_bary as Known)
        # But we leave it general
        v_bary_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary)
        σ_lsf_1_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_1)
        σ_lsf_2_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_2)

        # Systematics / calibration corrections
        v_cal_1 = WaveCalVelocity(C_v_cal=C_v_cal_1, μ=μ_1)
        v_cal_2 = WaveCalVelocity(C_v_cal=C_v_cal_2, μ=μ_2)

        # Emission lines
        self.line_1 = EmissionLineVCal(
            μ=μ_1,
            A=PositiveGPField(kernel=A_kernel, n_modes=n_modes),
            v=GPField(kernel=v_kernel, n_modes=n_modes),
            vσ=PositiveGPField(kernel=vσ_kernel, n_modes=n_modes),
            σ_lsf=σ_lsf_1_,
            v_bary=v_bary_,
            v_syst=v_syst_1,
            v_cal=v_cal_1,
        )
        self.line_2 = EmissionLineVCal(
            μ=μ_2,
            A=FieldFromRatio(
                base_field=self.line_1.A,
                log10_ratio_field=GPField(kernel=r_kernel, n_modes=n_modes),
                log10_ratio_mean=r_mean_log10,
            ),
            v=self.line_1.v,
            vσ=self.line_1.vσ,
            σ_lsf=σ_lsf_2_,
            v_bary=v_bary_,
            v_syst=v_syst_2,
            v_cal=v_cal_2,
        )
        # Local continuum to each line
        self.cont_1 = WindowConstant(
            const=PerSpaxel(n_spaxels=n_spaxels),
            λ_min=line_1_λ_window[0],
            λ_max=line_1_λ_window[1],
        )
        self.cont_2 = WindowConstant(
            const=PerSpaxel(n_spaxels=n_spaxels),
            λ_min=line_2_λ_window[0],
            λ_max=line_2_λ_window[1],
        )

    # Convenience function
    def log10_ratio(self, spatial_data: SpatialDataLVM):
        return self.line_2.A.log10_ratio(spatial_data)

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> tuple[Array, Array]:
        """Return the model flux for both lines at the given wavelengths and spatial data."""
        comp_1 = self.line_1(λ, spatial_data) + self.cont_1(λ, spatial_data)
        comp_2 = self.line_2(λ, spatial_data) + self.cont_2(λ, spatial_data)
        return comp_1 + comp_2


def neg_ln_posterior(model, λ, xy_data, data, u_data, mask):
    vmapped_model = jax.vmap(model, in_axes=(0, None))
    ln_like = ln_likelihood(vmapped_model, λ, xy_data, data, u_data, mask)
    ln_prior = (
        model.line_1.A.gp.prior_logpdf()
        + model.line_1.v.gp.prior_logpdf()
        + model.line_1.vσ.gp.prior_logpdf()
        + model.line_2.A.log10_ratio_field.gp.prior_logpdf()
    )
    return -1 * (ln_like + ln_prior)
