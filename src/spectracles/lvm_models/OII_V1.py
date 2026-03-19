import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import Array

from spectracles.model.parameter import Known

from .. import (
    AnyParameter,
    Kernel,
    Parameter,
    PerSpaxel,
    SpatialData,
    SpatialDataLVM,
    SpectralSpatialModel,
    WindowConstant,
    l_bounded,
)
from ..model.spatial import PerIFU, PerTile, PerTilePinned, SpatialModel
from .fields import FieldFromRatio, GPField, PositiveGPField, ScaledField, SumTwoFields
from .likelihood import ln_likelihood
from .line_single import EmissionLineVCal, WaveCalVelocity
from .ratios import FluxCalFactor

μ_v1 = [
    4638.86,
    4641.81,
    4649.13,
    4650.84,
    4661.63,
    4673.73,
    4676.23,
    4696.35,
]
μ_nii_4643 = 4643.16
μ_feiii_4658 = 4658.10


class OIIV1Model(SpectralSpatialModel):
    """Plan:

    Included weak lines:
    - entire OII V1 multiplet
    - contaminating [Fe III] 4658
    - contaminating N II 4643

    Included strong lines:
    - either [O III] 5007 or Hbeta

    Ratio models total inensity of V1 to strong line, e.g.:
    - ratio = total_V1 / [O III] 5007

    Fractional intensities of V1 lines fixed
    - V1_line_1 = const * total_V1

    Contaminating lines
    - N II 4643 / O II V1 ratio constant everwhere but fitted
    - [Fe III] 4658 / O II V1 ratio constant everwhere but fitted

    Shared kinematics for [O III] 5007 but not for Hbeta.
    """

    # O II V1 multiplet lines
    oii_4638 = EmissionLineVCal
    oii_4641 = EmissionLineVCal
    oii_4649 = EmissionLineVCal
    oii_4650 = EmissionLineVCal
    oii_4661 = EmissionLineVCal
    oii_4673 = EmissionLineVCal
    oii_4676 = EmissionLineVCal
    oii_4696 = EmissionLineVCal

    # Contaminating lines
    nii_4643 = EmissionLineVCal
    feiii_4658 = EmissionLineVCal

    # Bright line (ratio denominator)
    line_s = EmissionLineVCal

    # Continuum near each window
    cont_w = WindowConstant
    cont_s = WindowConstant

    # Calibration/Nuisances
    flux_cal_s: FluxCalFactor
    flux_cal_w: FluxCalFactor

    def __init__(
        self,
        n_tiles: int,
        n_spaxels: int,
        n_modes: tuple[int, int],
        μ_s: AnyParameter,
        v_bary: AnyParameter,
        v_syst_s: AnyParameter,
        v_syst_w: AnyParameter,
        σ_lsfs_oii_v1: tuple[AnyParameter],  # 8 of these
        σ_lsf_nii_4643: AnyParameter,
        σ_lsf_feiii_4658: AnyParameter,
        σ_lsf_line_s: AnyParameter,
        A_kernel: Kernel,
        v_kernel: Kernel,
        vσ_kernel: Kernel,
        r_kernel: Kernel,
        r_mean_log10: AnyParameter,
        oii_v1_λ_window: tuple[float, float],
        line_s_λ_window: tuple[float, float],
        frac_intensities_oii_v1: tuple[AnyParameter],  # 8 of these
        nii_oii_v1_ratio: AnyParameter,
        feiii_oii_v1_ratio: AnyParameter,
        C_v_cal_oii_v1: AnyParameter,  # MUST be 2 values i.e. shape is (2,)
        C_v_cal_s: AnyParameter,  # MUST be 2 values i.e. shape is (2,)
        share_f_cal: bool = False,
        share_v_cal: bool = False,
        share_kinematics: bool = True,
    ):
        # Barycentric correction and LSFs as per-spaxel sub-models
        # Most like Knowns but generically parameters
        v_bary_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary)
        σ_lsf_oii_ = tuple(
            PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_oii) for σ_lsf_oii in σ_lsfs_oii_v1
        )
        σ_lsf_nii_4643_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_nii_4643)
        σ_lsf_feiii_4658_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_feiii_4658)
        σ_lsf_line_s_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_line_s)
        # Systematics / calibration corrections
        v_cal_s_ = WaveCalVelocity(
            C_v_cal=C_v_cal_s,
            μ=μ_s,  # Already AnyParameter
        )
        if share_v_cal:
            C_v_cal_w_ = C_v_cal_s
        else:
            C_v_cal_w_ = C_v_cal_oii_v1
        v_cal_oii_ = tuple(
            WaveCalVelocity(
                C_v_cal=C_v_cal_w_,
                μ=Known(μ_v1[i]),
            )
            for i in range(8)
        )
        v_cal_nii_4643_ = WaveCalVelocity(
            C_v_cal=C_v_cal_w_,
            μ=Known(μ_nii_4643),
        )
        v_cal_feiii_4658_ = WaveCalVelocity(
            C_v_cal=C_v_cal_w_,
            μ=Known(μ_feiii_4658),
        )
        # Emission lines
        self.line_s = EmissionLineVCal(
            μ=μ_s,
            A=PositiveGPField(kernel=A_kernel, n_modes=n_modes),
            v=GPField(kernel=v_kernel, n_modes=n_modes),
            vσ=GPField(kernel=vσ_kernel, n_modes=n_modes),
            σ_lsf=σ_lsf_line_s_,
            v_bary=v_bary_,
            v_syst=v_syst_s,
            v_cal=v_cal_s_,
        )
        oii_v1_ratio = FieldFromRatio(
            base_field=self.line_s.A,
            log10_ratio_field=GPField(kernel=r_kernel, n_modes=n_modes),
            log10_ratio_mean=r_mean_log10,
        )
        # Are the kinematics coupled, or independent?
        if share_kinematics:
            v1_v = self.line_s1.v
            v1_vσ = self.line_s1.vσ
        else:
            v1_v = GPField(kernel=v_kernel, n_modes=n_modes)
            v1_vσ = PositiveGPField(kernel=v_kernel, n_modes=n_modes)
        oii_lines = tuple(
            EmissionLineVCal(
                μ=Known(μ_v1[i]),
                A=ScaledField(
                    scalar=frac_intensities_oii_v1[i],
                    field=oii_v1_ratio,
                ),
                v=v1_v,
                vσ=v1_vσ,
                σ_lsf=σ_lsf_oii_[i],
                v_bary=v_bary_,
                v_syst=v_syst_w,
                v_cal=v_cal_oii_[i],
            )
            for i in range(8)
        )
        self.oii_4638 = oii_lines[0]
        self.oii_4641 = oii_lines[1]
        self.oii_4649 = oii_lines[2]
        self.oii_4650 = oii_lines[3]
        self.oii_4661 = oii_lines[4]
        self.oii_4673 = oii_lines[5]
        self.oii_4676 = oii_lines[6]
        self.oii_4696 = oii_lines[7]
        self.nii_4643 = EmissionLineVCal(
            μ=Known(μ_nii_4643),
            A=ScaledField(
                scalar=nii_oii_v1_ratio,  # NII 4643 / OII V1 ratio
                field=oii_v1_ratio,
            ),
            v=v1_v,
            vσ=v1_vσ,
            σ_lsf=σ_lsf_nii_4643_,
            v_bary=v_bary_,
            v_syst=v_syst_w,
            v_cal=v_cal_nii_4643_,
        )
        self.feiii_4658 = EmissionLineVCal(
            μ=Known(μ_feiii_4658),
            A=ScaledField(
                scalar=feiii_oii_v1_ratio,  # FeIII 4658 / OII V1 ratio
                field=oii_v1_ratio,
            ),
            v=v1_v,
            vσ=v1_vσ,
            σ_lsf=σ_lsf_feiii_4658_,
            v_bary=v_bary_,
            v_syst=v_syst_w,
            v_cal=v_cal_feiii_4658_,
        )
        # Local continuum to each
        self.cont_s = WindowConstant(
            const=PerSpaxel(n_spaxels=n_spaxels),
            λ_min=line_s_λ_window[0],
            λ_max=line_s_λ_window[1],
        )
        self.cont_w = WindowConstant(
            const=PerSpaxel(n_spaxels=n_spaxels),
            λ_min=oii_v1_λ_window[0],
            λ_max=oii_v1_λ_window[1],
        )
        # Flux cal
        self.flux_cal_s = FluxCalFactor(
            n_tiles=n_tiles,
            tile_values=Parameter(jnp.zeros((n_tiles,)), fixed=True),
            ifu_values=Parameter(jnp.zeros((3,)), fixed=True),
        )
        if share_f_cal:
            self.flux_cal_w = self.flux_cal_s
        else:
            self.flux_cal_w = FluxCalFactor(
                n_tiles=n_tiles,
                tile_values=Parameter(jnp.zeros((n_tiles,)), fixed=True),
                ifu_values=Parameter(jnp.zeros((3,)), fixed=True),
            )

    # Convenience function for ratio
    def log10_ratio(self, spatial_data: SpatialDataLVM):
        return self.oii_4638.A.field.log10_ratio(spatial_data)

    # Convenience function for v1 total
    def line_v1(self, λ: Array, spatial_data: SpatialDataLVM):
        return (
            self.oii_4638(λ, spatial_data)
            + self.oii_4641(λ, spatial_data)
            + self.oii_4649(λ, spatial_data)
            + self.oii_4650(λ, spatial_data)
            + self.oii_4661(λ, spatial_data)
            + self.oii_4673(λ, spatial_data)
            + self.oii_4676(λ, spatial_data)
            + self.oii_4696(λ, spatial_data)
        )

    # Convenience function for v1 with contaminants
    def line_v1_plus_contaminants(self, λ: Array, spatial_data: SpatialDataLVM):
        return (
            self.line_v1(λ, spatial_data)
            + self.nii_4643(λ, spatial_data)
            + self.feiii_4658(λ, spatial_data)
        )

    # Convenience function for total intensity field of v1
    def line_v1_total_intensity(self, spatial_data: SpatialDataLVM):
        return self.oii_4638.A.field(spatial_data)  # This should work despite it being confusing

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        comp_s = self.line_s(λ, spatial_data) + self.cont_s(λ, spatial_data)
        comp_w = self.line_v1_plus_contaminants(λ, spatial_data) + self.cont_w(λ, spatial_data)
        return self.flux_cal_s(spatial_data) * comp_s + self.flux_cal_w(spatial_data) * comp_w


def neg_ln_posterior(
    model,
    λ,
    xy_data,
    data,
    u_data,
    mask,
    shared_kinematics=True,
):
    vmapped_model = jax.vmap(model, in_axes=(0, None))
    ln_like = ln_likelihood(vmapped_model, λ, xy_data, data, u_data, mask)
    locked_model = model.get_locked_model()
    # === Get the prior
    ln_prior = 0
    # Strong line contributions
    ln_prior = ln_prior + locked_model.line_s.A.gp.prior_logpdf()
    ln_prior = ln_prior + locked_model.line_s.v.gp.prior_logpdf()
    ln_prior = ln_prior + locked_model.line_s.vσ.gp.prior_logpdf()
    # Weak line contributions
    ln_prior = ln_prior + locked_model.oii_4638.A.field.gp.prior_logpdf()  # ratio field (only one)
    if not shared_kinematics:
        ln_prior = ln_prior + locked_model.oii_4638.v.gp.prior_logpdf()
        ln_prior = ln_prior + locked_model.oii_4638.vσ.gp.prior_logpdf()
    # Other priors
    ln_prior = (
        ln_prior
        + norm.logpdf(x=locked_model.flux_cal_s.f_cal_raw.tile_values.val, loc=0, scale=0.1).sum()
    )
    ln_prior = (
        ln_prior
        + norm.logpdf(x=locked_model.flux_cal_w.f_cal_raw.tile_values.val, loc=0, scale=0.1).sum()
    )
    return -1 * (ln_like + ln_prior)
