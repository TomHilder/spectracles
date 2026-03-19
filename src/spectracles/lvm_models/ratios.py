"""ratios.py - Spectrospatial models joint over multiple emission lines where the spatial fields of the lines are modelled in terms of line ratio field(s)."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import Array

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
from .fields import FieldFromRatio, GPField, PositiveGPField, SumTwoFields
from .likelihood import ln_likelihood
from .line_single import EmissionLineVCal, WaveCalVelocity


class FluxCalFactor(SpatialModel):
    # Hierarchical flux calibration factor per tile, with per-IFU variations
    f_cal_raw: PerTilePinned  # Unconstrained flux calibration factor per tile # N_TILES values
    delta_f_cal: PerIFU  # Additive per-IFU variation in flux calibration factor # 3 values

    def __init__(self, n_tiles, tile_values, ifu_values):
        # self.f_cal_raw = PerTilePinned(n_tiles=n_tiles, tile_values=tile_values)
        self.f_cal_raw = PerTile(n_tiles=n_tiles, tile_values=tile_values)
        self.delta_f_cal = PerIFU(n_ifus=3, ifu_values=ifu_values)

    def __call__(self, s: SpatialData) -> Array:
        return l_bounded(self.f_cal_raw(s) + self.delta_f_cal(s), lower=0.0) / l_bounded(
            0, lower=0.0
        )


class LineRatioModel(SpectralSpatialModel):
    """Model of two emission lines where the flux spatial field of the weak line is represented via a line ratio field multiplied by the flux spatial field of the strong line. line_1 is the strong line (denominator), line_2 is the weak line (numerator)."""

    # Emission lines
    line_1: EmissionLineVCal
    line_2: EmissionLineVCal

    # Continuum level
    cont_1: WindowConstant
    cont_2: WindowConstant

    # Calibration/Nuisances
    flux_cal_1: FluxCalFactor
    flux_cal_2: FluxCalFactor

    def __init__(
        self,
        n_tiles: int,
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
        share_f_cal: bool = False,
        share_v_cal: bool = False,
        share_kinematics: bool = True,
    ):
        # Barycentric correction and LSF as per-spaxel sub-models
        # Very likely these will be fixed (σ_lsf_1, σ_lsf_2v_bary as Known)
        # But we leave it general
        v_bary_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary)
        σ_lsf_1_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_1)
        σ_lsf_2_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_2)
        # Systematics / calibration corrections
        v_cal_1 = WaveCalVelocity(C_v_cal=C_v_cal_1, μ=μ_1)
        if share_v_cal:
            # v_cal_2 = v_cal_1
            v_cal_2 = WaveCalVelocity(C_v_cal=C_v_cal_1, μ=μ_2)
        else:
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
        # Are the kinematics coupled, or independent?
        if share_kinematics:
            l2_v = self.line_1.v
            l2_vσ = self.line_1.vσ
        else:
            l2_v = GPField(kernel=v_kernel, n_modes=n_modes)
            l2_vσ = PositiveGPField(kernel=vσ_kernel, n_modes=n_modes)
        # Line 2 setup with intensity given by a ratio to line 1
        self.line_2 = EmissionLineVCal(
            μ=μ_2,
            A=FieldFromRatio(
                base_field=self.line_1.A,
                log10_ratio_field=GPField(kernel=r_kernel, n_modes=n_modes),
                log10_ratio_mean=r_mean_log10,
            ),
            v=l2_v,
            vσ=l2_vσ,
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
        self.flux_cal_1 = FluxCalFactor(
            n_tiles=n_tiles,
            # tile_values=Parameter(jnp.zeros((n_tiles - 1,)), fixed=True),
            tile_values=Parameter(jnp.zeros((n_tiles,)), fixed=True),
            ifu_values=Parameter(jnp.zeros((3,)), fixed=True),
        )
        if share_f_cal:
            self.flux_cal_2 = self.flux_cal_1
        else:
            self.flux_cal_2 = FluxCalFactor(
                n_tiles=n_tiles,
                # tile_values=Parameter(jnp.zeros((n_tiles - 1,)), fixed=True),
                tile_values=Parameter(jnp.zeros((n_tiles,)), fixed=True),
                ifu_values=Parameter(jnp.zeros((3,)), fixed=True),
            )

    # Convenience function
    def log10_ratio(self, spatial_data: SpatialDataLVM):
        return self.line_2.A.log10_ratio(spatial_data)

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> tuple[Array, Array]:
        """Return the model flux for both lines at the given wavelengths and spatial data."""
        # return (
        #     self.flux_cal_1(spatial_data) * self.line_1(λ, spatial_data)
        #     + self.cont_1(λ, spatial_data)
        #     + self.flux_cal_2(spatial_data) * self.line_2(λ, spatial_data)
        #     + self.cont_2(λ, spatial_data)
        # )
        comp_1 = self.line_1(λ, spatial_data) + self.cont_1(λ, spatial_data)
        comp_2 = self.line_2(λ, spatial_data) + self.cont_2(λ, spatial_data)
        # fcal = self.flux_cal(spatial_data)
        # return fcal * (comp_1 + comp_2)
        fcal_1 = self.flux_cal_1(spatial_data)
        fcal_2 = self.flux_cal_2(spatial_data)
        return fcal_1 * comp_1 + fcal_2 * comp_2


class DoubleLineRatioModel(SpectralSpatialModel):
    """Mode of three emission lines. Two, stronger lines, with a total flux spatial field. The single, weaker line, is represented via a line ratio field multiplied by the total flux spatial field of the stronger lines. lines_1 are the total contributions from the stronger lines (denominator) and line_2 is the weaker line (numerator). TODO: update description."""

    # Emission lines
    line_s1: EmissionLineVCal
    line_s2: EmissionLineVCal
    line_w: EmissionLineVCal

    # Continuum level
    cont_s: WindowConstant
    cont_w: WindowConstant

    # Calibration/Nuisances
    flux_cal_s: FluxCalFactor
    flux_cal_w: FluxCalFactor

    def __init__(
        self,
        n_tiles: int,
        n_spaxels: int,
        n_modes: tuple[int, int],
        μ_s1: AnyParameter,
        μ_s2: AnyParameter,
        μ_w: AnyParameter,
        v_bary: AnyParameter,
        v_syst_s: AnyParameter,
        v_syst_w: AnyParameter,
        σ_lsf_s1: AnyParameter,
        σ_lsf_s2: AnyParameter,
        σ_lsf_w: AnyParameter,
        A_kernel: Kernel,
        v_kernel: Kernel,
        vσ_kernel: Kernel,
        r_kernel: Kernel,
        r_mean_log10: AnyParameter,
        line_s_λ_window: tuple[float, float],
        line_w_λ_window: tuple[float, float],
        C_v_cal_s: AnyParameter,  # MUST be 2 values i.e. shape is (2,)
        C_v_cal_w: AnyParameter,  # MUST be 2 values i.e. shape is (2,)
        share_f_cal: bool = False,
        share_v_cal: bool = False,
        share_kinematics: bool = True,
    ):
        # Barycentric correction and LSFs as per-spaxel sub-models
        # Most like Knowns but generically parameters
        v_bary_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary)
        σ_lsf_s1_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_s1)
        σ_lsf_s2_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_s2)
        σ_lsf_w_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_w)
        # Systematics / calibration corrections
        v_cal_s1 = WaveCalVelocity(C_v_cal=C_v_cal_s, μ=μ_s1)
        v_cal_s2 = WaveCalVelocity(C_v_cal=C_v_cal_s, μ=μ_s2)
        if share_v_cal:
            v_cal_w = WaveCalVelocity(C_v_cal=C_v_cal_s, μ=μ_w)
        else:
            v_cal_w = WaveCalVelocity(C_v_cal=C_v_cal_w, μ=μ_w)
        # Emission lines
        self.line_s1 = EmissionLineVCal(
            μ=μ_s1,
            A=PositiveGPField(kernel=A_kernel, n_modes=n_modes),
            v=GPField(kernel=v_kernel, n_modes=n_modes),
            vσ=PositiveGPField(kernel=vσ_kernel, n_modes=n_modes),
            σ_lsf=σ_lsf_s1_,
            v_bary=v_bary_,
            v_syst=v_syst_s,
            v_cal=v_cal_s1,
        )
        self.line_s2 = EmissionLineVCal(
            μ=μ_s2,
            A=PositiveGPField(kernel=A_kernel, n_modes=n_modes),
            v=self.line_s1.v,
            vσ=self.line_s1.vσ,
            σ_lsf=σ_lsf_s2_,
            v_bary=v_bary_,
            v_syst=v_syst_s,
            v_cal=v_cal_s2,
        )
        # Are the kinematics coupled, or independent?
        if share_kinematics:
            lw_v = self.line_s1.v
            lw_vσ = self.line_s1.vσ
        else:
            lw_v = GPField(kernel=v_kernel, n_modes=n_modes)
            lw_vσ = PositiveGPField(kernel=v_kernel, n_modes=n_modes)
        # Weak line setup with intensity given by a ratio to total of strong lines fluxes
        self.line_w = EmissionLineVCal(
            μ=μ_w,
            A=FieldFromRatio(
                base_field=SumTwoFields(self.line_s1.A, self.line_s2.A),
                log10_ratio_field=GPField(kernel=r_kernel, n_modes=n_modes),
                log10_ratio_mean=r_mean_log10,
            ),
            v=lw_v,
            vσ=lw_vσ,
            σ_lsf=σ_lsf_w_,
            v_bary=v_bary_,
            v_syst=v_syst_w,
            v_cal=v_cal_w,
        )
        # Local continuum to each line
        self.cont_s = WindowConstant(
            const=PerSpaxel(n_spaxels=n_spaxels),
            λ_min=line_s_λ_window[0],
            λ_max=line_s_λ_window[1],
        )
        self.cont_w = WindowConstant(
            const=PerSpaxel(n_spaxels=n_spaxels),
            λ_min=line_w_λ_window[0],
            λ_max=line_w_λ_window[1],
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
        return self.line_w.A.log10_ratio(spatial_data)

    # Convenience function for line_s total
    def line_s(self, λ: Array, spatial_data: SpatialDataLVM):
        return self.line_s1(λ, spatial_data) + self.line_s2(λ, spatial_data)

    # Convenience function for total A field
    def line_s_A(self, spatial_data: SpatialDataLVM):
        return self.line_s1.A(spatial_data) + self.line_s2.A(spatial_data)

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> tuple[Array, Array]:
        """Return the model flux for both lines at the given wavelengths and spatial data."""
        comp_s = self.line_s(λ, spatial_data) + self.cont_s(λ, spatial_data)
        comp_w = self.line_w(λ, spatial_data) + self.cont_w(λ, spatial_data)
        fcal_s = self.flux_cal_s(spatial_data)
        fcal_w = self.flux_cal_w(spatial_data)
        return fcal_s * comp_s + fcal_w * comp_w


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
    # Line 1 contributions
    ln_prior = ln_prior + locked_model.line_1.A.gp.prior_logpdf()
    ln_prior = ln_prior + locked_model.line_1.v.gp.prior_logpdf()
    ln_prior = ln_prior + locked_model.line_1.vσ.gp.prior_logpdf()
    # Line 2 contributions
    ln_prior = ln_prior + locked_model.line_2.A.log10_ratio_field.gp.prior_logpdf()
    if not shared_kinematics:
        ln_prior = ln_prior + locked_model.line_2.v.gp.prior_logpdf()
        ln_prior = ln_prior + locked_model.line_2.vσ.gp.prior_logpdf()
    # Other priors
    ln_prior = (
        ln_prior
        + norm.logpdf(x=locked_model.flux_cal_1.f_cal_raw.tile_values.val, loc=0, scale=0.1).sum()
    )
    ln_prior = (
        ln_prior
        + norm.logpdf(x=locked_model.flux_cal_2.f_cal_raw.tile_values.val, loc=0, scale=0.1).sum()
    )
    return -1 * (ln_like + ln_prior)


def neg_ln_posterior_double(
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
    ln_prior = ln_prior + locked_model.line_s1.A.gp.prior_logpdf()
    ln_prior = ln_prior + locked_model.line_s2.A.gp.prior_logpdf()
    ln_prior = ln_prior + locked_model.line_s1.v.gp.prior_logpdf()
    ln_prior = ln_prior + locked_model.line_s1.vσ.gp.prior_logpdf()
    # Weak line contributions
    ln_prior = ln_prior + locked_model.line_w.A.log10_ratio_field.gp.prior_logpdf()
    if not shared_kinematics:
        ln_prior = ln_prior + locked_model.line_w.v.gp.prior_logpdf()
        ln_prior = ln_prior + locked_model.line_w.vσ.gp.prior_logpdf()
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
