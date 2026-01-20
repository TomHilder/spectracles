"""line_ratio.py - Spectrospatial model of a two emission lines parameterised by their amplitude ratio."""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from spectracles import (
    Constant,
    FourierGP,
    Kernel,
    Parameter,
    PerSpaxel,
    SpatialDataLVM,
    SpatialModel,
    SpectralSpatialModel,
    l_bounded,
)
from spectracles.lvm_models.constants import C_KMS
from spectracles.lvm_models.likelihood import ln_likelihood
from spectracles.lvm_models.line_single import WaveCalVelocity
from spectracles.model.spatial import PerTile

A_LOWER = 0.0


class TwoLinesByRatio(SpectralSpatialModel):
    # Line centre in Angstroms
    μ_1: Parameter
    μ_2: Parameter
    # Model components for line 1
    A_raw_1: SpatialModel
    v_1: SpatialModel
    vσ_raw_1: SpatialModel
    # Model components for line 1
    A_log10_ratio: SpatialModel
    v_2: SpatialModel
    vσ_raw_2: SpatialModel
    A_log10_offs: SpatialModel
    # Measured line quantities
    σ_lsf_1: SpatialModel
    σ_lsf_2: SpatialModel
    v_bary: SpatialModel
    # Systematics
    v_syst_1: Parameter
    v_syst_2: Parameter
    v_cal_1: WaveCalVelocity
    v_cal_2: WaveCalVelocity
    f_cal_raw: SpatialModel

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs_1 = self.μ_obs_1(spatial_data)
        μ_obs_2 = self.μ_obs_2(spatial_data)
        σ2_obs_1 = self.σ2_obs_1(spatial_data)
        σ2_obs_2 = self.σ2_obs_2(spatial_data)
        peak_1 = self.A_1(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs_1)
        peak_2 = self.A_2(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs_2)
        f_cal = self.f_cal(spatial_data)
        line_1 = peak_1 * jnp.exp(-0.5 * (λ - μ_obs_1) ** 2 / σ2_obs_1)
        line_2 = peak_2 * jnp.exp(-0.5 * (λ - μ_obs_2) ** 2 / σ2_obs_2)
        return f_cal * (line_1 + line_2)

    def ratio(self, s) -> Array:
        return 10 ** (self.A_log10_ratio(s) + self.A_log10_offs(s))
        # return l_bounded(self.A_log10_ratio(s), lower=0.0)

    def A_1(self, s) -> Array:
        return l_bounded(self.A_raw_1(s), lower=A_LOWER)

    def A_2(self, s) -> Array:
        return self.ratio(s) * self.A_1(s)

    def vσ_1(self, s) -> Array:
        return l_bounded(self.vσ_raw_1(s), lower=0.0)

    def vσ_2(self, s) -> Array:
        return l_bounded(self.vσ_raw_2(s), lower=0.0)

    def v_obs_1(self, s) -> Array:
        return self.v_1(s) + self.v_syst_1.val + self.v_cal_1(s) - self.v_bary(s)

    def v_obs_2(self, s) -> Array:
        return self.v_2(s) + self.v_syst_2.val + self.v_cal_2(s) - self.v_bary(s)

    def μ_obs_1(self, s) -> Array:
        return self.μ_1.val * (1 + self.v_obs_1(s) / C_KMS)

    def μ_obs_2(self, s) -> Array:
        return self.μ_2.val * (1 + self.v_obs_2(s) / C_KMS)

    def σ2_obs_1(self, s) -> Array:
        return (self.vσ_1(s) * self.μ_obs_1(s) / C_KMS) ** 2 + self.σ_lsf_1(s) ** 2

    def σ2_obs_2(self, s) -> Array:
        return (self.vσ_2(s) * self.μ_obs_2(s) / C_KMS) ** 2 + self.σ_lsf_2(s) ** 2

    def f_cal(self, s) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class LVMModelRatio(SpectralSpatialModel):
    # Model components
    lines: TwoLinesByRatio
    offs: PerSpaxel

    def __init__(
        self,
        n_tiles: int,
        n_spaxels: int,
        offsets: Parameter,
        ratio_offsets: Parameter,
        line_centre_1: Parameter,
        line_centre_2: Parameter,
        n_modes: tuple[int, int],
        A_1_kernel: Kernel,
        r_2_kernel: Kernel,
        v_1_kernel: Kernel,
        v_2_kernel: Kernel,
        σ_1_kernel: Kernel,
        σ_2_kernel: Kernel,
        σ_lsf_1: Parameter,
        σ_lsf_2: Parameter,
        v_bary: Parameter,
        v_syst_1: Parameter,
        v_syst_2: Parameter,
        C_v_cal: Parameter,  # MUST be 2 values i.e. shape is (2,)
        f_cal_unconstrained: Parameter,
    ):
        self.offs = Constant(const=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offsets))
        self.lines = TwoLinesByRatio(
            μ_1=line_centre_1,
            μ_2=line_centre_2,
            A_raw_1=FourierGP(n_modes, kernel=A_1_kernel),
            v_1=FourierGP(n_modes, kernel=v_1_kernel),
            vσ_raw_1=FourierGP(n_modes, kernel=σ_1_kernel),
            A_log10_ratio=FourierGP(n_modes, kernel=r_2_kernel),
            v_2=FourierGP(n_modes, kernel=v_2_kernel),
            vσ_raw_2=FourierGP(n_modes, kernel=σ_2_kernel),
            A_log10_offs=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=ratio_offsets),
            σ_lsf_1=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_1),
            σ_lsf_2=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_2),
            v_bary=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary),
            v_syst_1=v_syst_1,
            v_syst_2=v_syst_2,
            v_cal_1=WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre_1),
            v_cal_2=WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre_2),
            f_cal_raw=PerTile(n_tiles=n_tiles, tile_values=f_cal_unconstrained),
        )

    def __call__(self, λ, spatial_data):
        return self.offs(λ, spatial_data) + self.lines(λ, spatial_data)


def neg_ln_posterior(model, λ, xy_data, data, u_data, mask):
    vmapped_model = jax.vmap(model, in_axes=(0, None))
    ln_like = ln_likelihood(vmapped_model, λ, xy_data, data, u_data, mask)
    ln_prior = (
        model.lines.A_raw_1.prior_logpdf()
        + model.lines.A_log10_ratio.prior_logpdf()
        + model.lines.v_1.prior_logpdf()
        + model.lines.v_2.prior_logpdf()
        + model.lines.vσ_raw_1.prior_logpdf()
        + model.lines.vσ_raw_2.prior_logpdf()
        + jnp.sum(
            jax.scipy.stats.laplace.logpdf(model.lines.A_log10_offs.spaxel_values.val, scale=1e-1)
        )
    )
    return -1 * (ln_like + ln_prior)
