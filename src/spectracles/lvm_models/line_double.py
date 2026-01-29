"""line_double.py - Spectrospatial model of a two (blended) emission lines with identical kinematics, for LVM."""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from spectracles import (
    AnyParameter,
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


class EmissionLineDouble(SpectralSpatialModel):
    # Line centre in Angstroms
    μ_1: AnyParameter
    μ_2: AnyParameter
    # Model components / line quantities
    A_raw_1: SpatialModel
    A_raw_2: SpatialModel
    v: SpatialModel
    vσ_raw: SpatialModel
    # Measured line quantities
    σ_lsf_1: SpatialModel
    σ_lsf_2: SpatialModel
    v_bary: SpatialModel
    # Systematics
    v_syst: AnyParameter
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

    def A_1(self, s) -> Array:
        return l_bounded(self.A_raw_1(s), lower=A_LOWER)

    def A_2(self, s) -> Array:
        return l_bounded(self.A_raw_2(s), lower=A_LOWER)

    def vσ(self, s) -> Array:
        return l_bounded(self.vσ_raw(s), lower=0.0)

    def v_obs_1(self, s) -> Array:
        return self.v(s) + self.v_syst.val + self.v_cal_1(s) - self.v_bary(s)

    def v_obs_2(self, s) -> Array:
        return self.v(s) + self.v_syst.val + self.v_cal_2(s) - self.v_bary(s)

    def μ_obs_1(self, s) -> Array:
        return self.μ_1.val * (1 + self.v_obs_1(s) / C_KMS)

    def μ_obs_2(self, s) -> Array:
        return self.μ_2.val * (1 + self.v_obs_2(s) / C_KMS)

    def σ2_obs_1(self, s) -> Array:
        return (self.vσ(s) * self.μ_obs_1(s) / C_KMS) ** 2 + self.σ_lsf_1(s) ** 2

    def σ2_obs_2(self, s) -> Array:
        return (self.vσ(s) * self.μ_obs_2(s) / C_KMS) ** 2 + self.σ_lsf_2(s) ** 2

    def f_cal(self, s) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class LVMModelDouble(SpectralSpatialModel):
    # Model components
    line: EmissionLineDouble  # Emission line model
    offs: PerSpaxel  # Nuisance offsets per spaxel

    def __init__(
        self,
        n_tiles: int,
        n_spaxels: int,
        offsets: AnyParameter,
        line_centre_1: AnyParameter,
        line_centre_2: AnyParameter,
        n_modes: tuple[int, int],
        A_kernel_1: Kernel,
        A_kernel_2: Kernel,
        v_kernel: Kernel,
        σ_kernel: Kernel,
        σ_lsf_1: AnyParameter,
        σ_lsf_2: AnyParameter,
        v_bary: AnyParameter,
        v_syst: AnyParameter,
        C_v_cal: AnyParameter,  # MUST be 2 values i.e. shape is (2,)
        f_cal_unconstrained: AnyParameter,
    ):
        self.offs = Constant(const=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offsets))
        self.line = EmissionLineDouble(
            μ_1=line_centre_1,
            μ_2=line_centre_2,
            A_raw_1=FourierGP(n_modes=n_modes, kernel=A_kernel_1),
            A_raw_2=FourierGP(n_modes=n_modes, kernel=A_kernel_2),
            v=FourierGP(n_modes=n_modes, kernel=v_kernel),
            vσ_raw=FourierGP(n_modes=n_modes, kernel=σ_kernel),
            σ_lsf_1=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_1),
            σ_lsf_2=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_2),
            v_bary=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary),
            v_syst=v_syst,
            v_cal_1=WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre_1),
            v_cal_2=WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre_2),
            f_cal_raw=PerTile(n_tiles=n_tiles, tile_values=f_cal_unconstrained),
        )

    def __call__(self, λ, spatial_data):
        return self.offs(λ, spatial_data) + self.line(λ, spatial_data)


def neg_ln_posterior_doublet(model, λ, xy_data, data, u_data, mask):
    vmapped_model = jax.vmap(model, in_axes=(0, None))
    ln_like = ln_likelihood(vmapped_model, λ, xy_data, data, u_data, mask)
    ln_prior = (
        model.line.A_raw_1.prior_logpdf()
        + model.line.A_raw_2.prior_logpdf()
        + model.line.v.prior_logpdf()
        + model.line.vσ_raw.prior_logpdf()
    )
    return -1 * (ln_like + ln_prior)
