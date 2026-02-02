"""line_single.py - Spectrospatial model of a single emission line for LVM."""

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
from spectracles.model.data import SpatialData
from spectracles.model.spatial import PerTile

A_LOWER = 0.0


class WaveCalVelocity(SpatialModel):
    """Per-IFU velocity calibration model based on line centre shifts."""

    # Model parameters
    C_v_cal: AnyParameter  # 2 value model parameter
    # Constants
    μ: AnyParameter  # line centre in Angstroms

    def __call__(self, data: SpatialData) -> Array:
        v0 = 0.0  # effectively pinned to v_syst
        v1 = v0 - C_KMS * (self.C_v_cal.val[0] / self.μ.val[0])
        v2 = v1 - C_KMS * (self.C_v_cal.val[1] / self.μ.val[0])
        v_ifu_vals = jnp.array([v0, v1, v2])
        return v_ifu_vals[data.ifu_idx]


class EmissionLine(SpectralSpatialModel):
    """Simple emission line model with no systematics or calibration."""

    # Line centre in Angstroms
    μ: AnyParameter

    # Model components
    A: SpatialModel  # Line flux
    v: SpatialModel  # Radial velocity in rest frame in km/s
    vσ: SpatialModel  # Broadening velocity in km/s

    # Measured line quantities
    σ_lsf: SpatialModel  # LSF width (std dev) in Angstroms
    v_bary: SpatialModel  # Barycentric velocity correction in km/s

    # Global parameters
    v_syst: AnyParameter  # Systematic velocity offset in km/s

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(spatial_data)
        σ2_obs = self.σ2_obs(spatial_data)
        peak = self.A(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        return peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs)

    def v_obs(self, s) -> Array:
        return self.v(s) + self.v_syst.val - self.v_bary(s)

    def μ_obs(self, s) -> Array:
        return self.μ.val * (1 + self.v_obs(s) / C_KMS)

    def σ2_obs(self, s) -> Array:
        return (self.vσ(s) * self.μ_obs(s) / C_KMS) ** 2 + self.σ_lsf(s) ** 2


class LinkedEmissionLine(SpectralSpatialModel):
    """Emission line where the velocity and broadening are linked to another line."""

    # Line centre in Angstroms
    μ: AnyParameter

    # Model components
    A: SpatialModel  # Line flux

    # Parent line
    parent_line: EmissionLine

    # Measured line quantities
    σ_lsf: SpatialModel  # LSF width (std dev) in Angstrom
    v_bary: SpatialModel  # Barycentric velocity correction in km/s

    # Global parameters
    v_syst: AnyParameter  # Systematic velocity offset in km/s

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(spatial_data)
        σ2_obs = self.σ2_obs(spatial_data)
        peak = self.A(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        return peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs)

    def μ_obs(self, s) -> Array:
        v_parent = self.parent_line.v_obs(s)
        return self.μ.val * (1 + v_parent / C_KMS)

    def σ2_obs(self, s) -> Array:
        vσ_parent = self.parent_line.vσ(s)
        return (vσ_parent * self.μ_obs(s) / C_KMS) ** 2 + self.σ_lsf(s) ** 2


class EmissionLineProduction(SpectralSpatialModel):
    # Line centre in Angstroms
    μ: AnyParameter
    # Model components / line quantities
    A_raw: SpatialModel  # Unconstrained line flux # TODO: refactor to A and move contraint to a positive GP class
    v: SpatialModel  # Radial velocity in rest frame in km/s
    vσ_raw: SpatialModel  # Broadening velocity in km/s before constraint
    # Measured line quantities
    σ_lsf: SpatialModel  # LSF width (std dev) in Angstroms
    v_bary: SpatialModel  # Barycentric velocity correction in km/s
    # Systematics
    v_syst: AnyParameter  # Systematic velocity offset in km/s
    v_cal: WaveCalVelocity  # Per-IFU Velocity calibration offset in km/s
    f_cal_raw: SpatialModel  # Flux calibration factor per tile

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(spatial_data)
        σ2_obs = self.σ2_obs(spatial_data)
        peak = self.A(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        f_cal = self.f_cal(spatial_data)
        return f_cal * peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def vσ(self, s) -> Array:
        return l_bounded(self.vσ_raw(s), lower=0.0)

    def v_obs(self, s) -> Array:
        return self.v(s) + self.v_syst.val + self.v_cal(s) - self.v_bary(s)

    def μ_obs(self, s) -> Array:
        return self.μ.val * (1 + self.v_obs(s) / C_KMS)

    def σ2_obs(self, s) -> Array:
        return (self.vσ(s) * self.μ_obs(s) / C_KMS) ** 2 + self.σ_lsf(s) ** 2

    def f_cal(self, s) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class LVMModelSingle(SpectralSpatialModel):
    # Model components
    line: EmissionLineProduction  # Emission line model
    offs: PerSpaxel  # Nuisance offsets per spaxel

    def __init__(
        self,
        n_tiles: int,
        n_spaxels: int,
        offsets: AnyParameter,
        line_centre: AnyParameter,
        n_modes: tuple[int, int],
        A_kernel: Kernel,
        v_kernel: Kernel,
        σ_kernel: Kernel,
        σ_lsf: AnyParameter,
        v_bary: AnyParameter,
        v_syst: AnyParameter,
        C_v_cal: AnyParameter,  # MUST be 2 values i.e. shape is (2,)
        f_cal_unconstrained: AnyParameter,
    ):
        self.offs = Constant(const=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offsets))
        self.line = EmissionLineProduction(
            μ=line_centre,
            A_raw=FourierGP(n_modes=n_modes, kernel=A_kernel),
            v=FourierGP(n_modes=n_modes, kernel=v_kernel),
            vσ_raw=FourierGP(n_modes=n_modes, kernel=σ_kernel),
            σ_lsf=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf),
            v_bary=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary),
            v_syst=v_syst,
            v_cal=WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre),
            f_cal_raw=PerTile(n_tiles=n_tiles, tile_values=f_cal_unconstrained),
        )

    def __call__(self, λ, spatial_data):
        return self.offs(λ, spatial_data) + self.line(λ, spatial_data)


def neg_ln_posterior(model, λ, xy_data, data, u_data, mask):
    vmapped_model = jax.vmap(model, in_axes=(0, None))
    ln_like = ln_likelihood(vmapped_model, λ, xy_data, data, u_data, mask)
    print(ln_like)
    ln_prior = (
        model.line.A_raw.prior_logpdf()
        + model.line.v.prior_logpdf()
        + model.line.vσ_raw.prior_logpdf()
    )
    print(ln_prior)
    return -1 * (ln_like + ln_prior)
