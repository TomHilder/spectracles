"""line_dev.py - Developmental spectrospatial model of a single emission line for LVM."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from spectracles import (
    AnyParameter,
    FourierGP,
    Kernel,
    PerSpaxel,
    SpatialDataLVM,
    SpatialModel,
    SpectralSpatialModel,
    l_bounded,
)
from spectracles.lvm_models.constants import C_KMS
from spectracles.lvm_models.likelihood import ln_likelihood
from spectracles.model.data import SpatialData
from spectracles.model.spatial import PerIFU, PerIFUAndTile, PerTile

A_LOWER = 0.0

N_FIBRES_LVM = 1944


def hermite3(x: Array) -> Array:
    return x**3 - 3.0 * x


def hermite4(x: Array) -> Array:
    return x**4 - 6.0 * x**2 + 3.0


def hermite5(x: Array) -> Array:
    return x**5 - 10.0 * x**3 + 15 * x


def hermite6(x: Array) -> Array:
    return x**6 - 15 * x**4 + 45 * x**2 - 15


# Dimensionless coordinate: use sigma = sqrt(w2_obs)
# x = (velocities - v_obs) / jnp.sqrt(w2_obs)

# # Effective GH coefficients from transformed GPs
# h3 = self.h3(spatial_data)
# h4 = self.h4(spatial_data)

# # Gaussian line profile
# gaussian = jnp.exp(-0.5 * (velocities - v_obs) ** 2 / w2_obs)

# # Hermite polynomials
# H3 = hermite3(x)
# H4 = hermite4(x)

# # Gauss–Hermite line profile
# return peak * gaussian * (1.0 + h3 * H3 + h4 * H4)


class WaveCalVelocity(SpatialModel):
    # Model parameters
    C_v_cal: AnyParameter  # 2 value model parameter
    # Constants
    μ: AnyParameter  # line centre in Angstroms

    def __call__(self, s: SpatialData) -> Array:
        v0 = 0.0  # effectively pinned to v_syst
        v1 = v0 - C_KMS * (self.C_v_cal.val[0] / self.μ.val[0])
        v2 = v1 - C_KMS * (self.C_v_cal.val[1] / self.μ.val[0])
        v_ifu_vals = jnp.array([v0, v1, v2])
        return v_ifu_vals[s.ifu_idx]


# class FluxCalFactor(SpatialModel):
#     # f_cal_raw: PerIFUAndTile  # Unconstrained flux calibration factor per tile and ifu
#     f_cal_raw: PerTile  # Unconstrained flux calibration factor per tile

#     # def __init__(self, n_tiles, ifu_values):
#     def __init__(self, n_tiles, tile_values):
#         # self.f_cal_raw = PerIFUAndTile(n_tiles=n_tiles, n_ifus=3, ifu_values=ifu_values)
#         self.f_cal_raw = PerTile(n_tiles=n_tiles, tile_values=tile_values)

#     def __call__(self, s: SpatialData) -> Array:
#         return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class FluxCalFactor(SpatialModel):
    # Hierarchical flux calibration factor per tile, with per-IFU variations
    f_cal_raw: PerTile  # Unconstrained flux calibration factor per tile # N_TILES values
    delta_f_cal: PerIFU  # Additive per-IFU variation in flux calibration factor # 3 values

    def __init__(self, n_tiles, tile_values, ifu_values):
        self.f_cal_raw = PerTile(n_tiles=n_tiles, tile_values=tile_values)
        self.delta_f_cal = PerIFU(n_ifus=3, ifu_values=ifu_values)

    def __call__(self, s: SpatialData) -> Array:
        return l_bounded(self.f_cal_raw(s) + self.delta_f_cal(s), lower=0.0) / l_bounded(
            0, lower=0.0
        )


class StellarContinuum(SpatialModel):
    offs: PerSpaxel

    def __init__(self, n_spaxels: int, offs: AnyParameter):
        self.offs = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offs)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.offs(s)


class LSFScatter(SpatialModel):
    Δσ_lsf: PerSpaxel

    def __init__(self, n_spaxels: int, delta_sigma_lsf: AnyParameter):
        self.Δσ_lsf = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=delta_sigma_lsf)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.Δσ_lsf(s)


class LineFluxScatter(SpatialModel):
    ΔA: PerSpaxel

    def __init__(self, n_spaxels: int, delta_A: AnyParameter):
        self.ΔA = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=delta_A)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.ΔA(s)


class VelocityScatter(SpatialModel):
    Δv: PerSpaxel

    def __init__(self, n_spaxels: int, delta_v: AnyParameter):
        self.Δv = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=delta_v)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.Δv(s)


class SmoothContinuum(SpatialModel):
    # Model components
    A_raw: SpatialModel  # Unconstrained line flux, shared with the emission line
    cont_residual_raw: SpatialModel  # "residual" GP part of the continuum
    eps_1: AnyParameter  # Continuum level as a fraction of line peak
    eps_2: AnyParameter  # Continuum level for residual GP (as a fraction of line peak if sharing kernel with A_raw, effectively scales down the variance)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.eps_1.val * self.A(s) + self.eps_2.val * self.cont_residual(s)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def cont_residual(self, s) -> Array:
        return l_bounded(self.cont_residual_raw(s), lower=A_LOWER)


# class SkyBackground(SpatialModel):
#     # level: PerIFUAndTile  # Sky background level per tile and IFU
#     level: PerTile  # Sky background level per tile

#     # def __init__(self, n_tiles, ifu_values):
#     def __init__(self, n_tiles, tile_values):
#         # self.level = PerIFUAndTile(n_tiles=n_tiles, n_ifus=3, ifu_values=ifu_values)
#         self.level = PerTile(n_tiles=n_tiles, tile_values=tile_values)

#     def __call__(self, s: SpatialDataLVM) -> Array:
#         return self.level(s)


class SkyBackground(SpatialModel):
    # Hierarchical sky background per tile, with per-IFU variations
    level: PerTile  # Sky background level per tile # N_TILES values
    delta_level: PerIFU  # Additive per-IFU variation in sky background # 3 values

    def __init__(self, n_tiles, tile_values, ifu_values):
        self.level = PerTile(n_tiles=n_tiles, tile_values=tile_values)
        self.delta_level = PerIFU(n_ifus=3, ifu_values=ifu_values)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.level(s) + self.delta_level(s)


class CrossTalk(eqx.Module):
    η: AnyParameter  # Amount of cross-talk

    n_tiles: int  # Number of tiles/pointings
    n_fibres: int  # Number of fibres per pointing (MORE than 1801)

    # def mixing_matrix(self) -> Array:
    #     diag = jnp.repeat(1 - 2 * self.η.val, self.n_fibres)
    #     k1_diag = jnp.repeat(self.η.val, self.n_fibres - 1)
    #     return jnp.diag(diag, k=0) + jnp.diag(k1_diag, k=-1) + jnp.diag(k1_diag, k=1)

    # def __call__(self, s: SpatialData, flux: Array):
    #     in_flux = jnp.zeros((flux.shape[0], self.n_fibres, self.n_tiles))
    #     in_flux = in_flux.at[:, s.fib_idx, s.tile_idx].set(flux)
    #     out_flux = jnp.einsum("ij,kjl->kil", self.mixing_matrix(), in_flux)
    #     # Return to original shape (n_lambda, n_spaxels)
    #     out_flux_flat = jnp.zeros_like(flux)
    #     out_flux_flat = out_flux_flat.at[:, s.idx].set(out_flux[:, s.fib_idx, s.tile_idx])
    #     # return in_flux, out_flux, out_flux_flat
    #     return out_flux_flat

    def __call__(self, s: SpatialData, flux: Array):
        # NOTE: LLM magic. Need to check carefully.
        # Ensure flux is 2D: (n_lambda, n_spaxels)
        input_shape = flux.shape
        flux_2d = flux.reshape(-1, flux.shape[-1]) if flux.ndim > 1 else flux[None, :]

        def process_tile(tile_id, out_flux_acc):
            tile_mask = s.tile_idx == tile_id

            # Populate array for this tile at the correct FIBER positions
            in_flux_tile = jnp.zeros((flux_2d.shape[0], self.n_fibres))
            # Use scatter to place flux values at their fiber indices
            in_flux_tile = in_flux_tile.at[:, s.fib_idx].add(
                jnp.where(tile_mask[None, :], flux_2d, 0.0)
            )

            # Apply crosstalk directly: f_out[i] = (1-2η)*f[i] + η*f[i-1] + η*f[i+1]
            out_flux_tile = (1 - 2 * self.η.val) * in_flux_tile
            out_flux_tile = out_flux_tile.at[:, 1:].add(self.η.val * in_flux_tile[:, :-1])
            out_flux_tile = out_flux_tile.at[:, :-1].add(self.η.val * in_flux_tile[:, 1:])

            # Extract back from fiber positions to spaxel positions
            extracted = out_flux_tile[:, s.fib_idx]

            # Only update where this tile's mask is True
            out_flux_updated = jnp.where(tile_mask[None, :], extracted, out_flux_acc)
            return out_flux_updated

        out_flux_flat = jax.lax.fori_loop(0, self.n_tiles, process_tile, jnp.zeros_like(flux_2d))

        return out_flux_flat.reshape(input_shape)


class EmissionLine(SpectralSpatialModel):
    # Line centre in Angstroms
    μ: AnyParameter
    # Model components / line quantities
    A_raw: SpatialModel  # Unconstrained line flux
    v: SpatialModel  # Radial velocity in rest frame in km/s
    vσ_raw: SpatialModel  # Broadening velocity in km/s before constraint
    vσ_raw_mean: AnyParameter  # Mean of vσ_raw GP
    # Scatter in line quantities
    ΔA: SpatialModel  # Line flux scatter
    Δv: SpatialModel  # Velocity scatter
    # Measured line quantities
    σ_lsf: SpatialModel  # LSF width (std dev) in Angstroms
    v_bary: SpatialModel  # Barycentric velocity CORRECTION in km/s
    # Systematics
    v_syst: AnyParameter  # Systematic velocity offset in km/s
    v_cal: WaveCalVelocity  # Per-IFU Velocity calibration offset in km/s
    Δσ_lsf: SpatialModel  # LSF scatter in Angstroms
    h3_raw: SpatialModel  # Skewness per-spaxel
    h3_max: AnyParameter  # max h3
    h4_raw: SpatialModel  # Kurtosis per-spaxel
    h4_max: AnyParameter  # max h4
    h5_raw: SpatialModel
    h5_max: AnyParameter
    h6_raw: AnyParameter
    h6_max: AnyParameter

    def __call__(self, λ: Array, s: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(s)
        σ2_obs = self.σ2_obs(s, μ_obs)
        peak = self.A_total(s) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        x = (λ - μ_obs) / jnp.sqrt(σ2_obs)
        skew = self.h3(s) * hermite3(x)
        kurt = self.h4(s) * hermite4(x)
        skew2 = self.h5(s) * hermite5(x)
        kurt2 = self.h6(s) * hermite6(x)
        return peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs) * (1 + skew + kurt + skew2 + kurt2)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def A_total(self, s) -> Array:
        return self.A(s) + self.ΔA(s)

    def vσ(self, s) -> Array:
        return l_bounded(self.vσ_raw(s) + self.vσ_raw_mean.val, lower=0.0)

    def v_total(self, s) -> Array:
        return self.v(s) + self.Δv(s)

    def v_obs(self, s) -> Array:
        # return self.v(s) + self.v_syst.val + self.v_cal(s) - self.v_bary(s)  # + self.Δv(s)
        return self.v_total(s) + self.v_syst.val + self.v_cal(s) - self.v_bary(s)

    def μ_obs(self, s) -> Array:
        return self.μ.val * (1 + self.v_obs(s) / C_KMS)

    def σ2_obs(self, s, μ_obs) -> Array:
        return (self.vσ(s) * μ_obs / C_KMS) ** 2 + (self.σ_lsf(s) + self.Δσ_lsf(s)) ** 2

    def f_cal(self, s) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)

    def h3(self, s) -> Array:
        z3 = self.h3_raw(s)
        return self.h3_max.val * jnp.tanh(z3)

    def h4(self, s) -> Array:
        z4 = self.h4_raw(s)
        return self.h4_max.val * jnp.tanh(z4)

    def h5(self, s) -> Array:
        z5 = self.h5_raw(s)
        return self.h5_max.val * jnp.tanh(z5)

    def h6(self, s) -> Array:
        z6 = self.h6_raw(s)
        return self.h6_max.val * jnp.tanh(z6)


class LVMModelSingle(SpectralSpatialModel):
    # Model components
    line: EmissionLine  # Emission line model
    stars: StellarContinuum  # Nuisance offsets per spaxel
    cont: SmoothContinuum  # Smooth continuum model

    # Calibration/Nuisances
    flux_cal: FluxCalFactor  # Flux calibration factor per tile
    sky: SkyBackground  # Sky background per tile and IFU
    crosstalk: CrossTalk  # Fibre-fibre crosstalk

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
        delta_f_cal_unconstrained: AnyParameter,
        eps_1: AnyParameter,  # Continuum level as a fraction of line peak
        eps_2: AnyParameter,  # Continuum residual GP level as a fraction of line peak level
        sky_level: AnyParameter,
        delta_sky_level: AnyParameter,
        Δσ_lsf: AnyParameter,
        ΔA: AnyParameter,
        Δv: AnyParameter,
        vσ_raw_mean: AnyParameter,
        h3_raw: AnyParameter,
        h3_max: AnyParameter,
        h4_raw: AnyParameter,
        h4_max: AnyParameter,
        h5_raw: AnyParameter,
        h5_max: AnyParameter,
        h6_raw: AnyParameter,
        h6_max: AnyParameter,
        η_ct: AnyParameter,
    ):
        # Latent GPs for line properties
        A_raw_ = FourierGP(n_modes=n_modes, kernel=A_kernel)
        v_ = FourierGP(n_modes=n_modes, kernel=v_kernel)
        vσ_raw_ = FourierGP(n_modes=n_modes, kernel=σ_kernel)

        # Scatter in line properties (not currently used)
        ΔA_ = LineFluxScatter(n_spaxels=n_spaxels, delta_A=ΔA)
        Δv_ = VelocityScatter(n_spaxels=n_spaxels, delta_v=Δv)

        # Latent GP for residual continuum component (shares kernel with A_raw)
        cont_residual_raw_ = FourierGP(n_modes=n_modes, kernel=A_kernel)

        # Per-spaxel stuff that is measured, not fit for
        σ_lsf_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf)
        v_bary_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary)

        # Calibration / nuisance components for line
        v_cal_ = WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre)
        Δσ_lsf_ = LSFScatter(n_spaxels=n_spaxels, delta_sigma_lsf=Δσ_lsf)
        h3_raw_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=h3_raw)
        h4_raw_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=h4_raw)
        h5_raw_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=h5_raw)
        h6_raw_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=h6_raw)

        self.stars = StellarContinuum(n_spaxels=n_spaxels, offs=offsets)
        self.line = EmissionLine(
            μ=line_centre,
            A_raw=A_raw_,
            v=v_,
            ΔA=ΔA_,
            Δv=Δv_,
            vσ_raw=vσ_raw_,
            vσ_raw_mean=vσ_raw_mean,
            σ_lsf=σ_lsf_,
            v_bary=v_bary_,
            v_syst=v_syst,
            v_cal=v_cal_,
            Δσ_lsf=Δσ_lsf_,
            h3_raw=h3_raw_,
            h3_max=h3_max,
            h4_raw=h4_raw_,
            h4_max=h4_max,
            h5_raw=h5_raw_,
            h5_max=h5_max,
            h6_raw=h6_raw_,
            h6_max=h6_max,
        )
        self.cont = SmoothContinuum(
            A_raw=A_raw_,
            cont_residual_raw=cont_residual_raw_,
            eps_1=eps_1,
            eps_2=eps_2,
        )
        # self.flux_cal = FluxCalFactor(n_tiles=n_tiles, ifu_values=f_cal_unconstrained)
        # self.sky = SkyBackground(n_tiles=n_tiles, ifu_values=sky_level)
        self.flux_cal = FluxCalFactor(
            n_tiles=n_tiles,
            tile_values=f_cal_unconstrained,
            ifu_values=delta_f_cal_unconstrained,
        )
        self.sky = SkyBackground(
            n_tiles=n_tiles,
            tile_values=sky_level,
            ifu_values=delta_sky_level,
        )
        self.crosstalk = CrossTalk(
            η=η_ct,
            n_tiles=n_tiles,
            n_fibres=N_FIBRES_LVM,
        )

    def __call__(self, λ, spatial_data):
        flux = self.cont(spatial_data) + self.stars(spatial_data) + self.line(λ, spatial_data)
        spectrum = self.crosstalk(spatial_data, flux)
        return self.flux_cal(spatial_data) * spectrum + self.sky(spatial_data)


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
