"""test_pgm.py - Probabilistic graphical models of models and posteriors."""

import equinox as eqx
import jax.numpy as jnp
import matplotlib as mpl
import pytest

from spectracles import (
    FourierGP,
    Known,
    Matern32,
    Parameter,
    PerSpaxel,
    SpatialDataGeneric,
    build_model,
)
from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior
from spectracles.pgm import pgm_graph
from spectracles.priors import Laplace, LogNormal

N = 20


class Line(eqx.Module):
    flux: FourierGP
    v: FourierGP

    def __call__(self, data):
        return jnp.exp(self.flux(data)) + 0.0 * self.v(data)


class TwoLines(eqx.Module):
    ha: Line
    nii: Line
    offset: PerSpaxel
    sigma: Parameter
    scale: Known

    def __call__(self, data):
        return (
            self.scale.val * (self.ha(data) + self.nii(data)) + self.offset(data) + self.sigma.val
        )


def kernel():
    return Matern32(Parameter(prior=LogNormal(0.5, 0.2)), Parameter(prior=LogNormal(1.0, 0.5)))


def make_model():
    v = FourierGP((5, 5), kernel())  # one velocity field, used by both lines
    return build_model(
        TwoLines,
        ha=Line(FourierGP((5, 5), kernel()), v),
        nii=Line(FourierGP((5, 5), kernel()), v),
        offset=PerSpaxel(N, Parameter(dims=N, prior=Laplace(0.02))),
        sigma=Parameter(prior=LogNormal(1.0, 0.3)),
        scale=Known(2.0),
    )


def make_posterior():
    latent = LatentSpace(make_model())
    data = SpatialDataGeneric(x=jnp.linspace(-1, 1, N), y=jnp.zeros(N), idx=jnp.arange(N))
    mask = jnp.arange(N) != 3
    likelihood = GaussianLikelihood(jnp.zeros(N), jnp.ones(N), mask)
    return Posterior(latent, forward=lambda m: m(data), likelihood=likelihood)


def kinds(graph):
    return {n: d["kind"] for n, d in graph.nodes(data=True)}


def test_parameter_view_structure():
    g = pgm_graph(make_model())
    k = kinds(g)
    # Hyperparameters are random and feed their GP, which is a random field
    assert k["par:ha.flux.kernel.length_scale"] == "random"
    assert k["mod:ha.flux"] == "random"
    assert g.has_edge("par:ha.flux.kernel.length_scale", "mod:ha.flux")
    # Kernels are part of their GP; coefficients are represented by the field
    assert not any(n.endswith(".kernel") for n in g)
    assert not any("coefficients" in n for n in g)
    # Other modules are deterministic and feed their parent, up to the root
    assert k["mod:ha"] == "deterministic"
    assert g.has_edge("mod:ha.flux", "mod:ha") and g.has_edge("mod:ha", "mod:")
    # Priors are hyperparameter nodes; fixed values are hidden by default
    assert k["hyp:sigma"] == "hyper" and g.nodes["hyp:sigma"]["label"] == "LogNormal(1, 0.3)"
    assert g.has_edge("hyp:sigma", "par:sigma")
    assert "par:scale" not in g and not any("zeromode" in n for n in g)
    # Arrays are plated with their shape
    assert g.nodes["par:offset.spaxel_values"]["plate"] == str(N)
    assert g.nodes["par:sigma"]["plate"] is None


def test_shared_component_is_one_node_feeding_every_user():
    g = pgm_graph(make_model())
    assert "mod:ha.v" in g and "mod:nii.v" not in g
    assert g.has_edge("mod:ha.v", "mod:ha") and g.has_edge("mod:ha.v", "mod:nii")
    assert "par:nii.v.kernel.length_scale" not in g
    assert set(g.successors("par:ha.v.kernel.length_scale")) == {"mod:ha.v"}


def test_shared_parameter_is_one_node():
    shared = Parameter(prior=LogNormal(0.5, 0.2))
    k1 = Matern32(shared, Parameter(prior=LogNormal(1.0, 0.5)))
    k2 = Matern32(shared, Parameter(prior=LogNormal(1.0, 0.5)))
    v = FourierGP((5, 5), kernel())
    model = build_model(
        TwoLines,
        ha=Line(FourierGP((5, 5), k1), v),
        nii=Line(FourierGP((5, 5), k2), v),
        offset=PerSpaxel(N, Parameter(dims=N, prior=Laplace(0.02))),
        sigma=Parameter(prior=LogNormal(1.0, 0.3)),
        scale=Known(2.0),
    )
    g = pgm_graph(model)
    ls = [n for n in g if n.startswith("par:") and n.endswith("flux.kernel.length_scale")]
    assert len(ls) == 1
    assert set(g.successors(ls[0])) == {"mod:ha.flux", "mod:nii.flux"}


def test_latent_view_structure():
    g = pgm_graph(make_model(), view="latent")
    k = kinds(g)
    # Each parameter is a transform of its latent; GPs are deterministic
    assert k["xi:ha.flux.kernel.length_scale"] == "latent"
    assert k["par:ha.flux.kernel.length_scale"] == "deterministic"
    assert g.has_edge("xi:ha.flux.kernel.length_scale", "par:ha.flux.kernel.length_scale")
    assert k["mod:ha.flux"] == "deterministic"
    # Whitened coefficients are the latents themselves (StdNormal), plated
    assert k["par:ha.flux.coefficients"] == "latent"
    assert "xi:ha.flux.coefficients" not in g
    assert g.nodes["par:ha.flux.coefficients"]["plate"] == "5×5"


def test_show_fixed_and_show_priors():
    g = pgm_graph(make_model(), show_fixed=True, show_priors=False)
    assert kinds(g)["par:scale"] == "fixed"
    assert kinds(g)["par:ha.flux.zeromode"] == "fixed"
    assert not any(d["kind"] == "hyper" for _, d in g.nodes(data=True))


def test_duplicate_names_get_their_owner():
    g = pgm_graph(make_model())
    assert g.nodes["par:ha.flux.kernel.length_scale"]["label"] == "length_scale\nha.flux"
    assert g.nodes["mod:nii.flux"]["label"] == "flux\nnii"
    assert g.nodes["par:sigma"]["label"] == "sigma"


def test_posterior_adds_observed_data():
    g = pgm_graph(make_posterior())
    assert kinds(g)["data"] == "observed"
    assert g.has_edge("mod:", "data")
    assert g.nodes["data"]["plate"] == str(N - 1)  # used pixels only
    assert g.nodes["mod:"]["label"] == "TwoLines"


def test_graph_is_acyclic_with_one_sink():
    from networkx import is_directed_acyclic_graph

    for view in ("parameter", "latent"):
        g = pgm_graph(make_posterior(), view=view, show_fixed=True)
        assert is_directed_acyclic_graph(g)
        assert [n for n in g if g.out_degree(n) == 0] == ["data"]


def test_bad_view_and_type():
    with pytest.raises(ValueError, match="view"):
        pgm_graph(make_model(), view="centred")
    with pytest.raises(TypeError):
        pgm_graph(Parameter(1.0))  # type: ignore[arg-type]


@pytest.mark.parametrize("view", ["parameter", "latent"])
def test_plot_pgm_renders(view, tmp_path):
    pytest.importorskip("daft")
    post = make_posterior()
    with mpl.rc_context({"text.usetex": True}):  # labels with underscores must still draw
        pgm = post.plot_pgm(view=view)
    assert {"data", "mod:", "par:sigma"} <= set(pgm._nodes)
    pgm.savefig(tmp_path / "pgm.png")
    assert (tmp_path / "pgm.png").stat().st_size > 0
    assert make_model().plot_pgm(view=view) is not None
