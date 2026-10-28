"""pgm.py - Probabilistic graphical models of spectracles models (optional: pip install spectracles[pgm]).

`pgm_graph` builds the graphical model of a `ShareModule` or a `Posterior` as a
networkx graph; `plot_pgm` draws it with daft. Two views:

- "parameter" (default): the model as usually written. Free parameters are
  random variables with their priors as fixed hyperparameter nodes, a
  `FourierGP` is a random field whose parents are its kernel's
  hyperparameters, and every other module is a deterministic function of its
  inputs.
- "latent": the model as every inference backend sees it. The random
  variables are the standard-normal latents ξ; each parameter is a
  deterministic transform of its latent, and a `FourierGP` is a deterministic
  function of its whitened coefficients and its kernel's hyperparameters.

Edges follow the module tree: a parameter feeds the module that holds it, and
each module feeds the module that holds it. A shared parameter is one node with
an edge to every module that uses it. Kernels are drawn as part of their GP.

Importing this module never imports daft; only `plot_pgm` needs it.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from networkx import DiGraph

from spectracles.model.kernels import Kernel
from spectracles.model.parameter import Known
from spectracles.model.share_module import ShareModule
from spectracles.model.spatial import FourierGP, MultiFourierGP
from spectracles.priors import StdNormal
from spectracles.tree.path_utils import str_to_leafpath, use_path_get_leaf

if TYPE_CHECKING:
    from spectracles.inference.posterior import Posterior

VIEWS = ("parameter", "latent")
GP_TYPES = (FourierGP, MultiFourierGP)
ROOT = "mod:"
DATA = "data"


def _import_daft():
    """Import daft, with a helpful error if it isn't installed."""
    try:
        import daft
    except ImportError as e:
        raise ImportError(
            "Plotting graphical models needs daft, which is an optional dependency of "
            "spectracles. Install it with: pip install 'spectracles[pgm]'"
        ) from e
    return daft


def _parent(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else ""


def _name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _shape_label(shape: tuple[int, ...]) -> str | None:
    if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
        return None
    return "×".join(str(n) for n in shape)


def pgm_graph(
    obj: ShareModule | Posterior,
    view: str = "parameter",
    show_fixed: bool = False,
    show_priors: bool = True,
) -> DiGraph:
    """The probabilistic graphical model of a model or posterior, as a DiGraph.

    Args:
        obj: A `ShareModule`, or a `Posterior` (which adds the observed data).
        view: "parameter" (parameters are the random variables) or "latent"
            (the standard-normal latents ξ are).
        show_fixed: Also show fixed parameters and Known values, as fixed nodes.
        show_priors: Show each prior as a fixed hyperparameter node.

    Returns:
        A DiGraph whose edges point from inputs to the quantities that depend
        on them. Node attributes: `kind` (one of "random", "latent",
        "deterministic", "fixed", "hyper", "observed"), `label`, and `plate`
        (the array shape as text, e.g. "300" or "15×15", or None).
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}.")
    from spectracles.inference.posterior import Posterior

    posterior = obj if isinstance(obj, Posterior) else None
    model = posterior.latent.model if posterior is not None else obj
    if not isinstance(model, ShareModule):
        raise TypeError(f"Expected a ShareModule or a Posterior, got {type(obj).__name__}.")

    locked = model.get_locked_model().model

    def get(path: str) -> Any:
        return locked if path == "" else use_path_get_leaf(locked, str_to_leafpath(path))

    param_parent = model._child_to_parent()
    component_parent = {
        child: parent
        for parent, children in model.get_sharing_summary(level="component").items()
        for child in children
    }

    def canonical_module(path: str) -> str:
        """Resolve shared components to their parent, and kernels to their GP."""
        changed = True
        while changed:
            changed = False
            for child, parent in component_parent.items():
                if path == child or path.startswith(child + "."):
                    path = parent + path[len(child) :]
                    changed = True
        while path and isinstance(get(path), Kernel):
            path = _parent(path)
        return path

    graph = DiGraph()

    def add_module(path: str) -> str:
        """Add a module node and the chain of modules holding it; return its id."""
        node = "mod:" + path
        if node in graph:
            return node
        module = get(path)
        is_gp = isinstance(module, GP_TYPES)
        kind = "random" if (is_gp and view == "parameter") else "deterministic"
        label = type(module).__name__ if path == "" else _name(path)
        graph.add_node(node, kind=kind, label=label, plate=None)
        if path != "":
            graph.add_edge(node, add_module(canonical_module(_parent(path))))
        return node

    add_module("")
    for path in model.get_parameter_paths(show_shared=True, show_knowns=True):
        source = param_parent.get(path, path)
        param = get(source)
        holder = canonical_module(_parent(path))
        is_known = isinstance(param, Known)
        is_fixed = is_known or param.fix
        is_coefficients = isinstance(get(holder), GP_TYPES) and _name(source) == "coefficients"
        if is_fixed and not show_fixed:
            continue
        if is_coefficients and view == "parameter":
            continue  # the GP field node stands for its coefficients

        node = "par:" + source
        plate = _shape_label(tuple(param.val.shape))
        if node not in graph:
            prior = None if is_fixed else param.prior
            if is_fixed:
                kind = "fixed"
            elif view == "latent" and not isinstance(prior, StdNormal):
                kind = "deterministic"
            else:
                kind = "latent" if view == "latent" else "random"
            graph.add_node(node, kind=kind, label=_name(source), plate=plate)
            if view == "latent" and kind == "deterministic":
                xi = "xi:" + source
                graph.add_node(xi, kind="latent", label="ξ " + _name(source), plate=plate)
                graph.add_edge(xi, node)
            if show_priors and prior is not None and not isinstance(prior, StdNormal):
                hyper = "hyp:" + source
                graph.add_node(hyper, kind="hyper", label=repr(prior), plate=None)
                graph.add_edge(hyper, node)
        graph.add_edge(node, add_module(holder))
        # A shared component sits in several modules: connect it to each of them
        prefix = _parent(path)
        while prefix:
            child, parent = canonical_module(prefix), canonical_module(_parent(prefix))
            if child != parent:
                graph.add_edge(add_module(child), add_module(parent))
            prefix = _parent(prefix)

    _disambiguate(graph)
    if posterior is not None:
        lik = posterior.likelihood
        graph.add_node(
            DATA,
            kind="observed",
            label=f"data ({type(lik).__name__.removesuffix('Likelihood')})",
            plate=str(lik.n_used),
        )
        graph.add_edge(ROOT, DATA)
    return graph


def _disambiguate(graph: DiGraph) -> None:
    """Add the owner's path as a second line to labels that occur more than once."""
    from collections import Counter

    counts = Counter(d["label"] for _, d in graph.nodes(data=True))
    for node, d in graph.nodes(data=True):
        if counts[d["label"]] > 1 and node.startswith(("par:", "mod:", "xi:")):
            path = node.split(":", 1)[1]
            owner = ".".join(c for c in _parent(path).split(".") if c != "kernel")
            if owner:
                d["label"] = f"{d['label']}\n{owner}"


def _layout(
    graph: DiGraph, widths: dict[str, float], gap: float
) -> dict[str, tuple[float, float]]:
    """Layered layout: each node one level above the deepest node it feeds."""
    from networkx import topological_sort

    level: dict[str, int] = {}
    for node in reversed(list(topological_sort(graph))):
        level[node] = 1 + max((level[c] for c in graph.successors(node)), default=-1)
    layers: dict[int, list[str]] = {}
    for node, lv in level.items():
        layers.setdefault(lv, []).append(node)

    x: dict[str, float] = {}

    def place(nodes: list[str]) -> None:
        total = sum(widths[n] for n in nodes) + gap * (len(nodes) - 1)
        left = -total / 2
        for n in nodes:
            x[n] = left + widths[n] / 2
            left += widths[n] + gap

    def mean_x(nodes, default: float) -> float:
        nodes = list(nodes)
        return sum(x[m] for m in nodes) / len(nodes) if nodes else default

    # Barycentre ordering: bottom up (towards the nodes each feeds), then
    # alternating sweeps to reduce edge crossings
    for lv in sorted(layers):
        if lv > 0:
            layers[lv].sort(key=lambda n: mean_x(graph.successors(n), 0.0))
        place(layers[lv])
    for _ in range(3):
        for lv in sorted(layers, reverse=True)[1:]:
            layers[lv].sort(key=lambda n: mean_x(graph.predecessors(n), x[n]))
            place(layers[lv])
        for lv in sorted(layers)[1:]:
            layers[lv].sort(key=lambda n: mean_x(graph.successors(n), x[n]))
            place(layers[lv])
    return {n: (x[n], float(level[n])) for n in graph}


def plot_pgm(
    obj: ShareModule | Posterior,
    view: str = "parameter",
    show_fixed: bool = False,
    show_priors: bool = True,
    fontsize: float = 8.0,
    dpi: float | None = None,
) -> Any:
    """Draw the probabilistic graphical model of a model or posterior with daft.

    Random variables are circles (ellipses), deterministic quantities
    rectangles, observed data shaded, and fixed values and prior
    hyperparameters small filled dots. Array-valued nodes sit in a plate
    labelled with their shape. See `pgm_graph` for the arguments and the two
    views.

    Args:
        obj: A `ShareModule`, or a `Posterior` (which adds the observed data).
        view: "parameter" (default) or "latent".
        show_fixed: Also show fixed parameters and Known values.
        show_priors: Show each prior as a hyperparameter node.
        fontsize: Font size of the node labels.
        dpi: Resolution of the rendered figure.

    Returns:
        The rendered `daft.PGM`; save it with `.savefig(...)` or reach the
        matplotlib axes as `.ax`.
    """
    daft = _import_daft()
    import matplotlib as mpl
    import numpy as np

    graph = pgm_graph(obj, view=view, show_fixed=show_fixed, show_priors=show_priors)
    # Sizes and positions share one unit (grid_unit = node_unit = 1 cm)
    char = 0.17 * fontsize / 8  # approximate label width per character, cm
    widths = {
        n: char * max(len(line) for line in d["label"].split("\n"))
        + (0.2 if d["kind"] in ("fixed", "hyper") else 0.5)
        for n, d in graph.nodes(data=True)
    }
    widths = {
        n: w if graph.nodes[n]["kind"] in ("fixed", "hyper") else max(1.0, w)
        for n, w in widths.items()
    }
    # Room for a plate's margins around plated nodes
    slots = {n: w + (0.4 if graph.nodes[n]["plate"] else 0.0) for n, w in widths.items()}
    pos = {n: (x, 1.9 * y) for n, (x, y) in _layout(graph, slots, gap=0.6).items()}

    tex = mpl.rc_context({"text.usetex": False}) if mpl.rcParams["text.usetex"] else nullcontext()
    with tex:
        pgm = daft.PGM(dpi=dpi, node_unit=1.0, grid_unit=1.0)
        for n, d in graph.nodes(data=True):
            x, y = pos[n]
            kind = d["kind"]
            if kind in ("fixed", "hyper"):
                pgm.add_node(n, d["label"], x, y, fixed=True, offset=(0, 6), fontsize=fontsize)
                continue
            pgm.add_node(
                n,
                d["label"],
                x,
                y,
                aspect=widths[n],
                fontsize=fontsize,
                observed=kind == "observed",
                shape="rectangle" if kind == "deterministic" else "ellipse",
            )
            if d["plate"] is not None:
                w = widths[n]
                pgm.add_plate(
                    [x - w / 2 - 0.2, y - 0.95, w + 0.4, 1.55],
                    label=d["plate"],
                    position="bottom right",
                    fontsize=fontsize - 1,
                )
        for a, b in graph.edges():
            pgm.add_edge(a, b)
        with np.errstate(divide="ignore"):  # daft divides by dx for vertical edges
            pgm.render(dpi=dpi)
    return pgm
