"""test_optional_deps.py - NIFTy.re, BlackJAX and daft are optional."""

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Make a package unimportable, then use the core: priors, latent space, posterior, gradients
CORE_WITHOUT = """
import sys
for name in {blocked!r}:
    sys.modules[name] = None  # import of this name now raises ImportError

import jax
jax.config.update("jax_enable_x64", True)
from jax import random

import spectracles
import spectracles.priors
import spectracles.inference
import spectracles.nifty
import spectracles.pgm
from spectracles.inference import GaussianLikelihood, LatentSpace, Posterior
from spectracles.nifty import stages, to_nifty
from spectracles.pgm import pgm_graph, plot_pgm
from spectracles import free_after
from tests.priors_models import line_data, line_forward, make_line_model, noisy_data

latent = LatentSpace(make_line_model())
forward = line_forward(line_data())
truth = latent.sample(random.PRNGKey(0))
data, ivar, mask = noisy_data(forward(latent.to_model(truth)), 0.05, 0.1, seed=1)
post = Posterior(latent, forward, GaussianLikelihood(data, ivar, mask))
value, grad = jax.jit(jax.value_and_grad(post.log_density))(truth)
assert bool(jax.numpy.isfinite(value))
st = stages(latent, {{"offset.tile_values": free_after(1)}}, n_iterations=3)
assert st["constants"](0) == ("offset.tile_values",)
assert "data" in pgm_graph(post)

loaded = [m for m, mod in sys.modules.items() if mod is not None and m.split(".")[0] in {blocked!r}]
assert not loaded, loaded

if "nifty" in {blocked!r}:
    try:
        to_nifty(post)
    except ImportError as e:
        assert "spectracles[nifty]" in str(e), str(e)
    else:
        raise AssertionError("to_nifty should need NIFTy")
if "daft" in {blocked!r}:
    try:
        plot_pgm(post)
    except ImportError as e:
        assert "spectracles[pgm]" in str(e), str(e)
    else:
        raise AssertionError("plot_pgm should need daft")
print("OK")
"""


def run_without(*blocked: str) -> subprocess.CompletedProcess:
    script = textwrap.dedent(CORE_WITHOUT.format(blocked=list(blocked)))
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_core_works_without_nifty():
    result = run_without("nifty")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def test_core_works_without_blackjax():
    result = run_without("blackjax")
    assert result.returncode == 0, result.stderr


def test_core_works_without_either():
    result = run_without("nifty", "blackjax")
    assert result.returncode == 0, result.stderr


def test_core_works_without_daft():
    result = run_without("daft")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def test_packaging_declares_nifty_as_extra_only():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    base = " ".join(pyproject["project"]["dependencies"])
    assert "nifty" not in base and "blackjax" not in base and "daft" not in base
    extras = pyproject["project"]["optional-dependencies"]
    assert any(dep.startswith("nifty[re]") for dep in extras["nifty"])
    assert any(dep.startswith("daft-pgm") for dep in extras["pgm"])
