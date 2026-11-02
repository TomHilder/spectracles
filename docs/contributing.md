# Contributing

Contributions to **spectracles** are welcome, whether a bug report, a fix, a new model component or a clearer page in these docs. This page describes how to set up a copy to work on and what a pull request needs.

## Setting up

spectracles uses [uv](https://github.com/astral-sh/uv) to manage its environment. Clone the repository and install it with its development dependencies:

``` bash
git clone https://github.com/TomHilder/spectracles.git
cd spectracles
uv sync
```

`uv sync` builds [jax-finufft](https://github.com/flatironinstitute/jax-finufft) from source, from its git repository, so you need [FFTW](https://www.fftw.org) installed first (for example `brew install fftw` on macOS, or `libfftw3-dev` on Debian and Ubuntu). The development dependencies include NIFTy.re, BlackJAX and astropy, so the whole test suite can run.

Run any command in the environment with `uv run`, e.g. `uv run python my_script.py`.

## Running the tests

``` bash
uv run pytest
```

runs everything. To run one file, one class or one test:

``` bash
uv run pytest tests/test_spatial.py
uv run pytest tests/test_spatial.py::TestFourierGP
uv run pytest tests/test_spatial.py::TestFourierGP::test_realised_covariance
```

Add `-x` to stop at the first failure, or `-k word` to run only tests whose names contain `word`. Tests run in float64. New functionality should come with tests.

## Linting and formatting

The code is linted and formatted with [Ruff](https://docs.astral.sh/ruff/), with a line length of 99:

``` bash
uv run ruff check .
uv run ruff format .
```

`uv run ruff check --fix .` fixes what it safely can. CI runs `ruff check .` and `ruff format --check .`, so run both before pushing.

## Type checking

The package is checked with [pyright](https://github.com/microsoft/pyright), which isn't in the development dependencies; install it however you like (for example `pipx install pyright`). Then

``` bash
pyright src/spectracles
```

should report no errors. `pyproject.toml` points pyright at the project's `.venv` and excludes `spectracles.lvm_models`, which is experimental and not type-checked.

## Documentation

The documentation is Markdown in `docs/`, built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/), and the API reference is generated from docstrings by [mkdocstrings](https://mkdocstrings.github.io/). Preview it with live reload:

``` bash
uv run --with mkdocs-material --with "mkdocstrings[python]" mkdocs serve
```

and open the address it prints. Before opening a pull request, check that it builds without warnings:

``` bash
uv run --with mkdocs-material --with "mkdocstrings[python]" mkdocs build --strict
```

New pages go in the `nav` section of `mkdocs.yml`. The docs are deployed when a version is tagged.

### Code on the docs pages is tested

`tests/test_docs.py` runs the ```` ```python ```` blocks of the guide and example pages, in order, as one script per page, so the examples can't go stale. Pages are listed in its `PAGES` dict with any optional packages they need. If you add a guide page with code, add it there too.

- A block whose first line is `# sketch` is illustrative only and is skipped.
- A line like `n_draws = 600  # fast in tests: 20` runs with the faster value in the tests, while readers see the real one.

Run just these tests with `uv run pytest tests/test_docs.py`.

### Docstrings

Docstrings use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`, `Returns:`, `Raises:` sections), which the API reference renders. Describe what the code does and how to use it. Leave out its history (what it used to do, why it changed, how it was tested): that belongs in commit messages and the changelog.

## Pull requests

1. Create a branch from `main` (on a fork, if you don't have write access).
2. Make your changes, with tests and docs as needed.
3. Add an entry to the [changelog](changelog.md) under **Unreleased**, in the right group (Added, Changed, Deprecated, Removed or Fixed). Say plainly if the change affects results or breaks existing code.
4. Open a pull request against `main`.

CI must pass: the lint job (`ruff check` and `ruff format --check`) and the tests on Python 3.11, 3.12, 3.13 and 3.14.

## Reporting issues

Report bugs and request features through [GitHub Issues](https://github.com/TomHilder/spectracles/issues). For a bug, include your Python, JAX and spectracles versions, a small example that shows the problem, and the full error message.
