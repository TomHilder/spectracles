"""test_docs.py - Run the code in the docs pages, so their snippets can't rot.

Every ```python block on a page is executed in order, in one namespace per
page. A block whose first line is `# sketch` is illustrative only and skipped.
A line like `n_draws = 600  # fast in tests: 20` runs with the faster value
here; the page (and scripts that run it) keep the real one.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"

# Pages whose code is run, and the optional packages they need
PAGES = {
    "inference.md": ("nifty.re", "optax", "arviz"),
    "examples/inference.md": ("nifty.re", "blackjax", "optax"),
}

BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
# Only plain assignments: `name = value  # fast in tests: other`
FAST = re.compile(r"^(\s*\w+\s*=\s*)[^#\n]+?\s*#\s*fast in tests:\s*(\S+)\s*$", re.MULTILINE)


def python_blocks(page: Path) -> list[str]:
    return [b for b in BLOCK.findall(page.read_text()) if not b.startswith("# sketch")]


def fast(code: str) -> str:
    """Swap in the `# fast in tests:` values."""
    return FAST.sub(lambda m: f"{m.group(1)}{m.group(2)}", code)


@pytest.mark.parametrize("page", sorted(PAGES))
def test_docs_page_runs(page, tmp_path, monkeypatch):
    for module in PAGES[page]:
        pytest.importorskip(module)
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)  # anything a page writes stays out of the repo
    blocks = [fast(b) for b in python_blocks(DOCS / page)]
    assert blocks, f"no runnable code found in {page}"
    namespace: dict = {"__name__": "__docs__"}
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"{page} [block {i}]", "exec"), namespace)  # noqa: S102 (the point of the test)
        except Exception as e:
            raise AssertionError(f"{page}, code block {i} failed:\n\n{code}") from e
