"""make_figures.py - Regenerate the figures for docs/examples/inference.md.

Runs the page's own code (the same blocks tests/test_docs.py executes) with
docs/examples/img as the working directory, so the figures always match the
page. Needs nifty[re] and blackjax:

    uv run python docs/examples/make_figures.py
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))

from test_docs import python_blocks

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    img = HERE / "img"
    img.mkdir(exist_ok=True)
    os.chdir(img)
    namespace: dict = {"__name__": "__docs__"}
    for code in python_blocks(HERE / "inference.md"):
        exec(compile(code, "inference.md", "exec"), namespace)  # noqa: S102 (runs the page's own code)
    print("Wrote", sorted(p.name for p in img.glob("*.png")))
