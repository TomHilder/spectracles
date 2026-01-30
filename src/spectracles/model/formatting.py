"""formatting.py - Pretty printing utilities with Catppuccin-inspired colors."""

from typing import Any

from rich.console import Console
from rich.text import Text
from rich.tree import Tree
from rich.table import Table
from rich.theme import Theme
import jax.numpy as jnp
from jaxtyping import Array


# Catppuccin Mocha-inspired theme
THEME = Theme({
    "type": "#89b4fa",       # Blue - class/type names
    "value": "#cdd6f4",      # Text - values
    "free": "#a6e3a1",       # Green - free parameters
    "fixed": "#f9e2af",      # Yellow - fixed parameters
    "shared": "#cba6f7",     # Mauve - shared references
    "tree": "#6c7086",       # Overlay0 - tree branches
    "dim": "#6c7086",        # Overlay0 - dimmed text
    "error": "#f38ba8",      # Red - errors/warnings
    "success": "#a6e3a1",    # Green - success
})

# Console with theme, but force_terminal=False to allow graceful fallback
_console = Console(theme=THEME, force_terminal=False)


def get_console() -> Console:
    """Get the configured Rich console."""
    return _console


# Maximum number of array elements to show values for
ARRAY_DISPLAY_THRESHOLD = 6


def format_array_value(arr: Array) -> str:
    """Format an array value for display.

    Small arrays (<=6 elements) show values, large arrays show dtype+shape.
    """
    if arr is None:
        return "None"

    size = arr.size

    if size <= ARRAY_DISPLAY_THRESHOLD:
        # Show actual values for small arrays
        flat_arr = arr.flatten()
        if size == 1:
            # Scalar-like: show single value
            return f"{float(flat_arr[0]):.4g}"
        else:
            # Small array: show all values
            vals = ", ".join(f"{float(v):.4g}" for v in flat_arr)
            return f"[{vals}]"
    else:
        # Large array: show dtype and shape
        dtype_str = str(arr.dtype).replace("float", "f").replace("int", "i")
        shape_str = ",".join(str(s) for s in arr.shape)
        return f"{dtype_str}[{shape_str}]"


def format_constraint(lower: float | None, upper: float | None, log: bool) -> str:
    """Format constraint bounds in x < notation."""
    if log:
        return "log"
    elif lower is not None and upper is not None:
        return f"{lower} < x < {upper}"
    elif lower is not None:
        return f"x > {lower}"
    elif upper is not None:
        return f"x < {upper}"
    else:
        return ""


def styled_type(name: str) -> Text:
    """Create a styled type/class name."""
    return Text(name, style="type")


def styled_value(value: str) -> Text:
    """Create a styled value."""
    return Text(value, style="value")


def styled_free() -> Text:
    """Create styled 'free' text."""
    return Text("free", style="free")


def styled_fixed() -> Text:
    """Create styled 'fixed' text."""
    return Text("fixed", style="fixed")


def styled_shared(parent_path: str) -> Text:
    """Create styled shared reference."""
    text = Text("→ ", style="shared")
    text.append(parent_path, style="shared")
    return text


def styled_dim(text: str) -> Text:
    """Create dimmed text."""
    return Text(text, style="dim")


def render_to_string(renderable: Any) -> str:
    """Render a Rich object to a plain string (with ANSI codes if terminal supports)."""
    console = get_console()
    with console.capture() as capture:
        console.print(renderable, end="")
    return capture.get()


def parameter_repr_parts(
    class_name: str,
    value_str: str,
    is_fixed: bool,
    constraint_str: str | None = None,
    shared_path: str | None = None,
) -> Text:
    """Build a styled parameter repr.

    Args:
        class_name: The class name (Parameter, ConstrainedParameter, Known)
        value_str: The formatted value string
        is_fixed: Whether the parameter is fixed
        constraint_str: Optional constraint description (e.g., "x > 0")
        shared_path: If this value is shared, the parent path

    Returns:
        Rich Text object with styled repr
    """
    text = styled_type(class_name)
    text.append("(", style="dim")

    if shared_path:
        text.append_text(styled_shared(shared_path))
    else:
        text.append(value_str, style="value")

    # Add constraint if present
    if constraint_str:
        text.append(", ", style="dim")
        text.append(constraint_str, style="dim")

    # Add fixed/free status
    text.append(", ", style="dim")
    if is_fixed:
        text.append_text(styled_fixed())
    else:
        text.append_text(styled_free())

    text.append(")", style="dim")
    return text


def known_repr_parts(value_str: str) -> Text:
    """Build a styled Known repr (simpler, no status needed)."""
    text = styled_type("Known")
    text.append("(", style="dim")
    text.append(value_str, style="value")
    text.append(")", style="dim")
    return text


def shared_repr_parts(parent_path: str) -> Text:
    """Build a styled Shared repr."""
    text = styled_type("Shared")
    text.append(" → ", style="shared")
    text.append(parent_path, style="shared")
    return text


def create_gradient_table() -> Table:
    """Create a styled gradient summary table."""
    table = Table(
        title="Gradient Summary",
        show_header=True,
        header_style="bold type",
        border_style="dim",
    )
    table.add_column("Parameter Path", style="value", no_wrap=True)
    table.add_column("Norm", justify="right", style="value")
    table.add_column("Max", justify="right", style="value")
    table.add_column("Issues", justify="center")
    return table


def create_model_tree(root_name: str, root_type: str) -> Tree:
    """Create a styled model tree with root node."""
    label = Text()
    label.append(root_name, style="value")
    label.append(" (", style="dim")
    label.append(root_type, style="type")
    label.append(")", style="dim")
    return Tree(label, guide_style="tree")


def add_tree_node(tree: Tree, name: str, content: Text | str) -> Tree:
    """Add a styled node to the tree."""
    label = Text()
    label.append(name, style="value")
    label.append(": ", style="dim")
    if isinstance(content, str):
        label.append(content)
    else:
        label.append_text(content)
    return tree.add(label)
