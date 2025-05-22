"""graph.py - Graph utilities for the modelling framework."""

from contextlib import contextmanager

import matplotlib as mpl
from networkx import DiGraph

DEFAULT_NX_KWDS = {
    "node_size": 8_000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 2,
    "arrowsize": 20,
    "font_size": 8,
    "font_color": "black",
}


@contextmanager
def temporarily_disable_tex():
    prev_setting = mpl.rcParams["text.usetex"]
    mpl.rcParams["text.usetex"] = False
    try:
        yield
    finally:
        mpl.rcParams["text.usetex"] = prev_setting


def print_graph(graph: DiGraph, root_id: int, indent: str = "", is_last: bool = True) -> None:
    # Get info for current node
    node_data = graph.nodes[root_id]
    name = node_data["name"]
    node_type = node_data["type"]
    # Format display text
    if name is None:
        # Root module without parent attribute name
        display_text = node_type
    else:
        # Regular format: "name (Type)"
        display_text = f"{name} ({node_type})"
    # Print this node
    print(f"{indent}{'└── ' if is_last else '├── '}{display_text}")
    # Find child nodes (those pointing to this node)
    children = []
    for src, dst in graph.edges():
        if src == root_id:
            children.append(dst)
    # Recurse for each child
    new_indent = indent + ("    " if is_last else "│   ")
    for i, child_id in enumerate(children):
        is_last_child = i == len(children) - 1
        print_graph(graph, child_id, new_indent, is_last_child)


def layered_hierarchy_pos(G, root, total_width=1.0, vert_gap=0.2):
    """NOTE: AI generated function"""
    from collections import defaultdict, deque

    levels = defaultdict(list)
    visited = set()
    queue = deque([(root, 0)])
    max_level = 0

    while queue:
        node, level = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        levels[level].append(node)
        max_level = max(max_level, level)
        for child in G.successors(node):
            queue.append((child, level + 1))

    pos = {}
    for level in range(max_level + 1):
        nodes = levels[level]
        n = len(nodes)
        gap = total_width / (n + 1)
        for i, node in enumerate(nodes):
            x = (i + 1) * gap
            y = -level * vert_gap
            pos[node] = (x, y)

    return pos
