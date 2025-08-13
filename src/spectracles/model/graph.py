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


def layered_hierarchy_pos(G: DiGraph, root, total_width: float = 1.0, vert_gap: float = 0.8):
    """
    NOTE: AI generated function.
    Layered layout rooted at `root`.
    - Levels by BFS distance from root.
    - Within each level, order nodes by the barycenter of their parents'
      x-positions (one top-down pass), which reduces crossings.
    """
    from collections import defaultdict, deque

    # --- levels via BFS ---
    level_nodes = defaultdict(list)
    level = {root: 0}
    q = deque([root])
    seen = {root}
    while q:
        u = q.popleft()
        l = level[u]
        level_nodes[l].append(u)
        for v in G.successors(u):
            if v not in seen:
                seen.add(v)
                level[v] = l + 1
                q.append(v)

    max_level = max(level_nodes) if level_nodes else 0

    # --- initial even spacing per level ---
    pos = {}
    for l in range(max_level + 1):
        nodes = level_nodes[l]
        n = max(1, len(nodes))
        gap = total_width / (n + 1)
        for i, u in enumerate(nodes):
            pos[u] = ((i + 1) * gap, -l * vert_gap)

    # --- one barycenter sweep (top -> down) ---
    for l in range(1, max_level + 1):
        nodes = level_nodes[l]

        def bary(u):
            preds = list(G.predecessors(u))
            if not preds:
                return pos[u][0]
            return sum(pos[p][0] for p in preds) / len(preds)

        ordered = sorted(nodes, key=bary)
        level_nodes[l] = ordered  # keep for consistency
        n = len(ordered)
        gap = total_width / (n + 1)
        for i, u in enumerate(ordered):
            pos[u] = ((i + 1) * gap, -l * vert_gap)

    return pos
