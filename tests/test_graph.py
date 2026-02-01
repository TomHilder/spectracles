"""test_graph.py - tests for the spectracles.model.graph module."""

import io
import sys

import pytest
from networkx import DiGraph
from spectracles.model.graph import (
    DEFAULT_NX_KWDS,
    layered_hierarchy_pos,
    print_graph,
    temporarily_disable_tex,
)
import matplotlib as mpl


class TestDefaultNxKwds:
    def test_defaults_exist(self):
        # Test that default kwargs are defined
        assert "node_size" in DEFAULT_NX_KWDS
        assert "node_color" in DEFAULT_NX_KWDS
        assert "font_size" in DEFAULT_NX_KWDS


class TestTemporarilyDisableTex:
    def test_context_manager(self):
        # Save original value
        original = mpl.rcParams["text.usetex"]

        # Use context manager
        with temporarily_disable_tex():
            assert mpl.rcParams["text.usetex"] is False

        # Check restored
        assert mpl.rcParams["text.usetex"] == original

    def test_restores_on_exception(self):
        original = mpl.rcParams["text.usetex"]

        try:
            with temporarily_disable_tex():
                raise ValueError("test")
        except ValueError:
            pass

        # Should still be restored
        assert mpl.rcParams["text.usetex"] == original


class TestPrintGraph:
    def test_simple_graph(self):
        # Create a simple graph
        G = DiGraph()
        G.add_node(0, name=None, type="RootModel", is_param=False)
        G.add_node(1, name="param", type="Parameter", is_param=True)
        G.add_edge(0, 1)

        # Should not raise
        print_graph(G, 0)

    def test_nested_graph(self):
        # Create a nested graph
        G = DiGraph()
        G.add_node(0, name=None, type="Model", is_param=False)
        G.add_node(1, name="inner", type="SubModel", is_param=False)
        G.add_node(2, name="param", type="Parameter", is_param=True)
        G.add_edge(0, 1)
        G.add_edge(1, 2)

        # Should not raise
        print_graph(G, 0)

    def test_multiple_children(self):
        # Create a graph with multiple children
        G = DiGraph()
        G.add_node(0, name=None, type="Model", is_param=False)
        G.add_node(1, name="a", type="Parameter", is_param=True)
        G.add_node(2, name="b", type="Parameter", is_param=True)
        G.add_node(3, name="c", type="Parameter", is_param=True)
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(0, 3)

        # Should not raise
        print_graph(G, 0)

    def test_with_custom_indent(self):
        G = DiGraph()
        G.add_node(0, name=None, type="Model", is_param=False)

        # Should not raise with custom indent
        print_graph(G, 0, indent="  ", is_last=False)


class TestLayeredHierarchyPos:
    def test_single_node(self):
        G = DiGraph()
        G.add_node(0)

        pos = layered_hierarchy_pos(G, 0)

        assert 0 in pos
        assert len(pos[0]) == 2  # x, y coordinates

    def test_simple_tree(self):
        G = DiGraph()
        G.add_node(0)
        G.add_node(1)
        G.add_node(2)
        G.add_edge(0, 1)
        G.add_edge(0, 2)

        pos = layered_hierarchy_pos(G, 0)

        # All nodes should have positions
        assert len(pos) == 3

        # Root should be at top (y=0)
        assert pos[0][1] == 0

        # Children should be below (y < 0)
        assert pos[1][1] < 0
        assert pos[2][1] < 0

    def test_deep_tree(self):
        G = DiGraph()
        G.add_node(0)
        G.add_node(1)
        G.add_node(2)
        G.add_node(3)
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)

        pos = layered_hierarchy_pos(G, 0)

        # Each level should be progressively lower
        assert pos[0][1] > pos[1][1] > pos[2][1] > pos[3][1]

    def test_custom_dimensions(self):
        G = DiGraph()
        G.add_node(0)
        G.add_node(1)
        G.add_edge(0, 1)

        pos = layered_hierarchy_pos(G, 0, total_width=2.0, vert_gap=0.5)

        # Root y should still be 0
        assert pos[0][1] == 0

        # Child should be at -0.5 (one level down with gap 0.5)
        assert pos[1][1] == -0.5

    def test_barycenter_ordering(self):
        # Create a tree where barycenter ordering matters
        G = DiGraph()
        # Root
        G.add_node(0)
        # Two children
        G.add_node(1)
        G.add_node(2)
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        # Grandchildren from first child
        G.add_node(3)
        G.add_node(4)
        G.add_edge(1, 3)
        G.add_edge(1, 4)

        pos = layered_hierarchy_pos(G, 0)

        # All 5 nodes should have positions
        assert len(pos) == 5

    def test_empty_graph_with_root(self):
        G = DiGraph()
        G.add_node(0)

        pos = layered_hierarchy_pos(G, 0)

        assert 0 in pos
