import pathlib

import networkx as nx
import pydot
import pytest

from project.graph_utils import (
    GraphInfo,
    create_two_cycles_graph,
    get_graph_info,
)


def test_get_graph_info_returns_correct_type():
    """Check that the function returns an instance of GraphInfo."""
    info = get_graph_info("bzip")
    assert isinstance(info, GraphInfo)


def test_get_graph_info_fields_are_correct_types():
    """Check that GraphInfo fields have correct types."""
    info = get_graph_info("bzip")
    assert isinstance(info.number_of_nodes, int)
    assert isinstance(info.number_of_edges, int)
    assert isinstance(info.labels, set)
    assert all(isinstance(label, str) for label in info.labels)


def test_get_graph_info_bzip():
    """Check that bzip graph has expected properties."""
    info = get_graph_info("bzip")
    assert info.number_of_nodes == 632
    assert info.number_of_edges == 556
    assert info.labels == {"a", "d"}


def test_get_graph_info_wc():
    """Check that wc graph has expected properties."""
    info = get_graph_info("wc")
    assert info.number_of_nodes == 332
    assert info.number_of_edges == 269
    assert info.labels == {"a", "d"}


def test_get_graph_info_unknown_raises_error():
    """Check that unknown graph name raises an error."""
    with pytest.raises(FileNotFoundError):
        get_graph_info("nonexistent_graph_xyz")


def test_create_two_cycles_graph_returns_multidigraph(tmp_path: pathlib.Path):
    """Check that the function returns a networkx MultiDiGraph."""
    output = tmp_path / "graph.dot"
    graph = create_two_cycles_graph(3, 2, ("a", "b"), output)
    assert isinstance(graph, nx.MultiDiGraph)


def test_create_two_cycles_graph_node_count(tmp_path: pathlib.Path):
    """Check that the created graph has correct number of nodes."""
    output = tmp_path / "graph.dot"
    n, m = 3, 2
    graph = create_two_cycles_graph(n, m, ("a", "b"), output)
    assert graph.number_of_nodes() == n + m + 1


def test_create_two_cycles_graph_edge_count(tmp_path: pathlib.Path):
    """Check that the created graph has correct number of edges."""
    output = tmp_path / "graph.dot"
    n, m = 3, 2
    graph = create_two_cycles_graph(n, m, ("a", "b"), output)
    assert graph.number_of_edges() == n + m + 2


def test_create_two_cycles_graph_creates_file(tmp_path: pathlib.Path):
    """Check that the DOT file is created."""
    output = tmp_path / "graph.dot"
    create_two_cycles_graph(3, 4, ("x", "y"), output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_create_two_cycles_graph_labels_in_file(tmp_path: pathlib.Path):
    """Check that the specified labels are present in the DOT file."""
    output = tmp_path / "graph.dot"
    create_two_cycles_graph(3, 4, ("first", "second"), output)

    dot_graphs = pydot.graph_from_dot_file(str(output))
    assert len(dot_graphs) == 1
    dot_graph = dot_graphs[0]

    edge_labels = {edge.get_label().strip('"') for edge in dot_graph.get_edges()}
    assert edge_labels == {"first", "second"}


def test_create_two_cycles_graph_with_pathlib_path(tmp_path: pathlib.Path):
    """Check that the function works with pathlib.Path."""
    output = tmp_path / "graph.dot"
    graph = create_two_cycles_graph(2, 3, ("a", "b"), output)
    assert output.exists()
    assert isinstance(graph, nx.MultiDiGraph)


def test_create_two_cycles_graph_with_str_path(tmp_path: pathlib.Path):
    """Check that the function works with str path."""
    output = str(tmp_path / "graph.dot")
    graph = create_two_cycles_graph(2, 3, ("a", "b"), output)
    assert pathlib.Path(output).exists()
    assert isinstance(graph, nx.MultiDiGraph)


def test_create_two_cycles_graph_round_trip(tmp_path: pathlib.Path):
    """Check that saved DOT file can be loaded back and matches original."""
    output = tmp_path / "graph.dot"
    n, m = 3, 2
    original = create_two_cycles_graph(n, m, ("x", "y"), output)

    (dot_graph,) = pydot.graph_from_dot_file(str(output))
    assert len(dot_graph.get_nodes()) == original.number_of_nodes()
    assert len(dot_graph.get_edges()) == original.number_of_edges()


@pytest.mark.parametrize("n,m", [(1, 1), (5, 3), (10, 7), (2, 8)])
def test_create_two_cycles_graph_various_sizes(tmp_path: pathlib.Path, n: int, m: int):
    """Test with various cycle sizes."""
    output = tmp_path / f"graph_{n}_{m}.dot"
    graph = create_two_cycles_graph(n, m, ("a", "b"), output)
    assert graph.number_of_nodes() == n + m + 1
    assert graph.number_of_edges() == n + m + 2
    assert output.exists()
