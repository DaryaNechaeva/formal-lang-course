import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import Symbol

from project.automata_utils import graph_to_nfa, regex_to_dfa
from project.rpq_utils import ms_bfs_based_rpq, tensor_based_rpq


@pytest.fixture
def two_cycles_graph() -> MultiDiGraph:
    """Two cycles through node 0: 0 -a-> 1 -a-> 2 -a-> 0 and 0 -b-> 3 -b-> 0."""
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 0, label="a")
    graph.add_edge(0, 3, label="b")
    graph.add_edge(3, 0, label="b")
    return graph


def test_ms_bfs_simple_chain():
    """Check MS-BFS on a simple chain graph."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="b")
    result = ms_bfs_based_rpq("a b*", graph, {1}, {2, 3, 4})
    assert result == {(1, 2), (1, 3), (1, 4)}


def test_ms_bfs_no_matching_path():
    """Check that MS-BFS returns an empty set when no path matches."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    result = ms_bfs_based_rpq("b", graph, {1}, {2})
    assert result == set()


def test_ms_bfs_two_cycles(two_cycles_graph: MultiDiGraph):
    """Check MS-BFS on the two-cycles graph."""
    result = ms_bfs_based_rpq("a a a | b b", two_cycles_graph, {0}, {0})
    assert result == {(0, 0)}


def test_ms_bfs_empty_word_start_equals_final():
    """Check that a regex accepting the empty word yields (v, v) pairs."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    result = ms_bfs_based_rpq("a*", graph, {1}, {1, 2})
    assert (1, 1) in result
    assert (1, 2) in result


def test_ms_bfs_multiple_starts():
    """Check that MS-BFS handles multiple start nodes simultaneously."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="a")
    graph.add_edge(4, 5, label="a")
    result = ms_bfs_based_rpq("a", graph, {1, 4}, {2, 5})
    assert result == {(1, 2), (4, 5)}


def test_ms_bfs_multiple_starts_shared_targets():
    """Multiple start nodes may reach the same final node."""
    graph = MultiDiGraph()
    graph.add_edge(1, 3, label="a")
    graph.add_edge(2, 3, label="a")
    result = ms_bfs_based_rpq("a", graph, {1, 2}, {3})
    assert result == {(1, 3), (2, 3)}


def test_ms_bfs_multiple_starts_disjoint():
    """Two start nodes reach disjoint target sets."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(3, 4, label="b")
    result = ms_bfs_based_rpq("a | b", graph, {1, 3}, {2, 4})
    assert result == {(1, 2), (3, 4)}


def test_ms_bfs_empty_start_nodes():
    """Empty start_nodes means all graph nodes are start states."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    result = ms_bfs_based_rpq("a", graph, set(), {2})
    assert result == {(1, 2)}


def test_ms_bfs_empty_final_nodes():
    """Empty final_nodes means all graph nodes are final states."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    result = ms_bfs_based_rpq("a", graph, {1}, set())
    assert result == {(1, 2)}


def test_ms_bfs_empty_graph():
    """MS-BFS on an empty graph returns an empty set."""
    result = ms_bfs_based_rpq("a", MultiDiGraph(), {1}, {2})
    assert result == set()


def test_ms_bfs_no_start_in_graph():
    """Start nodes not present in the graph produce no results."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    result = ms_bfs_based_rpq("a", graph, {99}, {2})
    assert result == set()


def test_ms_bfs_matches_tensor_based_rpq_simple():
    """MS-BFS and tensor-based RPQ give the same answer on a simple graph."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="a")
    regex = "a b* a"
    starts = {1}
    finals = {2, 3, 4}
    assert ms_bfs_based_rpq(regex, graph, starts, finals) == tensor_based_rpq(
        regex, graph, starts, finals
    )


def test_ms_bfs_matches_tensor_based_rpq_two_cycles(
    two_cycles_graph: MultiDiGraph,
):
    """MS-BFS and tensor-based RPQ agree on the two-cycles graph."""
    for regex in ["a", "a a a", "b b", "a a a | b b", "a*"]:
        assert ms_bfs_based_rpq(
            regex, two_cycles_graph, {0}, {0, 1, 2, 3}
        ) == tensor_based_rpq(regex, two_cycles_graph, {0}, {0, 1, 2, 3})


def test_ms_bfs_matches_tensor_based_rpq_multiple_starts():
    """MS-BFS and tensor-based RPQ agree with multiple start nodes."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="a")
    graph.add_edge(4, 5, label="a")
    regex = "a b* a*"
    starts = {1, 3}
    finals = {2, 4, 5}
    assert ms_bfs_based_rpq(regex, graph, starts, finals) == tensor_based_rpq(
        regex, graph, starts, finals
    )
