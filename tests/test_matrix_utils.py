import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import Symbol

from project.automata_utils import graph_to_nfa, regex_to_dfa
from project.matrix_utils import AdjacencyMatrixFA, intersect_automata, tensor_based_rpq


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


def test_adjacency_matrix_fa_states_count():
    """Check that the number of states matches the source automaton."""
    dfa = regex_to_dfa("a b*")
    fa = AdjacencyMatrixFA(dfa)
    assert fa.states_count == len(dfa.states)


def test_adjacency_matrix_fa_start_and_final_counts():
    """Check that start/final state counts match the source automaton."""
    dfa = regex_to_dfa("a b*")
    fa = AdjacencyMatrixFA(dfa)
    assert len(fa.start_states) == len(dfa.start_states)
    assert len(fa.final_states) == len(dfa.final_states)


def test_adjacency_matrix_fa_symbols():
    """Check that matrices are built exactly for the automaton's symbols."""
    dfa = regex_to_dfa("a b* | c")
    fa = AdjacencyMatrixFA(dfa)
    assert set(fa.matrices.keys()) == {Symbol("a"), Symbol("b"), Symbol("c")}


def test_accepts_word_in_language():
    """Check that words from the language are accepted."""
    fa = AdjacencyMatrixFA(regex_to_dfa("a b* | c"))
    assert fa.accepts([Symbol("a")])
    assert fa.accepts([Symbol("a"), Symbol("b"), Symbol("b")])
    assert fa.accepts([Symbol("c")])


def test_accepts_word_not_in_language():
    """Check that words outside the language are rejected."""
    fa = AdjacencyMatrixFA(regex_to_dfa("a b* | c"))
    assert not fa.accepts([Symbol("b")])
    assert not fa.accepts([Symbol("c"), Symbol("b")])
    assert not fa.accepts([])


def test_accepts_empty_word_when_start_is_final():
    """Check that the empty word is accepted when a start state is also final."""
    fa = AdjacencyMatrixFA(regex_to_dfa("a*"))
    assert fa.accepts([])


def test_accepts_on_graph_automaton(two_cycles_graph: MultiDiGraph):
    """Check accepts() on an automaton built from a graph."""
    fa = AdjacencyMatrixFA(graph_to_nfa(two_cycles_graph, {0}, {0}))
    assert fa.accepts([Symbol("a"), Symbol("a"), Symbol("a")])
    assert fa.accepts([Symbol("b"), Symbol("b")])
    assert not fa.accepts([Symbol("a")])
    assert not fa.accepts([Symbol("a"), Symbol("b")])


def test_is_empty_false_for_nonempty_language():
    """Check that a normal automaton is not empty."""
    fa = AdjacencyMatrixFA(regex_to_dfa("a b*"))
    assert not fa.is_empty()


def test_is_empty_true_when_final_unreachable():
    """Check that the language is empty when the final state cannot be reached."""
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    fa = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {2}))
    assert fa.is_empty()


def test_is_empty_true_without_start_states():
    """Check that an automaton with no start states is empty."""
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    fa = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {1}))
    fa.start_states = set()
    assert fa.is_empty()


def test_is_empty_true_for_empty_graph():
    """Check that an automaton built from an empty graph is empty."""
    fa = AdjacencyMatrixFA(graph_to_nfa(MultiDiGraph()))
    assert fa.is_empty()


def test_intersect_automata_equivalent_to_pyformlang():
    """Check that intersect_automata matches pyformlang's own intersection."""
    dfa1 = regex_to_dfa("a b* c")
    dfa2 = regex_to_dfa("a b c*")
    reference = dfa1.get_intersection(dfa2)

    result = intersect_automata(AdjacencyMatrixFA(dfa1), AdjacencyMatrixFA(dfa2))

    words = [
        [],
        [Symbol("a")],
        [Symbol("a"), Symbol("b"), Symbol("c")],
        [Symbol("a"), Symbol("b"), Symbol("b"), Symbol("c")],
        [Symbol("a"), Symbol("c")],
    ]
    for word in words:
        assert result.accepts(word) == reference.accepts(word)


def test_intersect_automata_disjoint_languages_is_empty():
    """Check that intersecting disjoint languages gives an empty automaton."""
    fa1 = AdjacencyMatrixFA(regex_to_dfa("a a*"))
    fa2 = AdjacencyMatrixFA(regex_to_dfa("b b*"))
    result = intersect_automata(fa1, fa2)
    assert result.is_empty()


def test_tensor_based_rpq_simple_chain():
    """Check RPQ on a simple chain graph."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="b")
    result = tensor_based_rpq("a b*", graph, {1}, {2, 3, 4})
    assert result == {(1, 2), (1, 3), (1, 4)}


def test_tensor_based_rpq_no_matching_path():
    """Check that RPQ returns an empty set when no path matches."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    result = tensor_based_rpq("b", graph, {1}, {2})
    assert result == set()


def test_tensor_based_rpq_two_cycles(two_cycles_graph: MultiDiGraph):
    """Check RPQ on the two-cycles graph."""
    result = tensor_based_rpq("a a a | b b", two_cycles_graph, {0}, {0})
    assert result == {(0, 0)}


def test_tensor_based_rpq_empty_word_start_equals_final():
    """Check that a regex accepting the empty word yields (v, v) pairs."""
    graph = MultiDiGraph()
    graph.add_edge(1, 2, label="a")
    result = tensor_based_rpq("a*", graph, {1}, {1, 2})
    assert (1, 1) in result
    assert (1, 2) in result
