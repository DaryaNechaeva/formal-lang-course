import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import Symbol

from project.automata_utils import graph_to_nfa, regex_to_dfa


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


def test_regex_to_dfa_is_deterministic():
    """Check that the result is a deterministic automaton."""
    dfa = regex_to_dfa("a b* | c")
    assert dfa.is_deterministic()


def test_regex_to_dfa_is_minimal():
    """Check that minimizing the result does not change the number of states."""
    dfa = regex_to_dfa("(a | b)* a b b")
    assert len(dfa.minimize().states) == len(dfa.states)


def test_regex_to_dfa_accepts_and_rejects():
    """Check that the automaton accepts words of the language and rejects others."""
    dfa = regex_to_dfa("a b* | c")
    assert dfa.accepts(["a"])
    assert dfa.accepts(["a", "b", "b"])
    assert dfa.accepts(["c"])
    assert not dfa.accepts([])
    assert not dfa.accepts(["b"])
    assert not dfa.accepts(["c", "b"])


def test_regex_to_dfa_equivalent_regexes():
    """Check that equivalent regexes give equivalent automata."""
    assert regex_to_dfa("a a*").is_equivalent_to(regex_to_dfa("a* a"))


def test_graph_to_nfa_states_and_symbols(two_cycles_graph: MultiDiGraph):
    """Check the number of states and the set of symbols."""
    nfa = graph_to_nfa(two_cycles_graph, {0}, {0})
    assert len(nfa.states) == two_cycles_graph.number_of_nodes()
    assert nfa.symbols == {Symbol("a"), Symbol("b")}


def test_graph_to_nfa_specified_start_and_final(two_cycles_graph: MultiDiGraph):
    """Check that specified start and final states are used."""
    nfa = graph_to_nfa(two_cycles_graph, {0}, {1, 2})
    assert {state.value for state in nfa.start_states} == {0}
    assert {state.value for state in nfa.final_states} == {1, 2}


def test_graph_to_nfa_not_specified_start_and_final(two_cycles_graph: MultiDiGraph):
    """Check that all nodes are start and final if the sets are not given."""
    nodes = set(two_cycles_graph.nodes)
    nfa = graph_to_nfa(two_cycles_graph)
    assert {state.value for state in nfa.start_states} == nodes
    assert {state.value for state in nfa.final_states} == nodes


def test_graph_to_nfa_empty_sets_mean_all_nodes(two_cycles_graph: MultiDiGraph):
    """Check that empty sets are treated as 'all nodes'."""
    nodes = set(two_cycles_graph.nodes)
    nfa = graph_to_nfa(two_cycles_graph, set(), set())
    assert {state.value for state in nfa.start_states} == nodes
    assert {state.value for state in nfa.final_states} == nodes


def test_graph_to_nfa_accepts_and_rejects(two_cycles_graph: MultiDiGraph):
    """Check that the automaton accepts exactly the paths from start to final."""
    nfa = graph_to_nfa(two_cycles_graph, {0}, {0})
    assert nfa.accepts(["a", "a", "a"])
    assert nfa.accepts(["b", "b"])
    assert nfa.accepts(["a", "a", "a", "b", "b"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b"])
    assert not nfa.accepts(["a", "b"])


def test_graph_to_nfa_different_start_and_final(two_cycles_graph: MultiDiGraph):
    """Check paths between different start and final states."""
    nfa = graph_to_nfa(two_cycles_graph, {0}, {2})
    assert nfa.accepts(["a", "a"])
    assert nfa.accepts(["b", "b", "a", "a"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b", "b"])


def test_graph_to_nfa_empty_graph():
    """Check that an empty graph gives an empty automaton."""
    nfa = graph_to_nfa(MultiDiGraph())
    assert nfa.is_empty()
