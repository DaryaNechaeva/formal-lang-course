from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: set[int] | None = None,
    final_states: set[int] | None = None,
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for source, target, label in graph.edges(data="label"):
        if label is not None:
            nfa.add_transition(State(source), Symbol(str(label)), State(target))

    all_nodes = set(graph.nodes())
    for state in start_states or all_nodes:
        nfa.add_start_state(State(state))
    for state in final_states or all_nodes:
        nfa.add_final_state(State(state))

    return nfa
