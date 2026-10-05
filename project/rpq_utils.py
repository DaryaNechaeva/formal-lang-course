import numpy as np
from networkx import MultiDiGraph

from project.automata_utils import graph_to_nfa, regex_to_dfa
from project.matrix_utils import AdjacencyMatrixFA, intersect_automata


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    intersection = intersect_automata(graph_fa, regex_fa)
    closure = intersection.transitive_closure().tocoo()

    n2 = regex_fa.states_count
    idx_to_graph_state = {idx: state for state, idx in graph_fa.states.items()}

    mask = np.isin(closure.row, list(intersection.start_states)) & np.isin(
        closure.col, list(intersection.final_states)
    )

    result: set[tuple[int, int]] = set()
    for row, col in zip(closure.row[mask].tolist(), closure.col[mask].tolist()):
        result.add(
            (
                idx_to_graph_state[row // n2].value,
                idx_to_graph_state[col // n2].value,
            )
        )
    return result
