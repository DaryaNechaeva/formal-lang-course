import numpy as np
from networkx import MultiDiGraph

from project.automata_utils import graph_to_nfa, regex_to_dfa
from project.matrix_utils import AdjacencyMatrixFA, intersect_automata
from scipy.sparse import coo_array, csr_array, eye_array, kron


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


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    n_r = regex_fa.states_count
    n_g = graph_fa.states_count
    start_list = sorted(graph_fa.start_states)
    k = len(start_list)

    if k == 0 or n_r == 0 or n_g == 0:
        return set()

    common_symbols = graph_fa.matrices.keys() & regex_fa.matrices.keys()

    identity_k = eye_array(k, dtype=bool, format="csr")
    regex_block_t = {
        symbol: kron(
            identity_k, regex_fa.matrices[symbol].transpose(), format="csr"
        ).astype(bool)
        for symbol in common_symbols
    }

    rows, cols = [], []
    for i, g_start in enumerate(start_list):
        for r in regex_fa.start_states:
            rows.append(i * n_r + r)
            cols.append(g_start)
    front = coo_array(
        ([True] * len(rows), (rows, cols)), shape=(k * n_r, n_g), dtype=bool
    ).tocsr()
    visited = front.copy()

    while front.nnz:
        next_front = csr_array((k * n_r, n_g), dtype=bool)
        for symbol in common_symbols:
            step = regex_block_t[symbol] @ (front @ graph_fa.matrices[symbol])
            next_front = (next_front + step).astype(bool)
        front = (next_front - next_front.multiply(visited)).astype(bool)
        visited = (visited + front).astype(bool)

    visited_coo = visited.tocoo()
    row_regex_idx = visited_coo.row % n_r
    row_start_idx = visited_coo.row // n_r
    mask = np.isin(row_regex_idx, list(regex_fa.final_states)) & np.isin(
        visited_coo.col, list(graph_fa.final_states)
    )

    idx_to_graph_state = {idx: state for state, idx in graph_fa.states.items()}

    result: set[tuple[int, int]] = set()
    for start_idx, graph_idx in zip(row_start_idx[mask], visited_coo.col[mask]):
        start_state = idx_to_graph_state[start_list[start_idx]]
        final_state = idx_to_graph_state[graph_idx]
        result.add((start_state.value, final_state.value))
    return result
