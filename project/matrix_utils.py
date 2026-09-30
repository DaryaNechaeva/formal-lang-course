from collections.abc import Iterable
from typing import Self
import numpy as np

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from scipy.sparse import coo_array, csr_array, eye_array, kron

from project.automata_utils import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton | None = None):
        self.states: dict[State, int] = {}
        self.states_count: int = 0
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()
        self.matrices: dict[Symbol, csr_array] = {}

        if automaton is not None:
            self._init_from_automaton(automaton)

    def _init_from_automaton(self, nfa: NondeterministicFiniteAutomaton) -> None:
        self.states = {state: idx for idx, state in enumerate(nfa.states)}
        self.states_count = len(self.states)
        self.start_states = {self.states[s] for s in nfa.start_states}
        self.final_states = {self.states[s] for s in nfa.final_states}

        coords: dict[Symbol, tuple[list[int], list[int]]] = {}
        for src, transitions in nfa.to_dict().items():
            for symbol, dst in transitions.items():
                targets = [dst] if isinstance(dst, State) else dst
                rows, cols = coords.setdefault(symbol, ([], []))
                for target in targets:
                    rows.append(self.states[src])
                    cols.append(self.states[target])

        n = self.states_count
        for symbol, (rows, cols) in coords.items():
            data = [True] * len(rows)
            self.matrices[symbol] = coo_array(
                (data, (rows, cols)), shape=(n, n), dtype=bool
            ).tocsr()

    @classmethod
    def from_matrices(
        cls,
        matrices: dict[Symbol, csr_array],
        start_states: set[int],
        final_states: set[int],
        states_count: int,
        states: dict[State, int] | None = None,
    ) -> Self:
        fa = cls()
        fa.matrices = {s: csr_array(m, dtype=bool) for s, m in matrices.items()}
        fa.start_states = set(start_states)
        fa.final_states = set(final_states)
        fa.states_count = states_count
        fa.states = states if states is not None else {}
        return fa

    def accepts(self, word: Iterable[Symbol]) -> bool:
        n = self.states_count
        if n == 0 or not self.start_states:
            return False

        start_list = list(self.start_states)
        current = coo_array(
            ([True] * len(start_list), ([0] * len(start_list), start_list)),
            shape=(1, n),
            dtype=bool,
        ).tocsr()

        for symbol in word:
            key = symbol if isinstance(symbol, Symbol) else Symbol(symbol)
            if key not in self.matrices:
                return False
            current = (current @ self.matrices[key]).astype(bool)
            if current.nnz == 0:
                return False

        reached = set(current.nonzero()[1])
        return bool(reached & self.final_states)

    def transitive_closure(self) -> csr_array:
        closure = eye_array(self.states_count, dtype=bool, format="csr")
        for matrix in self.matrices.values():
            closure = (closure + matrix).astype(bool)

        while True:
            prev_nnz = closure.nnz
            closure = (closure @ closure).astype(bool)
            if closure.nnz == prev_nnz:
                return closure

    def is_empty(self) -> bool:
        if not self.start_states or not self.final_states:
            return True
        closure = self.transitive_closure()
        return closure[list(self.start_states)][:, list(self.final_states)].nnz == 0


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    n1, n2 = automaton1.states_count, automaton2.states_count
    common_symbols = automaton1.matrices.keys() & automaton2.matrices.keys()
    matrices = {
        symbol: kron(
            automaton1.matrices[symbol], automaton2.matrices[symbol], format="csr"
        ).astype(bool)
        for symbol in common_symbols
    }
    start_states = {
        i * n2 + j for i in automaton1.start_states for j in automaton2.start_states
    }
    final_states = {
        i * n2 + j for i in automaton1.final_states for j in automaton2.final_states
    }
    return AdjacencyMatrixFA.from_matrices(
        matrices, start_states, final_states, n1 * n2
    )


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
