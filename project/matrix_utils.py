from typing import Self
from scipy.sparse import coo_array, csr_array
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)


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
