from collections import defaultdict, deque

import networkx as nx
import pyformlang
import pyformlang.cfg
import numpy as np
from pyformlang.cfg import CFG
from pyformlang.rsa import Box, RecursiveAutomaton
from pyformlang.cfg import Production, Terminal, Variable
from pyformlang.finite_automaton import State, Symbol
from scipy.sparse import csr_array, eye_array, kron

from project.automata_utils import graph_to_nfa, regex_to_dfa
from project.matrix_utils import AdjacencyMatrixFA


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    nullable_vars = cfg.get_nullable_symbols()
    cnf_cfg = cfg.to_normal_form()

    new_productions = set(cnf_cfg.productions)
    for var in nullable_vars:
        new_productions.add(Production(var, []))

    result_cfg = pyformlang.cfg.CFG(
        start_symbol=cnf_cfg.start_symbol,
        productions=new_productions,
    )

    return result_cfg.remove_useless_symbols()


def _parse_cfg_rules(cfg):
    cfg = cfg_to_weak_normal_form(cfg)

    term_rules = defaultdict(set)
    eps_rules = set()
    bin_rules = []

    for p in cfg.productions:
        if len(p.body) == 0:
            eps_rules.add(p.head)
        elif len(p.body) == 1 and isinstance(p.body[0], Terminal):
            term_rules[p.body[0].value].add(p.head)
        elif len(p.body) == 2:
            bin_rules.append((p.head, p.body[0], p.body[1]))

    return cfg, term_rules, eps_rules, bin_rules


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg, term_rules, eps_rules, bin_rules = _parse_cfg_rules(cfg)

    left_rules = defaultdict(set)
    right_rules = defaultdict(set)
    for head, left, right in bin_rules:
        left_rules[left].add((head, right))
        right_rules[right].add((head, left))

    nfa = graph_to_nfa(graph)

    derived = set()
    queue = deque()
    ending_at = defaultdict(set)
    starting_at = defaultdict(set)

    def add_triple(u: State, var: Variable, v: State) -> None:
        if (u, var, v) not in derived:
            derived.add((u, var, v))
            queue.append((u, var, v))
            ending_at[(v, var)].add(u)
            starting_at[(u, var)].add(v)

    for u, sym, v in nfa:
        for var in term_rules.get(sym.value, ()):
            add_triple(u, var, v)

    if eps_rules:
        all_nodes = [State(node) for node in graph.nodes()]
        for var in eps_rules:
            for u in all_nodes:
                add_triple(u, var, u)

    while queue:
        u, var, v = queue.popleft()

        for head, right in left_rules.get(var, ()):
            for w in list(starting_at.get((v, right), ())):
                add_triple(u, head, w)

        for head, left in right_rules.get(var, ()):
            for k in list(ending_at.get((u, left), ())):
                add_triple(k, head, v)

    starts = set(graph.nodes()) if not start_nodes else set(start_nodes)
    finals = set(graph.nodes()) if not final_nodes else set(final_nodes)
    start_sym = cfg.start_symbol

    return {
        (u.value, v.value)
        for u, var, v in derived
        if var == start_sym and u.value in starts and v.value in finals
    }


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg, term_rules, eps_rules, bin_rules = _parse_cfg_rules(cfg)
    nfa = graph_to_nfa(graph)
    graph_fa = AdjacencyMatrixFA(nfa)
    n = graph_fa.states_count

    if n == 0:
        return set()

    matrices = {var: csr_array((n, n), dtype=bool) for var in cfg.variables}

    for label, heads in term_rules.items():
        transitions = graph_fa.matrices.get(Symbol(label))
        if transitions is not None:
            for head in heads:
                matrices[head] = (matrices[head] + transitions).astype(bool)

    if eps_rules:
        identity = eye_array(n, dtype=bool, format="csr")
        for var in eps_rules:
            matrices[var] = matrices[var] + identity
            matrices[var] = matrices[var].astype(bool)

    changed = True
    while changed:
        changed = False
        for head, left, right in bin_rules:
            new_paths = matrices[left] @ matrices[right]

            old_nnz = matrices[head].nnz

            matrices[head] = matrices[head] + new_paths
            matrices[head] = matrices[head].astype(bool)

            if matrices[head].nnz > old_nnz:
                changed = True

    starts = set(graph.nodes()) if not start_nodes else set(start_nodes)
    finals = set(graph.nodes()) if not final_nodes else set(final_nodes)
    start_sym = cfg.start_symbol

    result = set()
    if start_sym in matrices:
        start_matrix = matrices[start_sym].tocoo()
        idx_to_node = {idx: state.value for state, idx in graph_fa.states.items()}

        for u_idx, v_idx in zip(start_matrix.row, start_matrix.col):
            u = idx_to_node[u_idx]
            v = idx_to_node[v_idx]
            if u in starts and v in finals:
                result.add((u, v))

    return result


def _build_rsm(by_head: dict[str, list[str]], initial_label: str) -> RecursiveAutomaton:
    boxes, labels = set(), set()
    for head, bodies in by_head.items():
        combined = " | ".join(f"({b})" for b in bodies)
        dfa = regex_to_dfa(combined)
        boxes.add(Box(dfa, Symbol(head)))
        labels.add(Symbol(head))

    return RecursiveAutomaton(
        labels=labels,
        initial_label=Symbol(initial_label),
        boxes=boxes,
    )


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    wnf = cfg_to_weak_normal_form(cfg)

    by_head: dict[str, list[str]] = defaultdict(list)
    for prod in wnf.productions:
        body = " ".join(sym.value for sym in prod.body) if prod.body else "epsilon"
        by_head[prod.head.value].append(body)

    return _build_rsm(by_head, wnf.start_symbol.value)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    by_head: dict[str, list[str]] = defaultdict(list)
    for line in ebnf.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        head, body = line.split("->", 1)
        by_head[head.strip()].append(body.strip())

    return _build_rsm(by_head, next(iter(by_head)))


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph))
    n = graph_fa.states_count
    if n == 0:
        return set()

    relations = {label: csr_array((n, n), dtype=bool) for label in rsm.labels}
    box_fas = {label: AdjacencyMatrixFA(box.dfa) for label, box in rsm.boxes.items()}

    base = {}
    for label, box_fa in box_fas.items():
        m = box_fa.states_count
        base[label] = csr_array((n * m, n * m), dtype=bool)
        for symbol, box_mat in box_fa.matrices.items():
            if symbol not in relations and symbol in graph_fa.matrices:
                base[label] = (
                    base[label] + kron(graph_fa.matrices[symbol], box_mat, format="csr")
                ).astype(bool)

    changed = True
    while changed:
        changed = False
        for label, box_fa in box_fas.items():
            m = box_fa.states_count
            N = n * m

            adjacency = base[label]
            for symbol, box_mat in box_fa.matrices.items():
                if symbol in relations:
                    adjacency = (
                        adjacency + kron(relations[symbol], box_mat, format="csr")
                    ).astype(bool)

            new_facts = csr_array((n, n), dtype=bool)
            for s in box_fa.start_states:
                front = csr_array(
                    ([True] * n, (range(n), np.arange(n) * m + s)),
                    shape=(n, N),
                    dtype=bool,
                )
                visited = front
                while front.nnz:
                    new_reach = (front @ adjacency).astype(bool)
                    front = (new_reach - new_reach.multiply(visited)).astype(bool)
                    visited = (visited + front).astype(bool)
                for f in box_fa.final_states:
                    new_facts = (new_facts + visited[:, f::m]).astype(bool)

            old_nnz = relations[label].nnz
            relations[label] = (relations[label] + new_facts).astype(bool)
            if relations[label].nnz > old_nnz:
                changed = True

    start_symbol = rsm.initial_label
    if start_symbol not in relations:
        return set()

    idx_to_node = {idx: state.value for state, idx in graph_fa.states.items()}
    starts = set(graph.nodes()) if not start_nodes else set(start_nodes)
    finals = set(graph.nodes()) if not final_nodes else set(final_nodes)

    rows, cols = relations[start_symbol].nonzero()
    return {
        (u, v)
        for u_i, v_i in zip(rows.tolist(), cols.tolist())
        if (u := idx_to_node[u_i]) in starts and (v := idx_to_node[v_i]) in finals
    }
