from collections import defaultdict, deque

import networkx as nx
import pyformlang
import pyformlang.cfg
from pyformlang.cfg import Production, Terminal, Variable
from pyformlang.finite_automaton import State

from scipy.sparse import csr_array, eye_array, coo_array

from project.automata_utils import graph_to_nfa


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

    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return set()

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    matrices = {var: csr_array((n, n), dtype=bool) for var in cfg.variables}

    term_coords = defaultdict(lambda: ([], []))
    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label in term_rules:
            for var in term_rules[label]:
                term_coords[var][0].append(node_to_idx[u])
                term_coords[var][1].append(node_to_idx[v])

    for var, (rows, cols) in term_coords.items():
        if rows:
            data = [True] * len(rows)
            matrices[var] = coo_array(
                (data, (rows, cols)), shape=(n, n), dtype=bool
            ).tocsr()

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
        for u_idx, v_idx in zip(start_matrix.row, start_matrix.col):
            u = nodes[u_idx]
            v = nodes[v_idx]
            if u in starts and v in finals:
                result.add((u, v))

    return result
