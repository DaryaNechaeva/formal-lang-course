import pytest
import networkx as nx
from pyformlang.cfg import CFG, Variable, Terminal, Epsilon

from project.cfg_utils import (
    cfg_to_weak_normal_form,
    cfg_to_rsm,
    ebnf_to_rsm,
    hellings_based_cfpq,
    matrix_based_cfpq,
    tensor_based_cfpq,
)


def _tensor_via_cfg(cfg, graph, start_nodes=None, final_nodes=None):
    return tensor_based_cfpq(cfg_to_rsm(cfg), graph, start_nodes, final_nodes)


CFPQ_ALGORITHMS = [hellings_based_cfpq, matrix_based_cfpq, _tensor_via_cfg]


def test_weak_normal_form_structure():
    """Check that all rules have a valid format in Weak Chomsky Normal Form."""
    cfg = CFG.from_text("""
        S -> A B C | epsilon
        A -> a
        B -> b
        C -> c
    """)
    wnf_cfg = cfg_to_weak_normal_form(cfg)

    assert isinstance(wnf_cfg, CFG)

    for prod in wnf_cfg.productions:
        body_len = len(prod.body)
        assert body_len <= 2

        if body_len == 1:
            assert isinstance(prod.body[0], Terminal)
        elif body_len == 2:
            assert isinstance(prod.body[0], Variable)
            assert isinstance(prod.body[1], Variable)


def test_weak_normal_form_preserves_epsilon():
    """Check that if the grammar generates epsilon, this property is preserved."""
    cfg = CFG.from_text("""
        S -> A B | epsilon
        A -> a
        B -> b
    """)
    wnf_cfg = cfg_to_weak_normal_form(cfg)

    assert wnf_cfg.contains([Epsilon()])


@pytest.mark.parametrize("algorithm", CFPQ_ALGORITHMS)
def test_cfpq_simple_path(algorithm):
    """Check a simple path without cycles."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    cfg = CFG.from_text("""
        S -> A B
        A -> a
        B -> b
    """)

    assert algorithm(cfg, graph) == {(0, 2)}


@pytest.mark.parametrize("algorithm", CFPQ_ALGORITHMS)
def test_cfpq_empty_start_final(algorithm):
    """Check that empty start_nodes and final_nodes mean all nodes."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")

    cfg = CFG.from_text("S -> a")

    assert algorithm(cfg, graph, start_nodes=set(), final_nodes=set()) == {(0, 1)}


@pytest.mark.parametrize("algorithm", CFPQ_ALGORITHMS)
def test_cfpq_specific_start_final(algorithm):
    """Check filtering by specified start and final nodes."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")

    cfg = CFG.from_text("S -> a")

    assert algorithm(cfg, graph, start_nodes={0}, final_nodes={1}) == {(0, 1)}


@pytest.mark.parametrize("algorithm", CFPQ_ALGORITHMS)
def test_cfpq_cycle(algorithm):
    """Check the algorithm on a graph with a cycle."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="a")

    cfg = CFG.from_text("""
        S -> A S
        S -> A
        A -> a
    """)

    result = algorithm(cfg, graph)

    assert (0, 1) in result
    assert (1, 0) in result
    assert (0, 0) in result
    assert (1, 1) in result


@pytest.mark.parametrize("algorithm", CFPQ_ALGORITHMS)
def test_cfpq_epsilon_rule(algorithm):
    """Check that epsilon rules produce zero-length path loops."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")

    cfg = CFG.from_text("S -> epsilon")

    assert algorithm(cfg, graph, start_nodes={0}, final_nodes={0}) == {(0, 0)}


@pytest.mark.parametrize("algorithm", CFPQ_ALGORITHMS)
def test_cfpq_empty_graph(algorithm):
    """Check that an empty graph returns an empty set, not an error."""
    graph = nx.DiGraph()
    cfg = CFG.from_text("S -> a")
    assert algorithm(cfg, graph) == set()


@pytest.mark.parametrize("algorithm", CFPQ_ALGORITHMS)
def test_cfpq_no_paths(algorithm):
    """Check when grammar and graph exist, but no paths match."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")

    cfg = CFG.from_text("S -> b")
    assert algorithm(cfg, graph) == set()


def test_cfg_to_rsm_structure():
    """Check that RSM built from CFG has the start symbol and a box for it."""
    cfg = CFG.from_text("""
        S -> a S b | epsilon
    """)
    rsm = cfg_to_rsm(cfg)

    assert rsm.initial_label.value == "S"
    labels = {label.value for label in rsm.labels}
    assert "S" in labels
    assert len(rsm.boxes) >= 1


def test_cfg_to_rsm_multiple_nonterminals():
    """Check that a box is created for every nonterminal."""
    cfg = CFG.from_text("""
        S -> A B
        A -> a
        B -> b
    """)
    rsm = cfg_to_rsm(cfg)

    labels = {label.value for label in rsm.labels}
    assert {"S", "A", "B"} <= labels
    assert rsm.initial_label.value == "S"


def test_cfg_to_rsm_preserves_language():
    """Check that RSM built from CFG accepts the same words as CFG."""
    cfg = CFG.from_text("""
        S -> a S b | a b
    """)
    rsm = cfg_to_rsm(cfg)

    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="b")

    result = tensor_based_cfpq(rsm, graph)
    assert (1, 3) in result
    assert (0, 4) in result


def test_ebnf_to_rsm_structure():
    """Check that RSM built from EBNF has the start symbol."""
    rsm = ebnf_to_rsm("S -> (a | b)* c")

    assert rsm.initial_label.value == "S"
    labels = {label.value for label in rsm.labels}
    assert "S" in labels


def test_ebnf_to_rsm_accepts_word():
    """Check that RSM built from EBNF accepts a word produced by this EBNF."""
    rsm = ebnf_to_rsm("S -> a b")

    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    assert tensor_based_cfpq(rsm, graph) == {(0, 2)}


def test_tensor_cfpq_nested_ab():
    """Check a^n b^n on a chain graph (tensor-specific)."""
    cfg = CFG.from_text("""
        S -> a S b | a b
    """)
    rsm = cfg_to_rsm(cfg)

    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(3, 4, label="b")

    result = tensor_based_cfpq(rsm, graph)
    assert (1, 3) in result
    assert (0, 4) in result
