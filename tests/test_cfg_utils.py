import networkx as nx
from pyformlang.cfg import CFG, Variable, Terminal, Epsilon
from project.cfg_utils import cfg_to_weak_normal_form, hellings_based_cfpq


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


def test_hellings_simple_path():
    """Check a simple path without cycles."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    cfg = CFG.from_text("""
        S -> A B
        A -> a
        B -> b
    """)

    result = hellings_based_cfpq(cfg, graph)
    assert result == {(0, 2)}


def test_hellings_empty_start_final():
    """Check that empty start_nodes and final_nodes mean all nodes."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")

    cfg = CFG.from_text("S -> a")

    result = hellings_based_cfpq(cfg, graph, start_nodes=set(), final_nodes=set())
    assert result == {(0, 1)}


def test_hellings_specific_start_final():
    """Check filtering by specified start and final nodes."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")

    cfg = CFG.from_text("S -> a")

    result = hellings_based_cfpq(cfg, graph, start_nodes={0}, final_nodes={1})
    assert result == {(0, 1)}


def test_hellings_cycle():
    """Check the algorithm on a graph with a cycle."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="a")

    cfg = CFG.from_text("""
        S -> A S
        S -> A
        A -> a
    """)

    result = hellings_based_cfpq(cfg, graph)

    assert (0, 1) in result
    assert (1, 0) in result
    assert (0, 0) in result
    assert (1, 1) in result


def test_hellings_epsilon_rule():
    """Check that epsilon rules produce zero-length path loops."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")

    cfg = CFG.from_text("S -> epsilon")

    result = hellings_based_cfpq(cfg, graph, start_nodes={0}, final_nodes={0})
    assert result == {(0, 0)}
