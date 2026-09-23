import pathlib
from typing import NamedTuple

import cfpq_data
import networkx as nx
import pydot


class GraphInfo(NamedTuple):
    number_of_nodes: int
    number_of_edges: int
    labels: set[str]


def get_graph_info(graph_name: str) -> GraphInfo:
    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)

    labels = {label for _, _, label in graph.edges(data="label") if label is not None}

    return GraphInfo(
        number_of_nodes=graph.number_of_nodes(),
        number_of_edges=graph.number_of_edges(),
        labels=labels,
    )


def create_two_cycles_graph(
    n: int,
    m: int,
    labels: tuple[str, str],
    filepath: str | pathlib.Path,
) -> nx.MultiDiGraph:
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)

    dot = pydot.Dot(graph_type="digraph")
    for node in graph.nodes():
        dot.add_node(pydot.Node(str(node)))
    for source, target, label in graph.edges(data="label"):
        dot.add_edge(pydot.Edge(str(source), str(target), label=str(label)))
    dot.write_raw(str(filepath))

    return graph
