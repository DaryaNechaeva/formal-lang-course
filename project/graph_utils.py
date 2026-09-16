from typing import NamedTuple
import cfpq_data


class GraphInfo(NamedTuple):
    """Information about a labeled graph."""

    number_of_nodes: int
    number_of_edges: int
    labels: set[str]


def get_graph_info(graph_name: str) -> GraphInfo:
    """Return basic information about a graph from the CFPQ_Data dataset."""

    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)

    labels = {label for _, _, label in graph.edges(data="label") if label is not None}

    return GraphInfo(
        number_of_nodes=graph.number_of_nodes(),
        number_of_edges=graph.number_of_edges(),
        labels=labels,
    )
