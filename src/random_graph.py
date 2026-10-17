"""Random null model for the São Paulo road network.

Builds a directed configuration model that preserves the in-degree and
out-degree sequences of the real network, exports it to GraphML, and reports
N, L, the average clustering coefficient and the average shortest path
length.

Run with:
    uv run python src/random_graph.py

Requires a previous ``main.py`` run: reads ``road_network.graphml`` from
``data/output`` (or ``data/test`` when ``TEST_RUN`` is True) and writes
``random_road_network.graphml`` and ``random_results.txt`` beside it.

The heavy step (the per-node BFS behind ``<l>``) scales with
``os.cpu_count()`` and is CPU-bound on a graph of this size.
"""

import logging

import networkx as nx
import numpy as np
import osmnx as ox

from main import (
    PROJECT_ROOT,
    _compute_avg_shortest_path_per_node,
    _compute_clustering_networkx,
    setup_logging,
)

TEST_RUN = False
SEED = 42


def degree_sequences(graph: nx.MultiDiGraph) -> tuple[list[int], list[int]]:
    """Extract in-degree and out-degree sequences from a directed graph.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Tuple of (in_degree_sequence, out_degree_sequence) in graph.nodes() order.
    """
    in_sequence = [degree for _, degree in graph.in_degree()]
    out_sequence = [degree for _, degree in graph.out_degree()]
    return in_sequence, out_sequence


def generate_random_graph(
    in_sequence: list[int], out_sequence: list[int], seed: int = SEED
) -> nx.MultiDiGraph:
    """Generate a random directed graph with the given degree sequences.

    Uses the directed configuration model, which preserves both degree
    sequences exactly while randomizing the wiring. Self-loops and parallel
    edges are kept as generated.

    Args:
        in_sequence: In-degree sequence.
        out_sequence: Out-degree sequence.
        seed: Random seed for reproducibility.

    Returns:
        NetworkX MultiDiGraph with nodes labelled 0..N-1.
    """
    return nx.directed_configuration_model(in_sequence, out_sequence, seed=seed)


def describe_topology(graph: nx.MultiDiGraph) -> dict:
    """Count nodes, edges, self-loops and parallel edges of a graph.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Dictionary with N, L, self_loops and parallel_edges.
    """
    n_edges = graph.number_of_edges()
    unique_pairs = len({(u, v) for u, v, _ in graph.edges(keys=True)})

    return {
        "N": graph.number_of_nodes(),
        "L": n_edges,
        "self_loops": nx.number_of_selfloops(graph),
        "parallel_edges": n_edges - unique_pairs,
    }


def compute_metrics(graph: nx.MultiDiGraph) -> dict:
    """Compute the average clustering coefficient and shortest path length.

    ``avg_l`` is the mean of the per-node averages over every node, including
    the nodes with an l_i of 0 (those that reach nothing), which matches the
    ``nodes["l_i"].mean()`` used as ``<l>`` in ``table.py``.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Dictionary with avg_c, avg_l and n_zero_l.
    """
    logging.info("Computing clustering coefficient using NetworkX (parallel)...")
    clustering = np.array(list(_compute_clustering_networkx(graph).values()))
    logging.info("Clustering computation completed")

    logging.info("Computing average shortest path length per node using NetworKit...")
    avg_shortest_path = np.array(
        list(_compute_avg_shortest_path_per_node(graph).values())
    )
    logging.info("Average shortest path length per node completed")

    return {
        "avg_c": float(clustering.mean()),
        "avg_l": float(avg_shortest_path.mean()),
        "n_zero_l": int((avg_shortest_path == 0).sum()),
    }


def save_random_results(metrics: dict, topology: dict, output_path):
    """Write the random graph metrics to a text file.

    Args:
        metrics: Dictionary returned by compute_metrics.
        topology: Dictionary returned by describe_topology.
        output_path: Path to the output text file.
    """
    lines = [
        "Random Directed Configuration Model",
        "=" * 50,
        "",
        f"Seed: {SEED}",
        "",
        "Main Parameters:",
        "-" * 50,
        f"Number of nodes (N): {topology['N']}",
        f"Number of edges (L): {topology['L']}",
        f"Average clustering coefficient (<c>): {metrics['avg_c']:.6f}",
        f"Average shortest path length (<l>): {metrics['avg_l']:.4f}",
        "",
        "Graph Composition:",
        "-" * 50,
        f"Self-loops: {topology['self_loops']}",
        f"Parallel edges: {topology['parallel_edges']}",
        f"Nodes with l_i = 0 (reach nothing): {metrics['n_zero_l']}",
        "",
        "Note: <l> is the mean of l_i over all nodes, including the ones with",
        "l_i = 0, matching nodes[\"l_i\"].mean() in table.py. The graph is not",
        "strongly connected, so each l_i itself averages only over the targets",
        "that node actually reaches.",
        "",
    ]
    output_path.write_text("\n".join(lines))
    logging.info(f"Results saved to {output_path}")


def main():
    """Generate a random null model from the SP network degree sequences."""
    setup_logging()

    output_dir = (
        PROJECT_ROOT / "data/test" if TEST_RUN else PROJECT_ROOT / "data/output"
    )
    graph_path = output_dir / "road_network.graphml"

    logging.info(f"Loading graph from {graph_path}")
    graph = ox.load_graphml(graph_path)
    logging.info(f"Loaded graph: {len(graph)} nodes, {len(graph.edges)} edges")

    in_sequence, out_sequence = degree_sequences(graph)
    logging.info(
        f"Degree sequences extracted: sum(in) = {sum(in_sequence)}, "
        f"sum(out) = {sum(out_sequence)}"
    )

    logging.info(f"Generating random graph with seed {SEED}...")
    random_graph = generate_random_graph(in_sequence, out_sequence)

    topology = describe_topology(random_graph)
    logging.info(
        f"Random graph: {topology['N']} nodes, {topology['L']} edges, "
        f"{topology['self_loops']} self-loops, "
        f"{topology['parallel_edges']} parallel edges"
    )

    random_path = output_dir / "random_road_network.graphml"
    nx.write_graphml(random_graph, random_path)
    logging.info(f"Random graph saved to {random_path}")

    metrics = compute_metrics(random_graph)
    save_random_results(metrics, topology, output_dir / "random_results.txt")

    logging.info(
        f"N = {topology['N']}, L = {topology['L']}, "
        f"<c> = {metrics['avg_c']:.6f}, <l> = {metrics['avg_l']:.4f}"
    )
    logging.info(f"Nodes with l_i = 0: {metrics['n_zero_l']}")


if __name__ == "__main__":
    main()
