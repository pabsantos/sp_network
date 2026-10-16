import logging
import re

import geopandas as gpd
import osmnx as ox

from main import PROJECT_ROOT, _compute_clustering_networkx, setup_logging


def update_nodes_gpkg(nodes_path, clustering: dict):
    """Overwrite the c_i column of the nodes GeoPackage with new values.

    Args:
        nodes_path: Path to nodes.gpkg.
        clustering: Dictionary mapping node IDs to clustering coefficient values.
    """
    nodes = gpd.read_file(nodes_path)
    nodes["c_i"] = nodes["osmid"].map(clustering)
    if nodes["c_i"].isna().any():
        raise ValueError("Some nodes in nodes.gpkg have no clustering value")
    nodes.to_file(nodes_path, driver="GPKG")
    logging.info(f"Updated c_i in {nodes_path}")


def update_results_txt(results_path, avg_c: float):
    """Replace the average clustering coefficient line in results.txt.

    Args:
        results_path: Path to results.txt.
        avg_c: New average clustering coefficient.
    """
    text = results_path.read_text()
    text = re.sub(
        r"Average clustering coefficient \(<c>\): .*",
        f"Average clustering coefficient (<c>): {avg_c:.4f}",
        text,
    )
    results_path.write_text(text)
    logging.info(f"Updated <c> in {results_path}")


def main():
    """Recompute only the clustering coefficient from an existing pipeline output."""
    setup_logging()

    TEST_RUN = False

    output_dir = (
        PROJECT_ROOT / "data/test" if TEST_RUN else PROJECT_ROOT / "data/output"
    )
    graph_path = output_dir / "road_network.graphml"

    logging.info(f"Loading graph from {graph_path}")
    graph = ox.load_graphml(graph_path)
    logging.info(f"Loaded graph: {len(graph)} nodes, {len(graph.edges)} edges")

    logging.info("Computing clustering coefficient using NetworkX (parallel)...")
    clustering = _compute_clustering_networkx(graph)
    logging.info("Clustering computation completed")

    for node, value in clustering.items():
        graph.nodes[node]["c_i"] = float(value)
    ox.save_graphml(graph, graph_path)
    logging.info(f"Updated c_i in {graph_path}")

    update_nodes_gpkg(output_dir / "nodes.gpkg", clustering)

    avg_c = sum(clustering.values()) / len(clustering)
    update_results_txt(output_dir / "results.txt", avg_c)
    logging.info(f"New average clustering coefficient: {avg_c:.4f}")


if __name__ == "__main__":
    main()
