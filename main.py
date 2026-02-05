import logging
import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import networkit as nk
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
from joblib import Parallel, delayed


def setup_logging():
    """Configure logging with INFO level, console output, and file output."""
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sp_network_{timestamp}.log"

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Log file: {log_file.absolute()}")


def load_od_zones(path: str) -> gpd.GeoDataFrame:
    """Load origin-destination zones shapefile.

    Args:
        path: Path to the OD zones shapefile.

    Returns:
        GeoDataFrame containing OD zone geometries.
    """
    logging.info("Loading OD zones shapefile...")
    od_zones = gpd.read_file(path)
    logging.info(f"OD zones loaded: {len(od_zones)} zone(s)")
    return od_zones


def filter_zones_by_municipality(
    od_zones: gpd.GeoDataFrame, municipality_id: int = 36
) -> gpd.GeoDataFrame:
    """Filter OD zones by municipality.

    Args:
        od_zones: GeoDataFrame with OD zones.
        municipality_id: Municipality ID to filter (36 = São Paulo).

    Returns:
        Filtered GeoDataFrame.
    """
    logging.info(f"Filtering OD zones by municipality ID {municipality_id}...")
    filtered = od_zones[od_zones["NumeroMuni"] == municipality_id]
    logging.info(f"Filtered to {len(filtered)} zones (from {len(od_zones)})")
    return filtered


def load_graph_from_zones(od_zones: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """Download road network from OpenStreetMap using zones as boundary.

    Args:
        od_zones: GeoDataFrame of OD zones.

    Returns:
        NetworkX MultiDiGraph representing the road network.
    """
    logging.info("Creating network graph from polygon...")
    graph_area = od_zones.to_crs(4326).geometry.make_valid().union_all().buffer(0)
    logging.info("Union of zones created successfully")

    logging.info("Downloading road network graph from OSM...")
    graph = ox.graph_from_polygon(graph_area, network_type="drive")
    logging.info(f"Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    return graph


def plot_graph(graph: nx.MultiDiGraph):
    """Visualize the road network graph.

    Args:
        graph: NetworkX MultiDiGraph to plot.
    """
    logging.info("Plotting graph...")
    ox.plot_graph(graph)
    logging.info("Graph plotted successfully")


def _nx_to_nk_graph(graph: nx.MultiDiGraph) -> tuple[nk.Graph, dict, dict]:
    """Convert NetworkX MultiDiGraph to NetworKit undirected Graph.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Tuple of (NetworKit graph, node_to_idx mapping, idx_to_node mapping).
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    nk_graph = nk.Graph(len(node_list), directed=False, weighted=False)
    for u, v in graph.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        if not nk_graph.hasEdge(u_idx, v_idx):
            nk_graph.addEdge(u_idx, v_idx)

    return nk_graph, node_to_idx, idx_to_node


def _nx_to_nk_weighted_graph(
    graph: nx.Graph, weight: str
) -> tuple[nk.Graph, dict, dict]:
    """Convert NetworkX Graph to NetworKit weighted undirected Graph.

    Args:
        graph: NetworkX Graph (undirected).
        weight: Edge weight attribute name.

    Returns:
        Tuple of (NetworKit weighted graph, node_to_idx mapping, idx_to_node mapping).
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    nk_graph = nk.Graph(len(node_list), directed=False, weighted=True)
    for u, v, data in graph.edges(data=True):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        w = data.get(weight, 1.0)
        if w is None or w <= 0:
            w = 1.0
        if not nk_graph.hasEdge(u_idx, v_idx):
            nk_graph.addEdge(u_idx, v_idx, w)

    return nk_graph, node_to_idx, idx_to_node


def _dijkstra_sum_for_source_nk(nk_graph: nk.Graph, source: int) -> float:
    """Compute sum of weighted distances from source to all other nodes using NetworKit.

    Args:
        nk_graph: NetworKit weighted Graph.
        source: Source node index.

    Returns:
        Sum of distances from source to all reachable nodes.
    """
    dijkstra = nk.distance.Dijkstra(nk_graph, source)
    dijkstra.run()
    distances = dijkstra.getDistances()
    total = 0.0
    for dist in distances:
        if 0 < dist < 1e308:
            total += dist
    return total


def _parallel_avg_shortest_path_nk(nk_graph: nk.Graph) -> float:
    """Calculate average shortest path length using parallel Dijkstra with NetworKit.

    Args:
        nk_graph: NetworKit weighted Graph.

    Returns:
        Average shortest path length.
    """
    n = nk_graph.numberOfNodes()
    results = Parallel(n_jobs=os.cpu_count(), prefer="threads")(
        delayed(_dijkstra_sum_for_source_nk)(nk_graph, source) for source in range(n)
    )
    total = sum(results)
    return total / (n * (n - 1))


def _bfs_distances_for_source(
    nk_graph: nk.Graph, source: int, n: int
) -> tuple[float, int]:
    """Compute sum of distances from source to all higher-indexed nodes.

    Args:
        nk_graph: NetworKit Graph.
        source: Source node index.
        n: Total number of nodes.

    Returns:
        Tuple of (total_distance, count_of_valid_pairs).
    """
    bfs = nk.distance.BFS(nk_graph, source)
    bfs.run()
    distances = bfs.getDistances()
    total_dist = 0.0
    count = 0
    for target in range(source + 1, n):
        dist = distances[target]
        if 0 < dist < 1e308:
            total_dist += dist
            count += 1
    return total_dist, count


def _compute_clustering_networkit(graph: nx.MultiDiGraph) -> dict:
    """Compute local clustering coefficient using NetworKit.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Dictionary mapping node IDs to clustering coefficient values.
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    nk_graph = nk.Graph(len(node_list), directed=False)
    for u, v in graph.edges():
        if u == v:
            continue
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        if not nk_graph.hasEdge(u_idx, v_idx):
            nk_graph.addEdge(u_idx, v_idx)

    lcc = nk.centrality.LocalClusteringCoefficient(nk_graph)
    lcc.run()
    scores = lcc.scores()

    return {node: scores[node_to_idx[node]] for node in node_list}


def _compute_betweenness_networkit(graph: nx.MultiDiGraph) -> dict:
    """Compute betweenness centrality using NetworKit.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Dictionary mapping node IDs to betweenness centrality values.
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    nk_graph = nk.Graph(len(node_list), directed=True)
    for u, v in graph.edges():
        nk_graph.addEdge(node_to_idx[u], node_to_idx[v])

    bc = nk.centrality.Betweenness(nk_graph, normalized=True)
    bc.run()
    scores = bc.scores()

    return {node: scores[node_to_idx[node]] for node in node_list}


def _compute_edge_betweenness_networkit(graph: nx.MultiDiGraph) -> dict:
    """Compute edge betweenness centrality using NetworKit.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        Dictionary mapping (u, v) tuples to edge betweenness centrality values.
    """
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    nk_graph = nk.Graph(len(node_list), directed=True)
    edge_map = {}

    for u, v in graph.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        if not nk_graph.hasEdge(u_idx, v_idx):
            nk_graph.addEdge(u_idx, v_idx)
        edge_map[(u_idx, v_idx)] = (u, v)

    nk_graph.indexEdges()
    bc = nk.centrality.Betweenness(
        nk_graph, normalized=True, computeEdgeCentrality=True
    )
    bc.run()

    edge_scores = bc.edgeScores()
    result = {}
    for u_idx, v_idx in edge_map:
        edge_id = nk_graph.edgeId(u_idx, v_idx)
        u, v = idx_to_node[u_idx], idx_to_node[v_idx]
        result[(u, v)] = edge_scores[edge_id]

    return result


def calculate_node_parameters(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Calculate local parameters for each node using NetworKit.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        DataFrame with node parameters (k_i, c_i, b_i, avg_l_i).
    """
    logging.info("Calculating node parameters...")

    logging.info("Computing degree for each node...")
    degree = dict(graph.degree())

    logging.info("Computing clustering coefficient using NetworKit...")
    clustering = _compute_clustering_networkit(graph)
    logging.info("Clustering computation completed")

    logging.info("Computing betweenness centrality using NetworKit...")
    betweenness = _compute_betweenness_networkit(graph)
    logging.info("Betweenness computation completed")

    logging.info("Computing average edge length for each node...")
    avg_edge_length = {}
    for node in graph.nodes():
        edges = graph.edges(node, data=True)
        lengths = [data.get("length", 0) for _, _, data in edges]
        avg_edge_length[node] = sum(lengths) / len(lengths) if lengths else 0

    node_data = pd.DataFrame(
        {
            "node": list(graph.nodes()),
            "k_i": [degree[n] for n in graph.nodes()],
            "c_i": [clustering[n] for n in graph.nodes()],
            "b_i": [betweenness[n] for n in graph.nodes()],
            "avg_l_i": [avg_edge_length[n] for n in graph.nodes()],
        }
    )

    logging.info(f"Node parameters calculated for {len(node_data)} nodes")
    return node_data


def calculate_edge_parameters(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Calculate parameters for each edge using NetworKit.

    Args:
        graph: NetworkX MultiDiGraph.

    Returns:
        DataFrame with edge parameters (l_topo, l_eucl, l_manh, length, e_ij).
    """
    logging.info("Calculating edge parameters...")

    logging.info("Computing edge betweenness centrality using NetworKit...")
    edge_betweenness_raw = _compute_edge_betweenness_networkit(graph)
    edge_betweenness = {}
    for u, v, key in graph.edges(keys=True):
        edge_betweenness[(u, v, key)] = edge_betweenness_raw.get((u, v), 0)
    logging.info("Edge betweenness computation completed")

    edges_list = list(graph.edges(keys=True, data=True))
    n_edges = len(edges_list)

    logging.info("Computing distance metrics for edges...")
    edges_data = []
    geod = pyproj.Geod(ellps="WGS84")

    for i, (u, v, key, data) in enumerate(edges_list, 1):
        if i % 500 == 0 or i == n_edges:
            logging.info(f"Processing edge metrics {i}/{n_edges}")

        u_node = graph.nodes[u]
        v_node = graph.nodes[v]
        u_x, u_y = u_node.get("x"), u_node.get("y")
        v_x, v_y = v_node.get("x"), v_node.get("y")

        l_topo = 1
        length = data.get("length", 0)

        if u_x is not None and u_y is not None and v_x is not None and v_y is not None:
            _, _, l_eucl = geod.inv(u_x, u_y, v_x, v_y)
            _, _, dx = geod.inv(u_x, u_y, u_x, v_y)
            _, _, dy = geod.inv(u_x, u_y, v_x, u_y)
            l_manh = abs(dx) + abs(dy)
        else:
            l_eucl = 0
            l_manh = 0

        edges_data.append(
            {
                "u": u,
                "v": v,
                "key": key,
                "l_topo": l_topo,
                "l_eucl": l_eucl,
                "l_manh": l_manh,
                "length": length,
                "e_ij": edge_betweenness.get((u, v, key), 0),
            }
        )

    edge_df = pd.DataFrame(edges_data)
    logging.info(f"Edge parameters calculated for {len(edge_df)} edges")
    return edge_df


def calculate_global_parameters(
    graph: nx.MultiDiGraph,
    node_data: pd.DataFrame,
    edge_data: pd.DataFrame,
) -> dict:
    """Calculate global network parameters using NetworKit.

    Args:
        graph: NetworkX MultiDiGraph.
        node_data: DataFrame with node parameters.
        edge_data: DataFrame with edge parameters.

    Returns:
        Dictionary with global parameters.
    """
    logging.info("Calculating global parameters...")

    N = len(graph.nodes())
    L = len(graph.edges())

    avg_degree = node_data["k_i"].mean()
    avg_clustering = node_data["c_i"].mean()
    avg_l_eucl = edge_data["l_eucl"].mean()
    avg_l_manh = edge_data["l_manh"].mean()
    avg_length = edge_data["length"].mean()

    max_l_eucl = edge_data["l_eucl"].max()
    max_l_manh = edge_data["l_manh"].max()
    max_length = edge_data["length"].max()

    logging.info("Finding largest connected component using NetworKit...")
    nk_graph_full, node_to_idx_full, idx_to_node_full = _nx_to_nk_graph(graph)
    cc = nk.components.ConnectedComponents(nk_graph_full)
    cc.run()
    largest_cc_idx = max(
        range(nk_graph_full.numberOfNodes()), key=lambda i: cc.componentOfNode(i)
    )
    largest_cc_id = cc.componentOfNode(largest_cc_idx)
    largest_cc_nodes = [
        idx_to_node_full[i]
        for i in range(nk_graph_full.numberOfNodes())
        if cc.componentOfNode(i) == largest_cc_id
    ]
    subgraph = graph.subgraph(largest_cc_nodes)
    logging.info(f"Largest connected component has {len(largest_cc_nodes)} nodes")

    logging.info("Computing diameter using NetworKit...")
    nk_subgraph, node_to_idx, _ = _nx_to_nk_graph(subgraph)

    try:
        diam = nk.distance.Diameter(nk_subgraph, algo=nk.distance.DiameterAlgo.EXACT)
        diam.run()
        diameter = int(diam.getDiameter()[0])
        logging.info(f"Diameter calculated: {diameter}")
    except Exception as e:
        diameter = None
        logging.warning(f"Could not calculate diameter: {e}")

    try:
        logging.info(
            "Computing average shortest path length (topological) using parallel BFS..."
        )
        n = nk_subgraph.numberOfNodes()
        n_jobs = os.cpu_count()
        logging.info(f"Using {n_jobs} parallel workers for {n} BFS computations")
        results = Parallel(n_jobs=n_jobs, verbose=10, prefer="threads")(
            delayed(_bfs_distances_for_source)(nk_subgraph, source, n)
            for source in range(n)
        )
        total_dist = sum(r[0] for r in results)
        count = sum(r[1] for r in results)
        avg_shortest_path_topo = total_dist / count if count > 0 else None
        logging.info("Average shortest path length (topological) calculated")
    except Exception as e:
        avg_shortest_path_topo = None
        logging.warning(
            f"Could not calculate average shortest path length (topological): {e}"
        )

    undirected_subgraph = graph.to_undirected().subgraph(largest_cc_nodes)
    n_jobs = os.cpu_count()

    try:
        logging.info(
            f"Computing avg shortest path (physical) with {n_jobs} workers using NetworKit..."
        )
        nk_weighted, _, _ = _nx_to_nk_weighted_graph(
            undirected_subgraph, weight="length"
        )
        avg_shortest_path_length = _parallel_avg_shortest_path_nk(nk_weighted)
        logging.info("Average shortest path length (physical) calculated")
    except Exception as e:
        avg_shortest_path_length = None
        logging.warning(
            f"Could not calculate average shortest path length (physical): {e}"
        )

    try:
        logging.info(
            f"Computing avg shortest path (Euclidean) with {n_jobs} workers using NetworKit..."
        )
        nk_weighted, _, _ = _nx_to_nk_weighted_graph(
            undirected_subgraph, weight="l_eucl"
        )
        avg_shortest_path_eucl = _parallel_avg_shortest_path_nk(nk_weighted)
        logging.info("Average shortest path length (Euclidean) calculated")
    except Exception as e:
        avg_shortest_path_eucl = None
        logging.warning(
            f"Could not calculate average shortest path length (Euclidean): {e}"
        )

    try:
        logging.info(
            f"Computing avg shortest path (Manhattan) with {n_jobs} workers using NetworKit..."
        )
        nk_weighted, _, _ = _nx_to_nk_weighted_graph(
            undirected_subgraph, weight="l_manh"
        )
        avg_shortest_path_manh = _parallel_avg_shortest_path_nk(nk_weighted)
        logging.info("Average shortest path length (Manhattan) calculated")
    except Exception as e:
        avg_shortest_path_manh = None
        logging.warning("Could not calculate average shortest path length (Manhattan)")

    p_random = (2 * L) / (N * (N - 1)) if N > 1 else 0
    k_star = p_random * (N - 1)
    c_star = k_star / N if N > 0 else 0
    l_star = np.log(N) / np.log(k_star) if k_star > 1 else None

    logging.info(
        f"Theoretical random graph G(N,p): p={p_random:.6f}, k*={k_star:.4f}, "
        f"c*={c_star:.6f}, l*={l_star:.4f}"
        if l_star is not None
        else f"Theoretical random graph G(N,p): p={p_random:.6f}, k*={k_star:.4f}, "
        f"c*={c_star:.6f}, l*=N/A"
    )

    global_params = {
        "N": N,
        "L": L,
        "avg_k": avg_degree,
        "avg_c": avg_clustering,
        "avg_l_eucl": avg_l_eucl,
        "avg_l_manh": avg_l_manh,
        "avg_length": avg_length,
        "max_l_eucl": max_l_eucl,
        "max_l_manh": max_l_manh,
        "max_length": max_length,
        "diameter_D": diameter,
        "avg_shortest_path_topo": avg_shortest_path_topo,
        "avg_shortest_path_length": avg_shortest_path_length,
        "avg_shortest_path_eucl": avg_shortest_path_eucl,
        "avg_shortest_path_manh": avg_shortest_path_manh,
        "p_random": p_random,
        "k_star": k_star,
        "c_star": c_star,
        "l_star": l_star,
    }

    logging.info(f"Global parameters calculated: N={N}, L={L}")
    return global_params


def add_parameters_to_graph(
    graph: nx.MultiDiGraph, node_data: pd.DataFrame, edge_data: pd.DataFrame
) -> nx.MultiDiGraph:
    """Add calculated parameters as node and edge attributes to the graph.

    Args:
        graph: NetworkX MultiDiGraph.
        node_data: DataFrame with node parameters.
        edge_data: DataFrame with edge parameters.

    Returns:
        Graph with added attributes.
    """
    logging.info("Adding parameters to graph as attributes...")

    for _, row in node_data.iterrows():
        node = row["node"]
        graph.nodes[node]["k_i"] = int(row["k_i"])
        graph.nodes[node]["c_i"] = float(row["c_i"])
        graph.nodes[node]["b_i"] = float(row["b_i"])
        graph.nodes[node]["avg_l_i"] = float(row["avg_l_i"])

    for _, row in edge_data.iterrows():
        u, v, key = row["u"], row["v"], int(row["key"])
        if graph.has_edge(u, v, key):
            l_topo_val = row["l_topo"]
            graph.edges[u, v, key]["l_topo"] = (
                int(l_topo_val) if l_topo_val != float("inf") else -1
            )
            graph.edges[u, v, key]["l_eucl"] = float(row["l_eucl"])
            graph.edges[u, v, key]["l_manh"] = float(row["l_manh"])
            graph.edges[u, v, key]["length"] = float(row["length"])
            graph.edges[u, v, key]["e_ij"] = float(row["e_ij"])

    logging.info("Parameters added to graph")
    return graph


def graph_to_spatial_objects(
    graph: nx.MultiDiGraph,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert graph nodes and edges to spatial GeoDataFrames.

    Args:
        graph: NetworkX MultiDiGraph with spatial data.

    Returns:
        Tuple of (nodes GeoDataFrame, edges GeoDataFrame).
    """
    logging.info("Converting graph to spatial objects...")

    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False)

    logging.info(
        f"Created spatial objects: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges"
    )
    return nodes_gdf, edges_gdf


def save_results_txt(global_params: dict, output_path: Path):
    """Save global and average parameters to a text file.

    Args:
        global_params: Dictionary with global parameters.
        output_path: Path to save the results.txt file.
    """
    logging.info(f"Writing results to {output_path}")

    with open(output_path, "w") as f:
        f.write("Global Network Parameters\n")
        f.write("=" * 50 + "\n\n")

        f.write("Number of nodes (N): {}\n".format(global_params["N"]))
        f.write("Number of edges (L): {}\n\n".format(global_params["L"]))

        f.write("Average Parameters:\n")
        f.write("-" * 50 + "\n")
        f.write("Average degree (<k>): {:.4f}\n".format(global_params["avg_k"]))
        f.write(
            "Average clustering coefficient (<c>): {:.4f}\n".format(
                global_params["avg_c"]
            )
        )
        f.write(
            "Average Euclidean distance (<l_eucl>): {:.4f} m\n".format(
                global_params["avg_l_eucl"]
            )
        )
        f.write(
            "Average Manhattan distance (<l_manh>): {:.4f} m\n".format(
                global_params["avg_l_manh"]
            )
        )
        f.write(
            "Average physical length (<length>): {:.4f} m\n".format(
                global_params["avg_length"]
            )
        )
        f.write("\n")

        f.write("Average Shortest Path Lengths:\n")
        f.write("-" * 50 + "\n")
        if global_params["avg_shortest_path_topo"] is not None:
            f.write(
                "Topological (edges): {:.4f}\n".format(
                    global_params["avg_shortest_path_topo"]
                )
            )
        else:
            f.write("Topological (edges): N/A (graph not fully connected)\n")

        if global_params["avg_shortest_path_length"] is not None:
            f.write(
                "Physical length (m): {:.4f}\n".format(
                    global_params["avg_shortest_path_length"]
                )
            )
        else:
            f.write("Physical length (m): N/A (graph not fully connected)\n")

        if global_params["avg_shortest_path_eucl"] is not None:
            f.write(
                "Euclidean distance (m): {:.4f}\n".format(
                    global_params["avg_shortest_path_eucl"]
                )
            )
        else:
            f.write("Euclidean distance (m): N/A (graph not fully connected)\n")

        if global_params["avg_shortest_path_manh"] is not None:
            f.write(
                "Manhattan distance (m): {:.4f}\n".format(
                    global_params["avg_shortest_path_manh"]
                )
            )
        else:
            f.write("Manhattan distance (m): N/A (graph not fully connected)\n")
        f.write("\n")

        f.write("Maximum Parameters:\n")
        f.write("-" * 50 + "\n")
        f.write(
            "Maximum Euclidean distance (max_l_eucl): {:.4f} m\n".format(
                global_params["max_l_eucl"]
            )
        )
        f.write(
            "Maximum Manhattan distance (max_l_manh): {:.4f} m\n".format(
                global_params["max_l_manh"]
            )
        )
        f.write(
            "Maximum physical length (max_length): {:.4f} m\n".format(
                global_params["max_length"]
            )
        )
        if global_params["diameter_D"] is not None:
            f.write("Diameter (D): {}\n".format(global_params["diameter_D"]))
        else:
            f.write("Diameter (D): N/A (graph not fully connected)\n")
        f.write("\n")

        f.write("Theoretical Random Graph G(N, p):\n")
        f.write("-" * 50 + "\n")
        f.write("p = 2L / N(N-1): {:.6f}\n".format(global_params["p_random"]))
        f.write("k* = p(N-1): {:.4f}\n".format(global_params["k_star"]))
        f.write("c* = k*/N: {:.6f}\n".format(global_params["c_star"]))
        if global_params["l_star"] is not None:
            f.write("l* = logN/logk*: {:.4f}\n".format(global_params["l_star"]))
        else:
            f.write("l* = logN/logk*: N/A (k* <= 1)\n")

    logging.info(f"Results saved to {output_path}")


def main():
    """Execute the main pipeline: load OD zones, download network, and compute parameters."""
    setup_logging()

    TEST_RUN = True
    TEST_DISTRICTS = [80, 67]
    SP_MUNICIPALITY_ID = 36

    PATH_OD_ZONES = "data/raw/od_zones/Zonas_2023.shp"

    od_zones = load_od_zones(PATH_OD_ZONES)
    od_zones_sp = filter_zones_by_municipality(od_zones, SP_MUNICIPALITY_ID)

    if TEST_RUN:
        logging.info(f"Test run mode: filtering to districts {TEST_DISTRICTS}")
        od_zones_sp = od_zones_sp[od_zones_sp["NumDistrit"].isin(TEST_DISTRICTS)]
        logging.info(f"Test run: {len(od_zones_sp)} zone(s) after district filter")

    graph = load_graph_from_zones(od_zones_sp)

    node_params = calculate_node_parameters(graph)
    edge_params = calculate_edge_parameters(graph)

    graph = add_parameters_to_graph(graph, node_params, edge_params)

    global_params = calculate_global_parameters(graph, node_params, edge_params)
    nodes_gdf, edges_gdf = graph_to_spatial_objects(graph)

    output_dir = Path("data/test") if TEST_RUN else Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to {output_dir}/")

    zones_output = output_dir / "od_zones_sp.gpkg"
    od_zones_sp.to_file(zones_output, driver="GPKG")
    logging.info(f"Filtered OD zones saved to {zones_output}")

    graph_output = output_dir / "road_network.graphml"
    ox.save_graphml(graph, graph_output)
    logging.info(f"Road network graph saved to {graph_output}")

    nodes_spatial_output = output_dir / "nodes.gpkg"
    nodes_gdf.to_file(nodes_spatial_output, driver="GPKG")
    logging.info(f"Nodes spatial object saved to {nodes_spatial_output}")

    edges_spatial_output = output_dir / "edges.gpkg"
    edges_gdf.to_file(edges_spatial_output, driver="GPKG")
    logging.info(f"Edges spatial object saved to {edges_spatial_output}")

    results_txt_output = output_dir / "results.txt"
    save_results_txt(global_params, results_txt_output)

    return od_zones_sp, graph


if __name__ == "__main__":
    main()
