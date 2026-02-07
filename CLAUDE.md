# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geospatial network analysis project for the City of São Paulo. The project loads OD zones from the metropolitan region, filters zones within the municipality of São Paulo, downloads road network data from OpenStreetMap, calculates network parameters, and aggregates results by district.

## Development Setup

This project uses `uv` as the Python package manager. Python 3.12+ is required.

**Install dependencies:**
```bash
uv sync
```

**Run the main script:**
```bash
uv run python main.py
```

**Run district aggregation (after main.py):**
```bash
uv run python aggregate_districts.py
```

## Project Architecture

### Data Flow

The main pipeline in `main.py` follows this sequence:

1. **Load spatial data**: OD zones shapefile (`Zonas_2023.shp`)
2. **Municipality filtering**: Filters zones within São Paulo municipality (NumeroMuni == 36)
3. **Optional test filtering**: Filters by district for test runs
4. **Network download**: Downloads road network from OSM using filtered zones as boundary
5. **Parameter calculation**: Computes node and edge parameters using NetworKit
6. **Output generation**: Exports results as GeoPackage, GraphML, and text files

The aggregation pipeline in `aggregate_districts.py` follows this sequence:

1. **Load outputs**: Reads nodes.gpkg, edges.gpkg, and od_zones_sp.gpkg from `data/output/`
2. **Spatial join**: Assigns nodes and edges to districts via spatial join with OD zones
3. **Aggregation**: Computes mean, median, max of k_i, c_i, b_i (nodes) and e_ij (edges) per district
4. **Dissolve**: Merges OD zone polygons into district polygons
5. **Export**: Saves district_summary.gpkg with polygons and statistics

### Key Functions

- `load_od_zones()`: Load OD zones shapefile into GeoDataFrame
- `filter_zones_by_municipality()`: Filter zones by municipality ID (36 = São Paulo)
- `load_graph_from_zones()`: Download OSM road network using OSMnx
- `calculate_node_parameters()`: Compute degree, clustering, betweenness, avg edge length
- `calculate_edge_parameters()`: Compute distance metrics and edge betweenness
- `calculate_global_parameters()`: Compute network-wide statistics
- `setup_logging()`: Configure logging with timestamp format

**`aggregate_districts.py`:**

- `load_data()`: Load nodes, edges, and OD zones GeoPackage files
- `assign_nodes_to_districts()`: Spatial join nodes to OD zones for district assignment
- `assign_edges_to_districts()`: Spatial join edges (via representative point) to OD zones
- `aggregate_node_params()`: Groupby district, compute mean/median/max for k_i, c_i, b_i
- `aggregate_edge_params()`: Groupby district, compute mean/median/max for e_ij
- `build_district_geodataframe()`: Dissolve zones into district polygons and merge statistics

### Directory Structure

- `data/raw/od_zones/`: Origin-destination zone shapefiles (527 zones, 39 municipalities)
- `data/output/`: Output directory for full production runs
- `data/test/`: Output directory for test runs (when `TEST_RUN = True`)
- `cache/`: OSMnx automatically caches downloaded network data here
- `log/`: Timestamped log files

### Key Dependencies

- **geopandas**: Spatial data operations and shapefile I/O
- **osmnx**: Download and analyze OpenStreetMap road networks
- **networkx**: Graph data structure and analysis
- **networkit**: High-performance graph algorithms (C++ backend)
- **joblib**: Parallel processing for CPU-intensive computations
- **pandas**: Tabular data operations
- **pyproj**: Geodesic distance calculations (WGS84 ellipsoid)
- **numpy**: Numerical operations

## Known Patterns

- The project uses structured logging throughout with INFO level messages
- Logs are saved to `log/` directory with timestamp-based filenames
- Geometry operations include validation (`.make_valid()`) and buffering
- When dissolving zones, `GeometryCollection` results must be filtered to extract only polygon parts (using `shapely.ops.unary_union`)
- OSMnx downloads are cached automatically to avoid repeated API calls
- Graph uses `network_type="drive"` for road networks

## Performance Optimization

The project uses NetworKit (C++ backend) for all computationally intensive graph algorithms, with parallel processing via joblib.

### When to use NetworKit vs NetworkX

| Algorithm | Use NetworKit | Notes |
|-----------|---------------|-------|
| Betweenness centrality (node) | Yes | `nk.centrality.Betweenness` |
| Betweenness centrality (edge) | Yes | `computeEdgeCentrality=True` |
| Clustering coefficient | Yes | `nk.centrality.LocalClusteringCoefficient` |
| All-pairs shortest paths | Yes | `nk.distance.BFS` for topological |
| Diameter | Yes | `nk.distance.Diameter` |
| Weighted shortest paths | No | NetworkX handles custom weights better |

### Graph conversion pattern

```python
def _nx_to_nk_graph(graph: nx.MultiDiGraph) -> tuple[nk.Graph, dict, dict]:
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    nk_graph = nk.Graph(len(node_list), directed=False)
    for u, v in graph.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        if not nk_graph.hasEdge(u_idx, v_idx):
            nk_graph.addEdge(u_idx, v_idx)

    return nk_graph, node_to_idx, idx_to_node
```

### Parallel processing

Use `joblib.Parallel` for embarrassingly parallel tasks:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=os.cpu_count())(
    delayed(compute_function)(arg) for arg in args_list
)
```

## Other instructions

- When running a python script, use 'uv run'
- Don't use excessive comments on code
- Always document methods using docstrings
- Prefer NetworKit over NetworkX for large graph computations
- Use parallel processing for independent, CPU-intensive operations
