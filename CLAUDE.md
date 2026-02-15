# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geospatial network analysis project for the City of São Paulo. The project loads OD zones from the metropolitan region, filters zones within the municipality of São Paulo, downloads road network data from OpenStreetMap, calculates network parameters, and aggregates results by OD zone, district, and subprefeitura.

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

**Run aggregation (after main.py):**
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
2. **Spatial join**: Assigns nodes and edges to OD zones via spatial join
3. **Zone aggregation**: Computes mean/median/max of k_i, c_i, b_i (nodes) and e_ij (edges) per OD zone
4. **District aggregation**: Groups zones by district, dissolves polygons, computes summary statistics
5. **Subprefeitura aggregation**: Maps districts to subprefeituras (hardcoded mapping), dissolves polygons, computes summary statistics
6. **Export**: Saves zone_summary.gpkg, district_summary.gpkg, and subprefeitura_summary.gpkg

The results notebook `index.ipynb` visualizes outputs with choropleths and summary tables. A static HTML export (`index.html`) is also generated.

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
- `assign_nodes_to_zones()`: Spatial join nodes to OD zones
- `assign_edges_to_zones()`: Spatial join edges (via representative point) to OD zones
- `aggregate_node_params()`: Groupby column, compute mean/median/max for k_i, c_i, b_i
- `aggregate_edge_params()`: Groupby column, compute mean/median/max for e_ij
- `build_zone_geodataframe()`: Create zone-level summary with aggregated statistics
- `build_district_geodataframe()`: Dissolve zones into district polygons and merge statistics
- `build_subprefeitura_geodataframe()`: Dissolve districts into subprefeitura polygons and merge statistics

### Directory Structure

- `data/raw/od_zones/`: Origin-destination zone shapefiles (527 zones, 39 municipalities)
- `data/output/`: Output directory for full production runs
- `data/test/`: Output directory for test runs (when `TEST_RUN = True`)
- `cache/`: OSMnx automatically caches downloaded network data here
- `log/`: Timestamped log files
- `index.ipynb`: Results visualization notebook (choropleths and summary tables)
- `index.html`: Static HTML export of the notebook

### Key Dependencies

- **geopandas**: Spatial data operations and shapefile I/O
- **osmnx**: Download and analyze OpenStreetMap road networks
- **networkx**: Graph data structure and analysis
- **networkit**: High-performance graph algorithms (C++ backend)
- **joblib**: Parallel processing for CPU-intensive computations
- **pandas**: Tabular data operations
- **numpy**: Numerical operations
- **matplotlib**: Plotting and visualization
- **contextily**: Basemap tiles for choropleth maps
- **nbclient/nbconvert**: Notebook execution and HTML export
- **pyproj**: Geodesic distance calculations (WGS84 ellipsoid, transitive dependency)

## Known Patterns

- The project uses structured logging throughout with INFO level messages
- Logs are saved to `log/` directory with timestamp-based filenames
- Geometry operations include validation (`.make_valid()`) and buffering
- When dissolving zones, `GeometryCollection` results must be filtered to extract only polygon parts (using `shapely.ops.unary_union`)
- OSMnx downloads are cached automatically to avoid repeated API calls
- Graph uses `network_type="drive"` for road networks
- `aggregate_districts.py` contains a hardcoded `DISTRICT_TO_SUBPREFEITURA` mapping (105 entries) for all São Paulo districts
- The results notebook uses dark-themed choropleths with CartoDB DarkMatter basemap via contextily

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
