# SP Network Analysis

Geospatial network analysis project for the City of São Paulo. This project loads origin-destination (OD) zones, downloads road network data from OpenStreetMap, and calculates various network parameters.

## Features

- Load and filter origin-destination (OD) zones for the City of São Paulo
- Download road network data from OpenStreetMap using OSMnx
- Calculate network parameters:
  - Node parameters: degree, clustering coefficient, betweenness centrality, average edge length
  - Edge parameters: topological/euclidean/manhattan distances, physical length, edge betweenness centrality
  - Global parameters: network size, averages, maximums, diameter, shortest paths, theoretical random graph comparisons
- **High-performance computing** using NetworKit (C++ backend) for graph algorithms
- **Parallel processing** using joblib for CPU-intensive operations
- Automatic logging to `log/` directory with timestamps
- Export results in multiple formats (GeoPackage, GraphML, text)
- **District-level aggregation**: Mean, median and max of network parameters per district

## Requirements

- Python 3.12+
- uv package manager

## Installation

Install dependencies using uv:

```bash
uv sync
```

## Usage

Run the script:

```bash
uv run python main.py
```

### District Aggregation

After running `main.py`, aggregate network parameters by district:

```bash
uv run python aggregate_districts.py
```

This generates `data/output/district_summary.gpkg` with district polygons (dissolved from OD zones) and summary statistics (mean, median, max) for `k_i`, `c_i`, `b_i`, and `e_ij`.

## Configuration

### Test Mode

Edit `main.py` to configure test mode:

```python
TEST_RUN = True  # Run test mode with limited districts
TEST_DISTRICTS = [80, 67]  # Districts to include in test
SP_MUNICIPALITY_ID = 36  # São Paulo municipality ID
```

When `TEST_RUN = True`, results are saved to `data/test/`
When `TEST_RUN = False`, results are saved to `data/output/`

## Input Data

Place your input files in:
- `data/raw/od_zones/Zonas_2023.shp` - Origin-destination zones shapefile

The shapefile should contain:
- `NumeroMuni`: Municipality ID (36 = São Paulo)
- `NumDistrit`: District number (for test filtering)
- `geometry`: Zone polygons

## Output Files

The script generates the following outputs:

- `od_zones_sp.gpkg` - Filtered OD zones for São Paulo (GeoPackage)
- `road_network.graphml` - Road network graph with all parameters
- `nodes.gpkg` - Nodes with calculated parameters (GeoPackage)
- `edges.gpkg` - Edges with calculated parameters (GeoPackage)
- `results.txt` - Summary of global and average network parameters
- `district_summary.gpkg` - District polygons with aggregated network parameters (mean, median, max)

## Network Parameters

### Node Parameters (Local)
- `k_i`: Degree of node i
- `c_i`: Clustering coefficient of node i
- `b_i`: Betweenness centrality of node i
- `avg_l_i`: Average length of edges connected to node i

### Edge Parameters
- `l_topo`: Topological distance (always 1 for direct edges)
- `l_eucl`: Euclidean distance (geodesic straight-line)
- `l_manh`: Manhattan distance
- `length`: Physical road length in meters
- `e_ij`: Edge betweenness centrality

### Global Parameters
- `N`: Number of nodes
- `L`: Number of edges
- `<k>`: Average degree
- `<c>`: Average clustering coefficient
- `<l_eucl>`, `<l_manh>`, `<length>`: Average distances
- `D`: Network diameter
- Average shortest path lengths (topological, physical, euclidean, manhattan)
- Theoretical random graph G(N,p) parameters: `p`, `k*`, `c*`, `l*`

## Performance

All graph computations use **NetworKit** (C++ backend), providing significant speedup over pure NetworkX.

### Optimized Operations

| Operation | Backend |
|-----------|---------|
| Node betweenness centrality | `nk.centrality.Betweenness` |
| Edge betweenness centrality | `nk.centrality.Betweenness(computeEdgeCentrality=True)` |
| Clustering coefficient | `nk.centrality.LocalClusteringCoefficient` |
| Diameter | `nk.distance.Diameter` |
| Average shortest path | Parallel BFS/Dijkstra across all nodes |
