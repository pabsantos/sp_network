import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely


def setup_logging():
    """Configure logging with INFO level and console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_data(
    output_dir: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load nodes, edges, and OD zones GeoPackage files.

    Args:
        output_dir: Path to the output directory containing the files.

    Returns:
        Tuple of (nodes, edges, zones) GeoDataFrames.
    """
    logging.info("Loading data...")
    nodes = gpd.read_file(output_dir / "nodes.gpkg")
    logging.info(f"Nodes loaded: {len(nodes)}")

    edges = gpd.read_file(output_dir / "edges.gpkg")
    logging.info(f"Edges loaded: {len(edges)}")

    zones = gpd.read_file(output_dir / "od_zones_sp.gpkg")
    logging.info(f"OD zones loaded: {len(zones)}")

    return nodes, edges, zones


def assign_nodes_to_districts(
    nodes: gpd.GeoDataFrame, zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatial join nodes to OD zones to assign district IDs.

    Args:
        nodes: GeoDataFrame with node geometries and parameters.
        zones: GeoDataFrame with OD zone polygons and district info.

    Returns:
        GeoDataFrame of nodes with district assignment.
    """
    logging.info("Assigning nodes to districts via spatial join...")
    zones_proj = zones.to_crs(nodes.crs)
    joined = gpd.sjoin(nodes, zones_proj[["NumDistrit", "NomeDistri", "geometry"]], predicate="within")
    joined = joined.drop(columns=["index_right"])
    logging.info(f"Nodes assigned to districts: {len(joined)} (of {len(nodes)})")
    return joined


def assign_edges_to_districts(
    edges: gpd.GeoDataFrame, zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatial join edges to OD zones using representative points.

    Args:
        edges: GeoDataFrame with edge geometries and parameters.
        zones: GeoDataFrame with OD zone polygons and district info.

    Returns:
        GeoDataFrame of edges with district assignment.
    """
    logging.info("Assigning edges to districts via spatial join (representative point)...")
    zones_proj = zones.to_crs(edges.crs)

    edge_points = edges.copy()
    edge_points["geometry"] = edge_points.geometry.representative_point()

    joined = gpd.sjoin(edge_points, zones_proj[["NumDistrit", "NomeDistri", "geometry"]], predicate="within")
    joined = joined.drop(columns=["index_right"])
    logging.info(f"Edges assigned to districts: {len(joined)} (of {len(edges)})")
    return joined


def aggregate_node_params(nodes_with_district: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate node parameters by district.

    Args:
        nodes_with_district: GeoDataFrame with nodes and district assignment.

    Returns:
        DataFrame with mean, median, max of k_i, c_i, b_i per district.
    """
    logging.info("Aggregating node parameters by district...")
    params = ["k_i", "c_i", "b_i"]
    agg_funcs = {col: ["mean", "median", "max"] for col in params}
    agg_funcs["NomeDistri"] = "first"

    result = nodes_with_district.groupby("NumDistrit").agg(agg_funcs)
    result.columns = [
        f"{col}_{func}" if col != "NomeDistri" else "NomeDistri"
        for col, func in result.columns
    ]
    result = result.reset_index()
    logging.info(f"Node aggregation completed for {len(result)} districts")
    return result


def aggregate_edge_params(edges_with_district: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate edge parameters by district.

    Args:
        edges_with_district: GeoDataFrame with edges and district assignment.

    Returns:
        DataFrame with mean, median, max of e_ij per district.
    """
    logging.info("Aggregating edge parameters by district...")
    agg_funcs = {"e_ij": ["mean", "median", "max"]}

    result = edges_with_district.groupby("NumDistrit").agg(agg_funcs)
    result.columns = [f"{col}_{func}" for col, func in result.columns]
    result = result.reset_index()
    logging.info(f"Edge aggregation completed for {len(result)} districts")
    return result


def build_district_geodataframe(
    zones: gpd.GeoDataFrame,
    node_stats: pd.DataFrame,
    edge_stats: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Dissolve zones into districts and merge aggregated statistics.

    Args:
        zones: GeoDataFrame with OD zone polygons.
        node_stats: DataFrame with aggregated node parameters per district.
        edge_stats: DataFrame with aggregated edge parameters per district.

    Returns:
        GeoDataFrame with district polygons and summary statistics.
    """
    logging.info("Dissolving OD zones into district polygons...")
    zones = zones.copy()
    zones["geometry"] = zones.geometry.make_valid()
    districts = zones.dissolve(by="NumDistrit", as_index=False)[
        ["NumDistrit", "NomeDistri", "geometry"]
    ]
    districts["geometry"] = districts.geometry.apply(
        lambda geom: shapely.ops.unary_union(
            [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        ) if geom.geom_type == "GeometryCollection" else geom
    )
    logging.info(f"Districts created: {len(districts)}")

    districts = districts.merge(node_stats.drop(columns=["NomeDistri"]), on="NumDistrit", how="left")
    districts = districts.merge(edge_stats, on="NumDistrit", how="left")

    return districts


def main():
    """Aggregate network parameters by district and export as GeoPackage."""
    setup_logging()

    output_dir = Path("data/output")
    nodes, edges, zones = load_data(output_dir)

    nodes_with_district = assign_nodes_to_districts(nodes, zones)
    edges_with_district = assign_edges_to_districts(edges, zones)

    node_stats = aggregate_node_params(nodes_with_district)
    edge_stats = aggregate_edge_params(edges_with_district)

    district_gdf = build_district_geodataframe(zones, node_stats, edge_stats)

    output_path = output_dir / "district_summary.gpkg"
    district_gdf.to_file(output_path, driver="GPKG")
    logging.info(f"District summary saved to {output_path}")

    stat_cols = [c for c in district_gdf.columns if c not in ("geometry", "NumDistrit", "NomeDistri")]
    logging.info(f"\nDistrict summary ({len(district_gdf)} districts):")
    logging.info(f"\n{district_gdf[['NomeDistri'] + stat_cols].to_string()}")


if __name__ == "__main__":
    main()
