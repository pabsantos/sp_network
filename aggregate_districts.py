import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

DISTRICT_TO_SUBPREFEITURA = {
    "Aricanduva": "Aricanduva/Formosa/Carrão",
    "Carrão": "Aricanduva/Formosa/Carrão",
    "Vila Formosa": "Aricanduva/Formosa/Carrão",
    "Butantã": "Butantã",
    "Morumbi": "Butantã",
    "Raposo Tavares": "Butantã",
    "Rio Pequeno": "Butantã",
    "Vila Sônia": "Butantã",
    "Campo Limpo": "Campo Limpo",
    "Capão Redondo": "Campo Limpo",
    "Vila Andrade": "Campo Limpo",
    "Cidade Dutra": "Capela do Socorro",
    "Grajaú": "Capela do Socorro",
    "Socorro": "Capela do Socorro",
    "Casa Verde": "Casa Verde",
    "Cachoeirinha": "Casa Verde",
    "Limão": "Casa Verde",
    "Cidade Ademar": "Cidade Ademar",
    "Pedreira": "Cidade Ademar",
    "Cidade Tiradentes": "Cidade Tiradentes",
    "Ermelino Matarazzo": "Ermelino Matarazzo",
    "Ponte Rasa": "Ermelino Matarazzo",
    "Freguesia do Ó": "Freguesia do Ó/Brasilândia",
    "Brasilândia": "Freguesia do Ó/Brasilândia",
    "Guaianases": "Guaianases",
    "Lajeado": "Guaianases",
    "Ipiranga": "Ipiranga",
    "Cursino": "Ipiranga",
    "Sacomã": "Ipiranga",
    "Itaim Paulista": "Itaim Paulista",
    "Vila Curuçá": "Itaim Paulista",
    "Itaquera": "Itaquera",
    "Cidade Líder": "Itaquera",
    "José Bonifácio": "Itaquera",
    "Parque do Carmo": "Itaquera",
    "Jabaquara": "Jabaquara",
    "Jaçanã": "Jaçanã/Tremembé",
    "Tremembé": "Jaçanã/Tremembé",
    "Lapa": "Lapa",
    "Barra Funda": "Lapa",
    "Jaguara": "Lapa",
    "Jaguaré": "Lapa",
    "Perdizes": "Lapa",
    "Vila Leopoldina": "Lapa",
    "Jardim Ângela": "M'Boi Mirim",
    "Jardim São Luís": "M'Boi Mirim",
    "Água Rasa": "Mooca",
    "Belém": "Mooca",
    "Brás": "Mooca",
    "Mooca": "Mooca",
    "Pari": "Mooca",
    "Tatuapé": "Mooca",
    "Parelheiros": "Parelheiros",
    "Marsilac": "Parelheiros",
    "Penha": "Penha",
    "Artur Alvim": "Penha",
    "Cangaíba": "Penha",
    "Vila Matilde": "Penha",
    "Perus": "Perus",
    "Anhanguera": "Perus",
    "Alto de Pinheiros": "Pinheiros",
    "Itaim Bibi": "Pinheiros",
    "Jardim Paulista": "Pinheiros",
    "Pinheiros": "Pinheiros",
    "Pirituba": "Pirituba/Jaraguá",
    "Jaraguá": "Pirituba/Jaraguá",
    "São Domingos": "Pirituba/Jaraguá",
    "Mandaqui": "Santana/Tucuruvi",
    "Santana": "Santana/Tucuruvi",
    "Tucuruvi": "Santana/Tucuruvi",
    "Campo Belo": "Santo Amaro",
    "Campo Grande": "Santo Amaro",
    "Santo Amaro": "Santo Amaro",
    "São Mateus": "São Mateus",
    "São Rafael": "São Mateus",
    "Iguatemi": "São Mateus",
    "São Miguel": "São Miguel Paulista",
    "Jardim Helena": "São Miguel Paulista",
    "Vila Jacuí": "São Miguel Paulista",
    "Sapopemba": "Sapopemba",
    "Bela Vista": "Sé",
    "Bom Retiro": "Sé",
    "Cambuci": "Sé",
    "Consolação": "Sé",
    "Liberdade": "Sé",
    "República": "Sé",
    "Santa Cecília": "Sé",
    "Sé": "Sé",
    "Vila Maria": "Vila Maria/Vila Guilherme",
    "Vila Guilherme": "Vila Maria/Vila Guilherme",
    "Vila Medeiros": "Vila Maria/Vila Guilherme",
    "Vila Mariana": "Vila Mariana",
    "Moema": "Vila Mariana",
    "Saúde": "Vila Mariana",
    "Vila Prudente": "Vila Prudente",
    "São Lucas": "Vila Prudente",
}


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


def assign_nodes_to_zones(
    nodes: gpd.GeoDataFrame, zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatial join nodes to OD zones to assign zone and district IDs.

    Args:
        nodes: GeoDataFrame with node geometries and parameters.
        zones: GeoDataFrame with OD zone polygons and district info.

    Returns:
        GeoDataFrame of nodes with zone, district, and subprefeitura assignment.
    """
    logging.info("Assigning nodes to zones via spatial join...")
    zones_proj = zones.to_crs(nodes.crs)
    join_cols = ["NumeroZona", "NomeZona", "NumDistrit", "NomeDistri", "geometry"]
    joined = gpd.sjoin(nodes, zones_proj[join_cols], predicate="within")
    joined = joined.drop(columns=["index_right"])
    joined["Subprefeitura"] = joined["NomeDistri"].map(DISTRICT_TO_SUBPREFEITURA)
    logging.info(f"Nodes assigned to zones: {len(joined)} (of {len(nodes)})")
    return joined


def assign_edges_to_zones(
    edges: gpd.GeoDataFrame, zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatial join edges to OD zones using representative points.

    Args:
        edges: GeoDataFrame with edge geometries and parameters.
        zones: GeoDataFrame with OD zone polygons and district info.

    Returns:
        GeoDataFrame of edges with zone, district, and subprefeitura assignment.
    """
    logging.info("Assigning edges to zones via spatial join (representative point)...")
    zones_proj = zones.to_crs(edges.crs)

    edge_points = edges.copy()
    edge_points["geometry"] = edge_points.geometry.representative_point()

    join_cols = ["NumeroZona", "NomeZona", "NumDistrit", "NomeDistri", "geometry"]
    joined = gpd.sjoin(edge_points, zones_proj[join_cols], predicate="within")
    joined = joined.drop(columns=["index_right"])
    joined["Subprefeitura"] = joined["NomeDistri"].map(DISTRICT_TO_SUBPREFEITURA)
    logging.info(f"Edges assigned to zones: {len(joined)} (of {len(edges)})")
    return joined


def aggregate_node_params(
    gdf: gpd.GeoDataFrame, group_col: str
) -> pd.DataFrame:
    """Aggregate node parameters by a grouping column.

    Args:
        gdf: GeoDataFrame with nodes and group assignment.
        group_col: Column name to group by.

    Returns:
        DataFrame with mean, median, max of k_i, c_i, b_i per group.
    """
    logging.info(f"Aggregating node parameters by {group_col}...")
    params = ["k_i", "c_i", "b_i"]
    agg_funcs = {col: ["mean", "median", "max"] for col in params}

    result = gdf.groupby(group_col).agg(agg_funcs)
    result.columns = [f"{col}_{func}" for col, func in result.columns]
    result = result.reset_index()
    logging.info(f"Node aggregation completed for {len(result)} groups")
    return result


def aggregate_edge_params(
    gdf: gpd.GeoDataFrame, group_col: str
) -> pd.DataFrame:
    """Aggregate edge parameters by a grouping column.

    Args:
        gdf: GeoDataFrame with edges and group assignment.
        group_col: Column name to group by.

    Returns:
        DataFrame with mean, median, max of e_ij per group.
    """
    logging.info(f"Aggregating edge parameters by {group_col}...")
    agg_funcs = {"e_ij": ["mean", "median", "max"]}

    result = gdf.groupby(group_col).agg(agg_funcs)
    result.columns = [f"{col}_{func}" for col, func in result.columns]
    result = result.reset_index()
    logging.info(f"Edge aggregation completed for {len(result)} groups")
    return result


def _filter_polygon_geometry(geom):
    """Extract polygon parts from a geometry, filtering out non-polygon types."""
    if geom.geom_type == "GeometryCollection":
        return shapely.ops.unary_union(
            [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        )
    return geom


def build_zone_geodataframe(
    zones: gpd.GeoDataFrame,
    node_stats: pd.DataFrame,
    edge_stats: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Merge aggregated statistics with OD zone polygons.

    Args:
        zones: GeoDataFrame with OD zone polygons.
        node_stats: DataFrame with aggregated node parameters per zone.
        edge_stats: DataFrame with aggregated edge parameters per zone.

    Returns:
        GeoDataFrame with zone polygons and summary statistics.
    """
    logging.info("Building zone-level summary...")
    zone_gdf = zones[["NumeroZona", "NomeZona", "NomeDistri", "geometry"]].copy()
    zone_gdf = zone_gdf.merge(node_stats, on="NumeroZona", how="left")
    zone_gdf = zone_gdf.merge(edge_stats, on="NumeroZona", how="left")
    logging.info(f"Zone summary built: {len(zone_gdf)} zones")
    return zone_gdf


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
    districts["geometry"] = districts.geometry.apply(_filter_polygon_geometry)
    logging.info(f"Districts created: {len(districts)}")

    districts = districts.merge(node_stats, on="NumDistrit", how="left")
    districts = districts.merge(edge_stats, on="NumDistrit", how="left")

    return districts


def build_subprefeitura_geodataframe(
    zones: gpd.GeoDataFrame,
    node_stats: pd.DataFrame,
    edge_stats: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Dissolve zones into subprefeituras and merge aggregated statistics.

    Args:
        zones: GeoDataFrame with OD zone polygons.
        node_stats: DataFrame with aggregated node parameters per subprefeitura.
        edge_stats: DataFrame with aggregated edge parameters per subprefeitura.

    Returns:
        GeoDataFrame with subprefeitura polygons and summary statistics.
    """
    logging.info("Dissolving OD zones into subprefeitura polygons...")
    zones = zones.copy()
    zones["geometry"] = zones.geometry.make_valid()
    zones["Subprefeitura"] = zones["NomeDistri"].map(DISTRICT_TO_SUBPREFEITURA)

    subprefs = zones.dissolve(by="Subprefeitura", as_index=False)[
        ["Subprefeitura", "geometry"]
    ]
    subprefs["geometry"] = subprefs.geometry.apply(_filter_polygon_geometry)
    logging.info(f"Subprefeituras created: {len(subprefs)}")

    subprefs = subprefs.merge(node_stats, on="Subprefeitura", how="left")
    subprefs = subprefs.merge(edge_stats, on="Subprefeitura", how="left")

    return subprefs


def main():
    """Aggregate network parameters by zone, district, and subprefeitura."""
    setup_logging()

    output_dir = Path("data/output")
    nodes, edges, zones = load_data(output_dir)

    nodes_joined = assign_nodes_to_zones(nodes, zones)
    edges_joined = assign_edges_to_zones(edges, zones)

    # Zone-level aggregation
    zone_node_stats = aggregate_node_params(nodes_joined, "NumeroZona")
    zone_edge_stats = aggregate_edge_params(edges_joined, "NumeroZona")
    zone_gdf = build_zone_geodataframe(zones, zone_node_stats, zone_edge_stats)

    zone_path = output_dir / "zone_summary.gpkg"
    zone_gdf.to_file(zone_path, driver="GPKG")
    logging.info(f"Zone summary saved to {zone_path}")

    # District-level aggregation
    district_node_stats = aggregate_node_params(nodes_joined, "NumDistrit")
    district_edge_stats = aggregate_edge_params(edges_joined, "NumDistrit")
    district_gdf = build_district_geodataframe(
        zones, district_node_stats, district_edge_stats
    )

    district_path = output_dir / "district_summary.gpkg"
    district_gdf.to_file(district_path, driver="GPKG")
    logging.info(f"District summary saved to {district_path}")

    # Subprefeitura-level aggregation
    subpref_node_stats = aggregate_node_params(nodes_joined, "Subprefeitura")
    subpref_edge_stats = aggregate_edge_params(edges_joined, "Subprefeitura")
    subpref_gdf = build_subprefeitura_geodataframe(
        zones, subpref_node_stats, subpref_edge_stats
    )

    subpref_path = output_dir / "subprefeitura_summary.gpkg"
    subpref_gdf.to_file(subpref_path, driver="GPKG")
    logging.info(f"Subprefeitura summary saved to {subpref_path}")

    logging.info(
        f"Aggregation complete: {len(zone_gdf)} zones, "
        f"{len(district_gdf)} districts, {len(subpref_gdf)} subprefeituras"
    )


if __name__ == "__main__":
    main()
