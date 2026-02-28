import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

SP_MUNICIPALITY_CODE = "3550308"


def setup_logging():
    """Configure logging with INFO level and console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_census_data(path: Path) -> gpd.GeoDataFrame:
    """Load census tract GeoPackage with population and geometry.

    Args:
        path: Path to the IBGE census GeoPackage file.

    Returns:
        GeoDataFrame with census tract polygons and population data.
    """
    logging.info(f"Loading census data from {path}...")
    census = gpd.read_file(path)
    logging.info(f"Census tracts loaded: {len(census)}")
    logging.info(f"Columns: {list(census.columns)}")
    return census


def filter_census_by_municipality(
    census: gpd.GeoDataFrame, cd_mun: str = SP_MUNICIPALITY_CODE
) -> gpd.GeoDataFrame:
    """Filter census tracts by municipality code.

    Args:
        census: GeoDataFrame with all census tracts.
        cd_mun: Municipality code to filter by.

    Returns:
        GeoDataFrame with only tracts from the specified municipality.
    """
    logging.info(f"Filtering census tracts for municipality {cd_mun}...")
    filtered = census[census["CD_MUN"] == cd_mun].copy()
    logging.info(f"Filtered: {len(filtered)} tracts (from {len(census)} total)")
    return filtered


def create_centroids(census: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create centroid points from census tract polygons.

    Args:
        census: GeoDataFrame with census tract polygons.

    Returns:
        GeoDataFrame with centroid geometries and population values.
    """
    logging.info("Creating centroids from census tract polygons...")
    centroids = census[["CD_SETOR", "v0001", "geometry"]].copy()
    centroids["v0001"] = (
        pd.to_numeric(centroids["v0001"], errors="coerce").fillna(0).astype(int)
    )
    centroids = centroids.to_crs(epsg=31983)
    centroids["geometry"] = centroids.geometry.centroid
    centroids = centroids.to_crs(census.crs)
    logging.info(f"Centroids created: {len(centroids)}")
    logging.info(f"Total population: {centroids['v0001'].sum():,}")
    return centroids


def assign_centroids_to_areas(
    centroids: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, join_cols: list[str]
) -> gpd.GeoDataFrame:
    """Spatial join centroids to area polygons.

    Args:
        centroids: GeoDataFrame with centroid points.
        areas: GeoDataFrame with area polygons.
        join_cols: Columns to keep from the areas GeoDataFrame.

    Returns:
        GeoDataFrame with centroids assigned to areas.
    """
    areas_proj = areas.to_crs(centroids.crs)
    cols = [c for c in join_cols if c in areas_proj.columns] + ["geometry"]
    joined = gpd.sjoin(centroids, areas_proj[cols], predicate="within")
    joined = joined.drop(columns=["index_right"])
    unmatched = len(centroids) - len(joined)
    if unmatched > 0:
        logging.warning(f"Unmatched centroids: {unmatched}")
    return joined


def aggregate_population(joined: gpd.GeoDataFrame, group_col: str) -> pd.DataFrame:
    """Sum population by grouping column.

    Args:
        joined: GeoDataFrame with centroids assigned to areas.
        group_col: Column name to group by.

    Returns:
        DataFrame with total population per group.
    """
    logging.info(f"Aggregating population by {group_col}...")
    result = (
        joined.groupby(group_col)["v0001"]
        .sum()
        .reset_index()
        .rename(columns={"v0001": "populacao"})
    )
    logging.info(f"Population aggregated for {len(result)} groups")
    logging.info(f"Total population in aggregation: {result['populacao'].sum():,}")
    return result


def main():
    """Aggregate census population by OD zone and subprefeitura."""
    setup_logging()

    census_path = Path("data/raw/censo/SP_setores_CD2022.gpkg")
    output_dir = Path("data/output")

    census = load_census_data(census_path)
    census_sp = filter_census_by_municipality(census)
    centroids = create_centroids(census_sp)

    zones = gpd.read_file(output_dir / "od_zones_sp.gpkg")
    logging.info(f"OD zones loaded: {len(zones)}")

    subpref = gpd.read_file(output_dir / "subprefeitura_summary.gpkg")
    logging.info(f"Subprefeituras loaded: {len(subpref)}")

    # Assign centroids to OD zones
    logging.info("Assigning centroids to OD zones...")
    centroids_zones = assign_centroids_to_areas(
        centroids, zones, ["NumeroZona", "NomeZona", "NumDistrit", "NomeDistri"]
    )
    zone_pop = aggregate_population(centroids_zones, "NumeroZona")

    # Assign centroids to subprefeituras
    logging.info("Assigning centroids to subprefeituras...")
    centroids_subpref = assign_centroids_to_areas(centroids, subpref, ["Subprefeitura"])
    subpref_pop = aggregate_population(centroids_subpref, "Subprefeitura")

    # Merge population into zone summary
    zone_summary = gpd.read_file(output_dir / "zone_summary.gpkg")
    if "populacao" in zone_summary.columns:
        zone_summary = zone_summary.drop(columns=["populacao"])
    zone_summary = zone_summary.merge(zone_pop, on="NumeroZona", how="left")
    zone_summary["populacao"] = zone_summary["populacao"].fillna(0).astype(int)

    zone_path = output_dir / "zone_summary.gpkg"
    zone_summary.to_file(zone_path, driver="GPKG")
    logging.info(f"Zone summary updated: {zone_path}")

    # Merge population into subprefeitura summary
    subpref_summary = gpd.read_file(output_dir / "subprefeitura_summary.gpkg")
    if "populacao" in subpref_summary.columns:
        subpref_summary = subpref_summary.drop(columns=["populacao"])
    subpref_summary = subpref_summary.merge(subpref_pop, on="Subprefeitura", how="left")
    subpref_summary["populacao"] = subpref_summary["populacao"].fillna(0).astype(int)

    subpref_path = output_dir / "subprefeitura_summary.gpkg"
    subpref_summary.to_file(subpref_path, driver="GPKG")
    logging.info(f"Subprefeitura summary updated: {subpref_path}")

    logging.info(
        f"Population aggregation complete: "
        f"{len(zone_summary)} zones, {len(subpref_summary)} subprefeituras"
    )


if __name__ == "__main__":
    main()
