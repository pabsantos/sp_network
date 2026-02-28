import logging
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
PLOT_DIR = PROJECT_ROOT / "plot"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

DARK_BG = "#1a1a2e"


def setup_logging():
    """Configure logging with INFO level and console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_dark_theme():
    """Set matplotlib rcParams for dark-themed plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_BG,
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        }
    )


def load_nodes() -> gpd.GeoDataFrame:
    """Load nodes GeoPackage and reproject to Web Mercator for basemap."""
    logging.info("Loading nodes...")
    nodes = gpd.read_file(OUTPUT_DIR / "nodes.gpkg")
    nodes = nodes.to_crs(epsg=3857)
    logging.info(f"Nodes loaded: {len(nodes)}")
    return nodes


def plot_node_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
    figsize: tuple = (12, 10),
    markersize: float = 0.15,
):
    """Create a point map of nodes colored by a parameter.

    Args:
        gdf: GeoDataFrame with node geometries in EPSG:3857.
        column: Column name to use for coloring.
        title: Plot title.
        output_path: Path to save the PNG file.
        figsize: Figure size in inches.
        markersize: Size of node markers.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    gdf.plot(
        column=column,
        ax=ax,
        cmap="plasma",
        markersize=markersize,
        alpha=0.6,
        legend=True,
        legend_kwds={"shrink": 0.6, "label": column},
    )

    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter)

    ax.set_title(title, fontsize=14, color="white", pad=12)
    ax.set_axis_off()

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def main():
    """Generate node parameter maps for the paper."""
    setup_logging()
    setup_dark_theme()

    PLOT_DIR.mkdir(exist_ok=True)

    nodes = load_nodes()

    maps = [
        ("k_i", "Node Degree ($k_i$)"),
        ("c_i", "Clustering Coefficient ($c_i$)"),
        ("l_i", "Average Shortest Path Length ($l_i$)"),
    ]

    for column, title in maps:
        output_path = PLOT_DIR / f"nodes_{column}.png"
        plot_node_map(nodes, column, title, output_path)

    logging.info("All maps generated")


if __name__ == "__main__":
    main()
