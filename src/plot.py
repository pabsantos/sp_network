import logging
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
PLOT_DIR = PROJECT_ROOT / "plot"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

BG_COLOR = "white"


def setup_logging():
    """Configure logging with INFO level and console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_theme():
    """Set matplotlib rcParams for white-background plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def load_nodes() -> gpd.GeoDataFrame:
    """Load nodes GeoPackage and reproject to Web Mercator for basemap."""
    logging.info("Loading nodes...")
    nodes = gpd.read_file(OUTPUT_DIR / "nodes.gpkg")
    nodes = nodes.to_crs(epsg=3857)
    logging.info(f"Nodes loaded: {len(nodes)}")
    return nodes


def load_edges() -> gpd.GeoDataFrame:
    """Load edges GeoPackage and reproject to Web Mercator for basemap."""
    logging.info("Loading edges...")
    edges = gpd.read_file(OUTPUT_DIR / "edges.gpkg")
    edges = edges.to_crs(epsg=3857)
    logging.info(f"Edges loaded: {len(edges)}")
    return edges


def plot_node_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    label: str,
    output_path: Path,
    figsize: tuple = (12, 10),
    markersize: float = 0.15,
):
    """Create a point map of nodes colored by a parameter.

    Args:
        gdf: GeoDataFrame with node geometries in EPSG:3857.
        column: Column name to use for coloring.
        label: LaTeX label for the colorbar legend.
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
        legend_kwds={"shrink": 0.6, "label": label},
    )

    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter)

    ax.set_axis_off()

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def plot_edge_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    label: str,
    output_path: Path,
    figsize: tuple = (12, 10),
    linewidth: float = 0.15,
    log_scale: bool = False,
):
    """Create a line map of edges colored by a parameter.

    Args:
        gdf: GeoDataFrame with edge geometries in EPSG:3857.
        column: Column name to use for coloring.
        label: LaTeX label for the colorbar legend.
        output_path: Path to save the PNG file.
        figsize: Figure size in inches.
        linewidth: Width of edge lines.
        log_scale: Use logarithmic color normalization.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    plot_kwargs = {}
    if log_scale:
        vmin = gdf[column][gdf[column] > 0].min()
        vmax = gdf[column].max()
        plot_kwargs["norm"] = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    gdf.plot(
        column=column,
        ax=ax,
        cmap="plasma",
        linewidth=linewidth,
        alpha=0.6,
        legend=True,
        legend_kwds={"shrink": 0.6, "label": label},
        **plot_kwargs,
    )

    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter)

    ax.set_axis_off()

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def plot_degree_distribution(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    figsize: tuple = (8, 5),
):
    """Create a log-log scatter plot of the degree distribution P(k).

    Args:
        gdf: GeoDataFrame with node data containing k_i column.
        output_path: Path to save the PNG file.
        figsize: Figure size in inches.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    k_values = gdf["k_i"].values
    unique, counts = np.unique(k_values, return_counts=True)
    proportion = counts / counts.sum()

    ax.scatter(unique, proportion, color="blue", s=30, alpha=0.8, edgecolors="none")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k_i$", fontsize=12)
    ax.set_ylabel(r"$P(k_i)$", fontsize=12)
    ax.grid(True, which="both", alpha=0.2, color="gray", linestyle="-")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def plot_histogram(
    values: np.ndarray,
    xlabel: str,
    output_path: Path,
    figsize: tuple = (8, 5),
    bins: int = 50,
    log_scale: bool = False,
):
    """Create a histogram of a parameter distribution.

    Args:
        values: Array of values to plot.
        xlabel: LaTeX label for the x-axis.
        output_path: Path to save the PNG file.
        figsize: Figure size in inches.
        bins: Number of histogram bins.
        log_scale: Use logarithmic x-axis scale.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if log_scale:
        values = values[values > 0]
        bins_arr = np.logspace(np.log10(values.min()), np.log10(values.max()), bins)
        ax.set_xscale("log")
    else:
        bins_arr = bins

    ax.hist(values, bins=bins_arr, color="black", alpha=0.7, edgecolor="white")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, which="both", alpha=0.2, color="gray", linestyle="-")

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


def main():
    """Generate node and edge parameter maps for the paper."""
    setup_logging()
    setup_theme()

    PLOT_DIR.mkdir(exist_ok=True)

    nodes = load_nodes()

    node_maps = [
        ("k_i", r"$k_i$"),
        ("c_i", r"$c_i$"),
        ("l_i", r"$l_i$"),
    ]

    for column, label in node_maps:
        output_path = PLOT_DIR / f"nodes_{column}.png"
        plot_node_map(nodes, column, label, output_path)

    edges = load_edges()
    plot_edge_map(
        edges, "e_ij", r"$e_{ij}$", PLOT_DIR / "edges_e_ij.png", log_scale=True
    )

    plot_degree_distribution(nodes, PLOT_DIR / "degree_distribution.png")

    plot_histogram(nodes["c_i"].values, r"$c_i$", PLOT_DIR / "hist_c_i.png")
    plot_histogram(nodes["l_i"].values, r"$l_i$", PLOT_DIR / "hist_l_i.png")
    plot_histogram(
        edges["e_ij"].values, r"$e_{ij}$", PLOT_DIR / "hist_e_ij.png", log_scale=True
    )

    logging.info("All maps generated")


if __name__ == "__main__":
    main()
