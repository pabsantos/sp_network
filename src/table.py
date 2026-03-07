import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


def setup_logging():
    """Configure logging with INFO level and console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def distribution_table(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> str:
    """Generate LaTeX table with distribution statistics of network parameters.

    Args:
        nodes: GeoDataFrame with node parameters.
        edges: GeoDataFrame with edge parameters.

    Returns:
        LaTeX table string.
    """
    params = [
        ("$k_i$", nodes["k_i"].values),
        ("$c_i$", nodes["c_i"].values),
        ("$l_i$", nodes["l_i"].values),
        ("$e_{ij}$", edges["e_ij"].values),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Distribution of network parameters.}",
        r"\label{tab:distribution}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Parameter & Mean & Median & Std & Min & Max \\",
        r"\hline",
    ]

    for name, values in params:
        lines.append(
            f"{name} & {values.mean():.4f} & {np.median(values):.4f} & "
            f"{values.std():.4f} & {values.min():.4f} & {values.max():.4f} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def comparison_table(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> str:
    """Generate LaTeX table comparing real graph vs random graph parameters.

    Args:
        nodes: GeoDataFrame with node parameters.
        edges: GeoDataFrame with edge parameters.

    Returns:
        LaTeX table string.
    """
    N = len(nodes)
    L = len(edges)
    avg_k = nodes["k_i"].mean()
    avg_c = nodes["c_i"].mean()
    avg_l = nodes["l_i"].mean()

    p_random = (2 * L) / (N * (N - 1)) if N > 1 else 0
    k_star = p_random * (N - 1)
    c_star = k_star / N if N > 0 else 0
    l_star = np.log(N) / np.log(k_star) if k_star > 1 else None

    l_star_str = f"{l_star:.4f}" if l_star is not None else "N/A"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Comparison between the real network and the theoretical random graph $G(N, p)$.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Network & $N$ & $L$ & $p$ & $\langle k \rangle$ & $\langle c \rangle$ & $\langle l \rangle$ \\",
        r"\hline",
        f"Real network & {N} & {L} & -- & {avg_k:.4f} & {avg_c:.6f} & {avg_l:.4f} \\\\",
        f"$G(N, p)$ & {N} & -- & {p_random:.6f} & {k_star:.4f} & {c_star:.6f} & {l_star_str} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    """Print LaTeX tables for network parameters."""
    setup_logging()

    logging.info("Loading data...")
    nodes = gpd.read_file(OUTPUT_DIR / "nodes.gpkg")
    edges = gpd.read_file(OUTPUT_DIR / "edges.gpkg")

    print("\n% === Distribution Table ===\n")
    print(distribution_table(nodes, edges))

    print("\n% === Comparison Table ===\n")
    print(comparison_table(nodes, edges))


if __name__ == "__main__":
    main()
