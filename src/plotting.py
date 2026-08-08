from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --- validated categorical palettes ---------------------------------------
CLASS3 = ["#2a78d6", "#eda100", "#008300"]              # Kanimblan, GAB, CSP
CLUSTER4 = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]  # validated all-pairs
# 9 hues for the side-by-side SOM panel only; NOT all-pairs CVD-safe (no 9-hue
# set is) -> must ship direct labels / a legend, never colour alone.
SOM9 = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a",
        "#eb6834", "#4a3aa7", "#e34948", "#7f7f7f"]

# Sequential single hue (light -> dark), for magnitude (e.g. confusion matrices).
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
            "#256abf", "#184f95", "#0d366b"]

# Chart chrome
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
NODATA = "#f0efec"

# Elsevier-style column widths (inches)
COL1 = 3.54   # 90 mm single column
COL2 = 7.48   # 190 mm double column


def use_paper_style():
    """Apply a consistent, journal-appropriate matplotlib style."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
    })


def discrete_cmap(colors: Sequence[str], nodata=NODATA) -> ListedColormap:
    cmap = ListedColormap(list(colors))
    cmap = cmap.copy()
    cmap.set_bad(nodata)
    return cmap


def plot_class_map(ax, arr: np.ndarray, extent, colors: Sequence[str],
                   labels: Sequence[str], title: str = "", start_at: int = 0,
                   legend: bool = True, ticks: bool = True):
    a = np.asarray(arr, dtype=float).copy()
    a = a - start_at
    n = len(colors)
    cmap = discrete_cmap(colors)
    im = ax.imshow(a, cmap=cmap, extent=extent, interpolation="nearest",
                   vmin=-0.5, vmax=n - 0.5)
    ax.set_title(title)
    if ticks:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="plain", axis="both")
        # Few ticks: full UTM coordinates are long and collide otherwise.
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    if legend:
        handles = [Patch(facecolor=colors[i], label=labels[i]) for i in range(n)]
        ax.legend(handles=handles, loc="upper right", framealpha=0.9,
                  frameon=True, fontsize=6, ncol=1)
    return im


def class_legend(fig, colors: Sequence[str], labels: Sequence[str], ncol=None):
    """One shared legend for a grid of small-multiple maps."""
    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower center", frameon=False,
               ncol=ncol or len(labels), fontsize=7,
               bbox_to_anchor=(0.5, -0.02))


def plot_cluster_smallmultiples(arr: np.ndarray, extent, n_clusters: int,
                                title: str = "", ncols: int = 3,
                                color: str = "#2a78d6"):
 
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL2, 2.4 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    valid = ~np.isnan(arr)
    for c in range(n_clusters):
        ax = axes[c]
        panel = np.full(arr.shape, np.nan)
        panel[valid] = 0.0
        panel[valid & (arr == c)] = 1.0
        cmap = discrete_cmap(["#e8e8e4", color])
        ax.imshow(panel, cmap=cmap, extent=extent, interpolation="nearest",
                  vmin=-0.5, vmax=1.5)
        pct = 100 * np.nansum(arr == c) / max(np.nansum(valid), 1)
        ax.set_title(f"Cluster {c} ({pct:.1f}%)")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n_clusters:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig, axes


def bar_with_error(ax, labels, means, stds, color="#2a78d6", ylabel=""):
    """Bar chart with error bars; values direct-labelled selectively."""
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=color, width=0.6,
           error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    return ax


def save(fig, path, also_pdf: bool = True):
    """Save PNG at 300 dpi plus a vector PDF (journals want vector)."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if also_pdf:
        fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
