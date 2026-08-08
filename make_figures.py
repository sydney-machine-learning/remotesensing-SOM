import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src import config
from src import plotting as P

FIG = config.FIGURES_DIR
OUT = config.OUTPUTS_DIR
METHODS = {"SOM": "som", "k-means + PCA": "pca", "k-means + CA": "ca",
           "k-means + SA": "sa"}


def _ref():
    if not config.GEOLOGY_REF_TIF.exists():
        return None, None
    import rioxarray as rxr
    g = rxr.open_rasterio(str(config.GEOLOGY_REF_TIF)).squeeze()
    extent = [float(g.x.min()), float(g.x.max()), float(g.y.min()), float(g.y.max())]
    return g.values, extent


def fig_geology_reference(ref, extent):
    """Paper Fig 6 - the simplified geological reference."""
    a = np.where(ref == -9999, np.nan, ref).astype(float)
    fig, ax = plt.subplots(figsize=(P.COL1 * 1.4, P.COL1 * 1.4))
    P.plot_class_map(ax, a, extent, P.CLASS3,
                     ["Kanimblan (basement)", "GAB", "CSP"],
                     title="Geological reference (NSW Seamless Geology)", start_at=1)
    P.save(fig, FIG / "fig_geology_reference.png")
    print("  fig_geology_reference.png")


def fig_pca_variance():
    """Paper Fig 8 - WITHOUT the dual axis."""
    fp = OUT / "pca_explained_variance.npy"
    if not fp.exists():
        print("  (skip PCA variance: run run_all_methods.py)"); return
    evr = np.load(fp)
    cum = np.cumsum(evr) * 100
    x = np.arange(1, len(evr) + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(P.COL1, 3.2), sharex=True,
                                 constrained_layout=True)
    a1.bar(x, evr * 100, color=P.CLUSTER4[0], width=0.65)
    a1.set_ylabel("Individual (%)")
    a1.grid(axis="y", alpha=0.5); a1.set_axisbelow(True)
    a2.plot(x, cum, "o-", color=P.CLUSTER4[1], markersize=4)
    a2.axhline(95, color=P.MUTED, ls="--", lw=0.8)
    a2.annotate("95%", (x[-1], 95), textcoords="offset points", xytext=(-18, 4),
                fontsize=6, color=P.MUTED)
    a2.set_ylabel("Cumulative (%)"); a2.set_xlabel("Principal component")
    a2.set_xticks(x); a2.grid(axis="y", alpha=0.5); a2.set_axisbelow(True)
    a2.annotate(f"{cum[-1]:.2f}%", (x[-1], cum[-1]), textcoords="offset points",
                xytext=(-26, -10), fontsize=6, color=P.CLUSTER4[1])
    a1.set_title("PCA explained variance")
    P.save(fig, FIG / "fig_pca_variance.png")
    print("  fig_pca_variance.png  (dual axis removed)")


def fig_method_maps(ref, extent):
    """Paper Fig 14 - side-by-side classification maps."""
    have = {k: v for k, v in METHODS.items()
            if (OUT / f"{v}_labels_spatial.npy").exists()}
    if not have:
        print("  (skip maps: no outputs)"); return
    n = len(have) + (1 if ref is not None else 0)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    # Panels are square maps: sizing height to the grid keeps them from being
    # stranded in a band of whitespace.
    fig, axes = plt.subplots(nrows, ncols, figsize=(P.COL2, 2.9 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    i = 0
    if ref is not None:
        a = np.where(ref == -9999, np.nan, ref).astype(float)
        P.plot_class_map(axes[0], a, extent, P.CLASS3,
                         ["Kanimblan", "GAB", "CSP"], title="Reference geology",
                         start_at=1, ticks=False)
        i = 1
    for name, key in have.items():
        m = np.load(OUT / f"{key}_labels_spatial.npy")
        nc = int(np.nanmax(m)) + 1
        colors = P.CLUSTER4 if nc <= 4 else P.SOM9[:nc]
        P.plot_class_map(axes[i], m, extent, colors,
                         [f"C{c}" for c in range(nc)],
                         title=f"{name} ({nc} clusters)", ticks=False)
        i += 1
    for ax in axes[i:]:
        ax.axis("off")
    P.save(fig, FIG / "fig_method_maps.png")
    print("  fig_method_maps.png")


def fig_kanimblan_binary(ref, extent):

    if ref is None:
        print("  (skip Kanimblan binary: no reference)"); return
    from src.metrics import apply_mapping, map_clusters_to_classes
    have = {k: v for k, v in METHODS.items()
            if (OUT / f"{v}_labels_spatial.npy").exists()}
    if not have:
        return
    flat_ref = ref.flatten()
    valid_classes = list(config.CLASS_NAMES.keys())
  
    recalls = {}
    rp = FIG / "model_comparison_results.csv"
    if rp.exists():
        df = pd.read_csv(rp, index_col=0)
        for idx in df.index:
            recalls[idx.split(" (")[0]] = df.loc[idx, "Kanimblan Recall"]


    fig, axes = plt.subplots(2, 3, figsize=(P.COL2, 5.0), constrained_layout=True)
    axes = axes.ravel()


    any_map = np.load(OUT / f"{list(have.values())[0]}_labels_spatial.npy")
    footprint = ~np.isnan(any_map)
    truth = np.where((ref == -9999) | (~footprint), np.nan,
                     (ref == config.TARGET_CLASS).astype(float))
    P.plot_class_map(axes[0], truth, extent, ["#e8e8e4", P.CLASS3[0]],
                     ["other", "Kanimblan"], title="Reference\n(within EnMAP swath)",
                     legend=False, ticks=False)
    for i, (name, key) in enumerate(have.items(), start=1):
        m = np.load(OUT / f"{key}_labels_spatial.npy")
        pred = m.flatten()
        mask = ~np.isnan(pred) & np.isin(flat_ref, valid_classes)
        mp = map_clusters_to_classes(flat_ref[mask].astype(int), pred[mask].astype(int))
        mapped = np.full(pred.shape, np.nan)
        ok = ~np.isnan(pred)
        mapped[ok] = apply_mapping(pred[ok].astype(int), mp)
        binary = np.where(np.isnan(mapped), np.nan,
                          (mapped == config.TARGET_CLASS).astype(float)
                          ).reshape(m.shape)
        r = recalls.get(name)
        title = f"{name}\nrecall = {r:.3f}" if r is not None else name
        P.plot_class_map(axes[i], binary, extent, ["#e8e8e4", P.CLASS3[0]],
                         ["other", "Kanimblan"], title=title,
                         legend=False, ticks=False)
    for ax in axes[len(have) + 1:]:
        ax.axis("off")
    P.class_legend(fig, ["#e8e8e4", P.CLASS3[0]], ["other", "Kanimblan"])
    P.save(fig, FIG / "fig_kanimblan_binary.png")
    print("  fig_kanimblan_binary.png  (NEW - the paper's actual objective)")


def fig_metrics(results_csv):
    """Paper Fig 15 - metrics, with scales separated."""
    if not results_csv.exists():
        print("  (skip metrics: run run_evaluation.py)"); return
    df = pd.read_csv(results_csv, index_col=0)
    unit = [c for c in ["Overall Accuracy", "Balanced Accuracy", "Kappa",
                        "Kanimblan Recall", "ARI", "NMI", "Purity", "V-measure"]
            if c in df.columns]
    fig, ax = plt.subplots(figsize=(P.COL2, 0.5 * len(df) + 1.4))
    import seaborn as sns
    sns.heatmap(df[unit], annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1,
                linewidths=1.2, linecolor="white", ax=ax,
                cbar_kws={"label": "score (higher = better)"},
                annot_kws={"fontsize": 7})
    ax.set_title("Agreement with geological reference")
    plt.yticks(rotation=0)
    P.save(fig, FIG / "fig_metrics_unit_scale.png")
    print("  fig_metrics_unit_scale.png  (CH/DB removed - different scale)")

    other = [c for c in ["Silhouette", "CH Index", "DB Index"] if c in df.columns]
    if other:
        fig, axes = plt.subplots(1, len(other), figsize=(P.COL2, 2.2),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, c in zip(axes, other):
            P.bar_with_error(ax, df.index.tolist(), df[c].values,
                             np.zeros(len(df)), color=P.CLUSTER4[0], ylabel=c)
            ax.set_title(c)
        fig.suptitle("Cluster-quality metrics (each on its own scale)", fontsize=9)
        P.save(fig, FIG / "fig_metrics_cluster_quality.png")
        print("  fig_metrics_cluster_quality.png")


def fig_confusion(ref):
  
    if ref is None:
        return
    from src.metrics import contingency_matrix
    import seaborn as sns
    have = {k: v for k, v in METHODS.items()
            if (OUT / f"{v}_labels_spatial.npy").exists()}
    if not have:
        return
    flat = ref.flatten()
    valid_classes = list(config.CLASS_NAMES.keys())
    fig, axes = plt.subplots(2, 2, figsize=(P.COL2, 5.4), constrained_layout=True)
    axes = axes.ravel()
    for ax, (name, key) in zip(axes, have.items()):
        m = np.load(OUT / f"{key}_labels_spatial.npy")
        pred = m.flatten()
        mask = ~np.isnan(pred) & np.isin(flat, valid_classes)
        yt, yp = flat[mask].astype(int), pred[mask].astype(int)
        cm, classes, clusters = contingency_matrix(yt, yp)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1,
                    # exactly one label per row, and rows are true classes only
                    yticklabels=[config.CLASS_NAMES[c] for c in classes],
                    xticklabels=clusters,
                    cbar_kws={"label": "proportion"}, annot_kws={"fontsize": 6})
        ax.set_title(name); ax.set_xlabel("Predicted cluster")
        ax.set_ylabel("True unit")
        ax.tick_params(axis="y", rotation=0)
    for ax in axes[len(have):]:
        ax.axis("off")
    P.save(fig, FIG / "fig_confusion_matrices.png")
    print("  fig_confusion_matrices.png  (y-label off-by-one fixed)")


def fig_som_training_fix():
    
    fp = FIG / "som_training_fix.csv"
    if not fp.exists():
        return
    df = pd.read_csv(fp, index_col=0)
    cols = [c for c in ["Topographic Error", "Kanimblan Recall"] if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(P.COL2, 2.6),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        colors = [P.SOM9[7] if "PAPER" in i else P.CLUSTER4[1] for i in df.index]
        ax.bar(np.arange(len(df)), df[c].values, color=colors, width=0.6)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df.index, rotation=25, ha="right", fontsize=6)
        ax.set_ylabel(c)
        ax.set_title(c + ("  (lower = better)" if "Error" in c else "  (higher = better)"))
        ax.grid(axis="y", alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Effect of the SOM training fix (red = paper's setting)", fontsize=9)
    P.save(fig, FIG / "fig_som_training_fix.png")
    print("  fig_som_training_fix.png  (NEW)")


def fig_multiseed():
  
    fp = FIG / "multiseed_results.csv"
    if not fp.exists():
        return
    df = pd.read_csv(fp, index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(P.COL2, 2.6), constrained_layout=True)
    if "Silhouette mean" in df.columns:
        P.bar_with_error(axes[0], df.index.tolist(), df["Silhouette mean"],
                         df.get("Silhouette std", pd.Series(np.zeros(len(df)))),
                         color=P.CLUSTER4[0], ylabel="Silhouette (219-D)")
        axes[0].set_title("Cluster quality (mean ± std over seeds)")
    if "Stability (mean pairwise ARI)" in df.columns:
        P.bar_with_error(axes[1], df.index.tolist(),
                         df["Stability (mean pairwise ARI)"],
                         df.get("Stability std", pd.Series(np.zeros(len(df)))),
                         color=P.CLUSTER4[1], ylabel="mean pairwise ARI")
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title("Stability across seeds (1.0 = identical map)")
    P.save(fig, FIG / "fig_multiseed.png")
    print("  fig_multiseed.png  (NEW)")


def fig_som_smallmultiples(extent):
  
    fp = OUT / "som_labels_spatial.npy"
    if not fp.exists():
        return
    m = np.load(fp)
    nc = int(np.nanmax(m)) + 1
    fig, _ = P.plot_cluster_smallmultiples(m, extent, nc,
                                           title="SOM clusters (one panel each)")
    P.save(fig, FIG / "fig_som_clusters_facet.png")
    print("  fig_som_clusters_facet.png  (NEW - CVD-safe alternative to 9 hues)")


def main():
    P.use_paper_style()
    FIG.mkdir(exist_ok=True)
    ref, extent = _ref()
    print("Generating figures ->", FIG)
    if ref is not None:
        fig_geology_reference(ref, extent)
    fig_pca_variance()
    fig_method_maps(ref, extent)
    fig_kanimblan_binary(ref, extent)
    fig_confusion(ref)
    fig_metrics(FIG / "model_comparison_results.csv")
    fig_som_training_fix()
    fig_multiseed()
    if extent is not None:
        fig_som_smallmultiples(extent)
    print("\nDone. PNG (300 dpi) + PDF (vector) written for each figure.")


if __name__ == "__main__":
    main()
