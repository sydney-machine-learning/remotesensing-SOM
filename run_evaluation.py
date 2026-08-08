
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.metrics import contingency_matrix, external_metrics

METHODS = {"SOM (9 clusters)": "som", "k-means + PCA": "pca",
           "k-means + CA": "ca", "k-means + SA": "sa"}
VALID_CLASSES = list(config.CLASS_NAMES.keys())


def main():
    if not config.GEOLOGY_REF_TIF.exists():
        print(f"ERROR: ground-truth raster not found:\n  {config.GEOLOGY_REF_TIF}\n"
              f"Place 'geology_reference_from_polygons.tif' in the repo root, then re-run.")
        sys.exit(1)

    import rioxarray as rxr
    geology = rxr.open_rasterio(str(config.GEOLOGY_REF_TIF)).squeeze()
    extent = [float(geology.x.min()), float(geology.x.max()),
              float(geology.y.min()), float(geology.y.max())]
    ref = geology.values.flatten()
    config.FIGURES_DIR.mkdir(exist_ok=True)

    maps, rows = {}, {}
    for label, key in METHODS.items():
        p = config.OUTPUTS_DIR / f"{key}_labels_spatial.npy"
        if not p.exists():
            print(f"  MISSING {p.name} - run run_all_methods.py first"); continue
        m = np.load(p); maps[label] = m
        pred = m.flatten()
        mask = ~np.isnan(pred) & np.isin(ref, VALID_CLASSES)
        ext = external_metrics(ref[mask].astype(int), pred[mask].astype(int))
        ij = config.OUTPUTS_DIR / f"{key}_intrinsic.json"
        if ij.exists():
            ext.update(json.load(open(ij)))
        rows[label] = ext
        print(f"{label:20s}  OA={ext['Overall Accuracy']:.3f}  "
              f"Balanced={ext['Balanced Accuracy']:.3f}  Kappa={ext['Kappa']:.3f}  "
              f"Kanimblan recall={ext['Kanimblan Recall']:.3f}")

    if not rows:
        print("No method outputs found."); sys.exit(1)

    show = ["Overall Accuracy", "Balanced Accuracy", "Kappa", "Kanimblan Recall",
            "ARI", "NMI", "Purity", "Homogeneity", "Completeness", "V-measure",
            "Silhouette", "CH Index", "DB Index", "n_clusters"]
    df = pd.DataFrame(rows).T
    df = df[[c for c in show if c in df.columns]].astype(float).round(4)
    df.to_csv(config.FIGURES_DIR / "model_comparison_results.csv")
    print("\n" + df.to_string())
    print(f"\nSaved table -> {config.FIGURES_DIR / 'model_comparison_results.csv'}")

  
    hm_cols = ["Overall Accuracy", "Balanced Accuracy", "Kappa", "Kanimblan Recall",
               "ARI", "NMI", "Purity", "V-measure"]
    hm = df[[c for c in hm_cols if c in df.columns]]
    plt.figure(figsize=(11, 4))
    sns.heatmap(hm, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=1, linecolor="white", cbar_kws={"label": "Score"})
    plt.title("Model comparison (higher = better)"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "model_comparison_heatmap.png", dpi=200)

  
    n = len(maps) + 1; ncol = 3; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 6 * nrow)); axes = axes.ravel()
    im = axes[0].imshow(geology.values, extent=extent, cmap="tab10", vmin=1, vmax=3,
                        interpolation="nearest")
    axes[0].set_title("Reference geology (Kanimblan/GAB/CSP)")
    cb = plt.colorbar(im, ax=axes[0], ticks=[1, 2, 3], fraction=0.046)
    cb.set_ticklabels(["Kanimblan", "GAB", "CSP"])
    for ax, (label, m) in zip(axes[1:], maps.items()):
        nc = int(np.nanmax(m)) + 1
        cmap = plt.get_cmap("tab10", nc).copy(); cmap.set_bad("white")
        ax.imshow(m, extent=extent, cmap=cmap, interpolation="nearest")
        ax.set_title(f"{label}\nOA={rows[label]['Overall Accuracy']:.3f} | "
                     f"Kappa={rows[label]['Kappa']:.3f}")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "model_comparison_maps.png", dpi=200)

   
    fig, axes = plt.subplots(2, 2, figsize=(14, 11)); axes = axes.ravel()
    for ax, (label, m) in zip(axes, maps.items()):
        pred = m.flatten(); mask = ~np.isnan(pred) & np.isin(ref, VALID_CLASSES)
        yt = ref[mask].astype(int); yp = pred[mask].astype(int)
        cm, classes, clusters = contingency_matrix(yt, yp)
        cmn = cm / cm.sum(axis=1, keepdims=True)   # row-normalise by true class
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1,
                    yticklabels=[config.CLASS_NAMES[c] for c in classes],
                    xticklabels=clusters)
        ax.set_title(label); ax.set_xlabel("Predicted cluster"); ax.set_ylabel("True unit")
        ax.tick_params(axis="y", rotation=0)
    for ax in axes[len(maps):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "confusion_matrices.png", dpi=200)
    print(f"Saved figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
