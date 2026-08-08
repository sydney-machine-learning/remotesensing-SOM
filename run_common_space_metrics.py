
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from src import config
from src.preprocessing import load_and_preprocess_enmap
from src.seeding import seed_everything

METHODS = {"SOM (9 clusters)": "som", "k-means + PCA": "pca",
           "k-means + CA": "ca", "k-means + SA": "sa"}

SAMPLE = 100_000
SILHOUETTE_SAMPLE = 10_000


def main():
    seed_everything()
    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid

    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(X), min(SAMPLE, len(X)), replace=False)
    Xs = X[idx]
  
    sil_idx = idx[:min(SILHOUETTE_SAMPLE, len(idx))]
    Xsil = X[sil_idx]
    print(f"Common space: {Xs.shape[1]}-D reflectance")
    print(f"  CH/DB sample: {len(idx):,} px | Silhouette sample: {len(sil_idx):,} px\n")

    rows = {}
    for label, key in METHODS.items():
        p = config.OUTPUTS_DIR / f"{key}_labels_valid.npy"
        if not p.exists():
            print(f"  MISSING {p.name}"); continue
        labels = np.load(p)
        if len(labels) != len(X):
            print(f"  {label}: label/pixel mismatch, skipping"); continue
        ls = labels[idx]
        rows[label] = {
            "n_clusters": int(len(np.unique(labels))),
            "Silhouette (219-D)": float(silhouette_score(Xsil, labels[sil_idx])),
            "CH (219-D)": float(calinski_harabasz_score(Xs, ls)),
            "DB (219-D)": float(davies_bouldin_score(Xs, ls)),
        }
        print(f"{label:20s} {rows[label]}")

    df = pd.DataFrame(rows).T
    config.FIGURES_DIR.mkdir(exist_ok=True)
    out = config.FIGURES_DIR / "common_space_intrinsic_metrics.csv"
    df.to_csv(out)
    print("\n" + df.round(4).to_string())
    print(f"\nSaved -> {out}")
    print("\nNote: all methods scored on the SAME 219-D data and the same sample,\n"
          "so these values are directly comparable (unlike the per-method spaces).")


if __name__ == "__main__":
    main()
