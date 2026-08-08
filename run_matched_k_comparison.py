
import numpy as np
import pandas as pd
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from src import config
from src.pipeline import train_som
from src.preprocessing import load_and_preprocess_enmap
from src.seeding import seed_everything

SAMPLE = 100_000
SILHOUETTE_SAMPLE = 10_000
KMEANS_METHODS = {"k-means + PCA": "pca", "k-means + CA": "ca", "k-means + SA": "sa"}


def score(X_ch, l_ch, X_sil, l_sil):
    return {
        "n_clusters": int(len(np.unique(l_ch))),
        "Silhouette (219-D)": float(silhouette_score(X_sil, l_sil)),
        "CH (219-D)": float(calinski_harabasz_score(X_ch, l_ch)),
        "DB (219-D)": float(davies_bouldin_score(X_ch, l_ch)),
    }


def main():
    seed_everything()
    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid

    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(X), min(SAMPLE, len(X)), replace=False)
    sil_idx = idx[:SILHOUETTE_SAMPLE]
    Xs, Xsil = X[idx], X[sil_idx]
    print(f"Common 219-D space | CH/DB n={len(idx):,} | Silhouette n={len(sil_idx):,}\n")

    rows = {}

   
    som4, _, som_info, _ = train_som(X, seed=config.SEED, grid=(2, 2))
    rows["SOM (4 clusters)"] = score(Xs, som4[idx], Xsil, som4[sil_idx])
    print(f"{'SOM (4 clusters)':20s} {rows['SOM (4 clusters)']}")
    print(f"{'':20s} QE={som_info['Quantization Error']:.4f} "
          f"TE={som_info['Topographic Error']:.4f}")

   
    for label, key in KMEANS_METHODS.items():
        p = config.OUTPUTS_DIR / f"{key}_labels_valid.npy"
        if not p.exists():
            print(f"  MISSING {p.name}"); continue
        lab = np.load(p)
        rows[label] = score(Xs, lab[idx], Xsil, lab[sil_idx])
        print(f"{label:20s} {rows[label]}")

   
    p = config.OUTPUTS_DIR / "som_labels_valid.npy"
    if p.exists():
        lab = np.load(p)
        rows["SOM (9 clusters, unmatched)"] = score(Xs, lab[idx], Xsil, lab[sil_idx])

    df = pd.DataFrame(rows).T
    config.FIGURES_DIR.mkdir(exist_ok=True)
    out = config.FIGURES_DIR / "matched_k_comparison.csv"
    df.to_csv(out)
    print("\n" + df.round(4).to_string())
    print(f"\nSaved -> {out}")
    print("\nRows 1-4 are a controlled comparison (same k=4, same 219-D space,\n"
          "same sample). The 9-cluster SOM is listed only for context - its\n"
          "metrics are not comparable to the k=4 rows.")


if __name__ == "__main__":
    main()
