import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from src import config
from src.preprocessing import load_and_preprocess_enmap
from src.pipeline import train_som
from src.seeding import seed_everything

SAMPLE = 100_000
SIL_SAMPLE = 10_000
SOM_GRIDS = [(2, 2), (2, 3), (3, 3), (4, 4), (5, 5)]
K_RANGE = range(2, 11)
FEATURE_METHODS = {"k-means + PCA": "pca", "k-means + CA": "ca", "k-means + SA": "sa"}


def main():
    seed_everything()
    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(X), min(SAMPLE, len(X)), replace=False)
    sil_idx = idx[:SIL_SAMPLE]
    config.FIGURES_DIR.mkdir(exist_ok=True)

   
    print("=== SOM grid selection (its own metrics) ===", flush=True)
    som_rows = {}
    for grid in SOM_GRIDS:
        seed_everything()
        _, _, info, _ = train_som(X, seed=config.SEED, grid=grid)
        som_rows[f"{grid[0]}x{grid[1]}"] = {
            "n_clusters": grid[0] * grid[1],
            "Quantization Error": info["Quantization Error"],
            "Topographic Error": info["Topographic Error"],
            "runtime_s": info["runtime_s"],
        }
        print(f"  {grid[0]}x{grid[1]}  QE={info['Quantization Error']:.4f}  "
              f"TE={info['Topographic Error']:.4f}", flush=True)
    som_df = pd.DataFrame(som_rows).T
    som_df.to_csv(config.FIGURES_DIR / "som_grid_selection.csv")
    print("\n" + som_df.round(4).to_string())
    print("\nQE falls as the grid grows (more prototypes = finer representation).")
    print("TE rising indicates the map is losing topology - the thing the SOM is for.")
    print(f"Saved -> {config.FIGURES_DIR / 'som_grid_selection.csv'}\n")

   
    print("=== k selection for k-means methods (uses saved features) ===", flush=True)
    all_k = {}
    for label, key in FEATURE_METHODS.items():
        fp = config.OUTPUTS_DIR / f"{key}_features.npy"
        if not fp.exists():
            print(f"  MISSING {fp.name} - run run_all_methods.py first"); continue
        F = np.load(fp)
        rows = {}
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=config.SEED,
                        n_init=config.KMEANS_N_INIT).fit(F)
            lab = km.labels_
            rows[k] = {
                "inertia": float(km.inertia_),
                "Silhouette": float(silhouette_score(F[sil_idx], lab[sil_idx])),
                "CH": float(calinski_harabasz_score(F[idx], lab[idx])),
                "DB": float(davies_bouldin_score(F[idx], lab[idx])),
            }
            print(f"  {label:15s} k={k:<2d} Sil={rows[k]['Silhouette']:.4f} "
                  f"CH={rows[k]['CH']:.0f} DB={rows[k]['DB']:.4f}", flush=True)
        df = pd.DataFrame(rows).T
        best_sil = int(df["Silhouette"].idxmax()); best_db = int(df["DB"].idxmin())
        print(f"  -> {label}: best k by silhouette={best_sil}, by DB={best_db} "
              f"(paper uses k={config.KMEANS_K})\n", flush=True)
        df.index.name = "k"
        all_k[label] = df
        df.to_csv(config.FIGURES_DIR / f"k_selection_{key}.csv")

    if all_k:
        summary = pd.DataFrame({
            lab: {"best k (silhouette)": int(d["Silhouette"].idxmax()),
                  "best k (DB)": int(d["DB"].idxmin()),
                  "paper k": config.KMEANS_K}
            for lab, d in all_k.items()}).T
        summary.to_csv(config.FIGURES_DIR / "k_selection_summary.csv")
        print("=== k selection summary ===")
        print(summary.to_string())
        print(f"\nSaved -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
