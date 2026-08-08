import argparse
import gc
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

from src import config
from src.preprocessing import load_and_preprocess_enmap
from src.pipeline import train_ca, train_pca, train_pca_scores, train_sa, train_som
from src.seeding import seed_everything

SAMPLE = 100_000      
SIL_SAMPLE = 10_000 


def score_common(X_ch, l_ch, X_sil, l_sil):
    return {
        "Silhouette": float(silhouette_score(X_sil, l_sil)),
        "CH": float(calinski_harabasz_score(X_ch, l_ch)),
        "DB": float(davies_bouldin_score(X_ch, l_ch)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-sa", action="store_true",
                    help="skip the stacked AE (slowest method)")
    args = ap.parse_args()
    seeds = [config.SEED + i for i in range(args.seeds)]

    seed_everything()
    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid

    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(X), min(SAMPLE, len(X)), replace=False)
    sil_idx = idx[:SIL_SAMPLE]
    Xs, Xsil = X[idx], X[sil_idx]
    print(f"Seeds: {seeds}\nCommon 219-D space | CH/DB n={len(idx):,} | "
          f"Silhouette n={len(sil_idx):,}\n", flush=True)

   
    pca_scores, _ = train_pca_scores(X, seed=config.SEED)

    methods = {
        "SOM": lambda s: train_som(X, seed=s)[0],
        "k-means + PCA": lambda s: train_pca(X, seed=s, scores=pca_scores)[0],
        "k-means + CA": lambda s: train_ca(X, seed=s)[0],
    }
    if not args.skip_sa:
        methods["k-means + SA"] = lambda s: train_sa(X, seed=s)[0]

    config.FIGURES_DIR.mkdir(exist_ok=True)
    out = config.FIGURES_DIR / "multiseed_results.csv"

    def checkpoint(rows_so_far):
       
        if rows_so_far:
            pd.DataFrame(rows_so_far).T.to_csv(out)

    rows, stability = {}, {}
    for name, fn in methods.items():
        runs, labels_per_seed = [], []
        for s in seeds:
            t0 = time.time()
            seed_everything(s)
            lab = fn(s)
            labels_per_seed.append(lab)
            m = score_common(Xs, lab[idx], Xsil, lab[sil_idx])
            m["n_clusters"] = int(len(np.unique(lab)))
            m["runtime_s"] = time.time() - t0
            runs.append(m)
            print(f"  {name:15s} seed={s}  Sil={m['Silhouette']:.4f}  "
                  f"CH={m['CH']:.0f}  DB={m['DB']:.4f}  ({m['runtime_s']:.0f}s)",
                  flush=True)
            gc.collect()

        df = pd.DataFrame(runs)
        rows[name] = {f"{c} mean": df[c].mean() for c in ["Silhouette", "CH", "DB"]}
        rows[name].update({f"{c} std": df[c].std(ddof=1) for c in ["Silhouette", "CH", "DB"]})
        rows[name]["n_clusters"] = int(df["n_clusters"].mode()[0])
        rows[name]["runtime_s mean"] = df["runtime_s"].mean()

      
        aris = [adjusted_rand_score(a[idx], b[idx])
                for a, b in itertools.combinations(labels_per_seed, 2)]
        rows[name]["Stability (mean pairwise ARI)"] = float(np.mean(aris))
        rows[name]["Stability std"] = float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0
        rows[name]["n_seeds"] = len(seeds)
        stability[name] = aris
        print(f"  {name:15s} STABILITY mean pairwise ARI = {np.mean(aris):.4f}\n", flush=True)

        checkpoint(rows)
        del labels_per_seed, runs
        gc.collect()

    df = pd.DataFrame(rows).T
    df.to_csv(out)
    with open(config.FIGURES_DIR / "multiseed_stability_pairs.json", "w") as f:
        json.dump(stability, f, indent=2)

    print("\n=== MULTI-SEED SUMMARY (mean +/- std over "
          f"{len(seeds)} seeds, common 219-D space) ===")
    print(df.round(4).to_string())
    print(f"\nSaved -> {out}")
    print("\nStability = mean pairwise ARI between seeds. 1.0 = identical map every\n"
          "run; lower means the published map depends on the random seed.")


if __name__ == "__main__":
    main()
