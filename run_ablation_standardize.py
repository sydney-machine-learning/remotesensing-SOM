import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

from src import config
from src.preprocessing import load_and_preprocess_enmap
from src.seeding import seed_everything

SAMPLE = 100_000
SIL_SAMPLE = 10_000


def main():
    seed_everything()
    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid

    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(X), min(SAMPLE, len(X)), replace=False)
    sil_idx = idx[:SIL_SAMPLE]
    Xs_common, Xsil_common = X[idx], X[sil_idx]

    rows = {}
    for label, standardize in [("PCA raw reflectance", False),
                               ("PCA standardised", True)]:
        feats = StandardScaler().fit_transform(X) if standardize else X
        pca = PCA(n_components=config.PCA_N_COMPONENTS, random_state=config.SEED)
        scores = pca.fit_transform(feats).astype(np.float32)
        if standardize:
            del feats
        lab = KMeans(config.KMEANS_K, random_state=config.SEED,
                     n_init=config.KMEANS_N_INIT).fit_predict(scores)

        # Scored in the common 219-D space so the two are comparable.
        rows[label] = {
            "cumulative variance (10 PCs)": float(np.sum(pca.explained_variance_ratio_)),
            "PC1 variance share": float(pca.explained_variance_ratio_[0]),
            "Silhouette (219-D)": float(silhouette_score(Xsil_common, lab[sil_idx])),
            "CH (219-D)": float(calinski_harabasz_score(Xs_common, lab[idx])),
            "DB (219-D)": float(davies_bouldin_score(Xs_common, lab[idx])),
        }
        print(f"{label:22s} {rows[label]}", flush=True)
        del scores

    df = pd.DataFrame(rows).T
    config.FIGURES_DIR.mkdir(exist_ok=True)
    out = config.FIGURES_DIR / "ablation_standardize.csv"
    df.to_csv(out)
    print("\n" + df.round(4).to_string())

    a, b = "PCA raw reflectance", "PCA standardised"
    d_sil = df.loc[b, "Silhouette (219-D)"] - df.loc[a, "Silhouette (219-D)"]
    d_db = df.loc[b, "DB (219-D)"] - df.loc[a, "DB (219-D)"]
    print(f"\nStandardising changes Silhouette by {d_sil:+.4f} and DB by {d_db:+.4f}")
    print("(higher Silhouette better, lower DB better)")
    verdict = "helps" if (d_sil > 0 and d_db < 0) else (
        "hurts" if (d_sil < 0 and d_db > 0) else "is mixed/negligible")
    print(f"Verdict: standardisation {verdict} for PCA + k-means here.")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
