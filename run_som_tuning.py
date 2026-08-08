import itertools

import numpy as np
import pandas as pd

from src import config
from src.metrics import external_metrics
from src.pipeline import train_som
from src.postprocess import labels_to_spatial, majority_filter
from src.preprocessing import load_and_preprocess_enmap
from src.seeding import seed_everything


SIGMAS = [0.5, 1.0, 1.5, 2.0]
LEARNING_RATES = [0.1, 0.3, 0.5, 0.8]
EPOCHS = [1, 3]


def main():
    seed_everything()
    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid

    ref = None
    if config.GEOLOGY_REF_TIF.exists():
        import rioxarray as rxr
        ref = rxr.open_rasterio(str(config.GEOLOGY_REF_TIF)).squeeze().values.flatten()
    valid_classes = list(config.CLASS_NAMES.keys())

    rows = {}
    for sigma, lr, ep in itertools.product(SIGMAS, LEARNING_RATES, EPOCHS):
        tag = f"sigma={sigma}, lr={lr}, epochs={ep}"
        seed_everything()
        labels, _, info, _ = train_som(X, seed=config.SEED, sigma=sigma,
                                       learning_rate=lr, epochs=ep)
        row = {"sigma": sigma, "learning_rate": lr, "epochs": ep,
               "QE": info["Quantization Error"],
               "TE": info["Topographic Error"],
               "n_clusters": info["n_clusters"],
               "runtime_s": round(info["runtime_s"], 1)}
        if ref is not None:
            spatial = majority_filter(
                labels_to_spatial(labels, res.valid_mask_full, res.rows, res.cols),
                config.MAJORITY_FILTER_SIZE)
            pred = spatial.flatten()
            mask = ~np.isnan(pred) & np.isin(ref, valid_classes)
            ext = external_metrics(ref[mask].astype(int), pred[mask].astype(int))
            row.update({"Kanimblan Recall": ext["Kanimblan Recall"],
                        "Balanced Accuracy": ext["Balanced Accuracy"],
                        "Kappa": ext["Kappa"],
                        "Overall Accuracy": ext["Overall Accuracy"]})
        rows[tag] = row
        print(f"  {tag:24s} QE={row['QE']:.4f} TE={row['TE']:.4f} "
              f"clusters={row['n_clusters']}"
              + (f" Kanimblan={row['Kanimblan Recall']:.3f} "
                 f"Kappa={row['Kappa']:.3f}" if ref is not None else ""), flush=True)

    df = pd.DataFrame(rows).T
    config.FIGURES_DIR.mkdir(exist_ok=True)
    out = config.FIGURES_DIR / "som_tuning.csv"
    df.to_csv(out)
    print("\n=== SOM TUNING (correct training) ===")
    print(df.round(4).to_string())


    best_te = df["TE"].astype(float).idxmin()
    print(f"\nBest topology preservation : {best_te}  (TE={df.loc[best_te,'TE']:.4f})")
    if ref is not None:
        best_k = df["Kanimblan Recall"].astype(float).idxmax()
        best_kappa = df["Kappa"].astype(float).idxmax()
        print(f"Best Kanimblan recall      : {best_k}  "
              f"({df.loc[best_k,'Kanimblan Recall']:.4f})")
        print(f"Best kappa                 : {best_kappa}  "
              f"({df.loc[best_kappa,'Kappa']:.4f})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
