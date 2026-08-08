import numpy as np
import pandas as pd
from src import config
from src.metrics import external_metrics
from src.pipeline import train_som
from src.postprocess import labels_to_spatial, majority_filter
from src.preprocessing import load_and_preprocess_enmap
from src.seeding import seed_everything


BUDGETS = [
    ("PAPER: 1000 iters, first 1000 px", {"legacy_iterations": 1000}),
    ("1 epoch  (all pixels)",            {"epochs": 1}),
    ("3 epochs (all pixels)",            {"epochs": 3}),
    ("5 epochs (all pixels)",            {"epochs": 5}),
]


def main():
    seed_everything()
    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid

    ref = None
    if config.GEOLOGY_REF_TIF.exists():
        import rioxarray as rxr
        ref = rxr.open_rasterio(str(config.GEOLOGY_REF_TIF)).squeeze().values.flatten()
        print("Ground truth found - geology metrics included.\n")
    else:
        print("No ground-truth raster - reporting SOM metrics only.\n")

    valid_classes = list(config.CLASS_NAMES.keys())
    rows = {}
    for label, kw in BUDGETS:
        seed_everything()
        labels, _, info, _ = train_som(X, seed=config.SEED, **kw)
        row = {
            "Quantization Error": info["Quantization Error"],
            "Topographic Error": info["Topographic Error"],
            "n_clusters": info["n_clusters"],
            "pixels_used": kw.get("legacy_iterations", info.get("updates", 0)),
            "runtime_s": round(info["runtime_s"], 1),
        }
        if ref is not None:
            spatial = majority_filter(
                labels_to_spatial(labels, res.valid_mask_full, res.rows, res.cols),
                config.MAJORITY_FILTER_SIZE)
            pred = spatial.flatten()
            mask = ~np.isnan(pred) & np.isin(ref, valid_classes)
            ext = external_metrics(ref[mask].astype(int), pred[mask].astype(int))
            row.update({
                "Kanimblan Recall": ext["Kanimblan Recall"],
                "Overall Accuracy": ext["Overall Accuracy"],
                "Balanced Accuracy": ext["Balanced Accuracy"],
                "Kappa": ext["Kappa"],
            })
        rows[label] = row
        print(f"{label:36s} QE={row['Quantization Error']:.4f} "
              f"TE={row['Topographic Error']:.4f}"
              + (f"  Kanimblan={row['Kanimblan Recall']:.3f}"
                 f"  OA={row['Overall Accuracy']:.3f}" if ref is not None else ""),
              flush=True)

    df = pd.DataFrame(rows).T
    config.FIGURES_DIR.mkdir(exist_ok=True)
    out = config.FIGURES_DIR / "som_training_fix.csv"
    df.to_csv(out)
    print("\n=== SOM TRAINING BUDGET: BEFORE / AFTER ===")
    print(df.round(4).to_string())
    print(f"\nSaved -> {out}")

    base = df.loc["PAPER: 1000 iters, first 1000 px"]
    best = df.loc["5 epochs (all pixels)"]
    print(f"\nTopographic error {base['Topographic Error']:.4f} -> "
          f"{best['Topographic Error']:.4f} "
          f"({base['Topographic Error'] / max(best['Topographic Error'], 1e-9):.0f}x better)")
    if ref is not None:
        print(f"Kanimblan recall  {base['Kanimblan Recall']:.4f} -> "
              f"{best['Kanimblan Recall']:.4f} "
              f"({best['Kanimblan Recall'] - base['Kanimblan Recall']:+.4f})")


if __name__ == "__main__":
    main()
