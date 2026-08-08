import argparse
import gc
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import load_and_preprocess_enmap
from src.pipeline import train_ca, train_pca, train_sa, train_som
from src.results import finalize_method
from src.seeding import seed_everything
from src.som_metrics import describe_topographic_error


def banner(msg):
    print("\n" + "=" * 64 + f"\n{msg}\n" + "=" * 64, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["som", "pca", "ca", "sa"],
                    help="run only these methods (others keep their existing "
                         "outputs). Useful to redo one method without repeating "
                         "the expensive ones.")
    args = ap.parse_args()
    want = set(args.only) if args.only else {"som", "pca", "ca", "sa"}

    seed_everything()
    t_all = time.time()
    runtimes = {}

    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid

    # ---- Method 1: SOM ----------------------------------------------------
    if "som" in want:
        banner(f"METHOD 1: SOM {config.SOM_GRID[0]}x{config.SOM_GRID[1]} on {X.shape[1]}-D")
        # verbose=False: MiniSom prints a progress line per update, and now that
        # the SOM correctly trains on all 1,044,103 pixels that is ~1M lines (~44 MB).
        labels, _, info, som = train_som(X, seed=config.SEED, verbose=False)
        print(f"  QE={info['Quantization Error']:.4f}  TE={info['Topographic Error']:.4f}"
              f"  -> {describe_topographic_error(info['Topographic Error'])}")
        finalize_method("som", res, labels, X, extra=info)
        np.save(config.OUTPUTS_DIR / "som_weights.npy", som.get_weights())
        runtimes["SOM"] = info["runtime_s"]
        del som; gc.collect()

    # ---- Method 2: k-means + PCA -----------------------------------------
    if "pca" in want:
        banner("METHOD 2: k-means + PCA (standardised)")
        labels, scores, info = train_pca(X, seed=config.SEED)
        print(f"  cumulative variance: {info['cumulative_variance'] * 100:.2f}%")
        # Saved so the explained-variance figure can be redrawn without refitting.
        np.save(config.OUTPUTS_DIR / "pca_explained_variance.npy",
                np.asarray(info.pop("explained_variance_ratio")))
        finalize_method("pca", res, labels, scores, extra=info)
        runtimes["PCA"] = info["runtime_s"] + info["kmeans_runtime_s"]
        del scores; gc.collect()

    # ---- Method 3: k-means + Canonical AE --------------------------------
    if "ca" in want:
        banner("METHOD 3: k-means + Canonical Autoencoder")
        labels, latent, info = train_ca(X, seed=config.SEED, verbose=True)
        print(f"  epochs run: {info['epochs_run']} (early stop: {info['stopped_early']})")
        finalize_method("ca", res, labels, latent, extra=info)
        runtimes["CA"] = info["runtime_s"]
        del latent; gc.collect()

    # ---- Method 4: k-means + Stacked AE ----------------------------------
    if "sa" in want:
        banner("METHOD 4: k-means + Stacked Autoencoder")
        labels, latent, info = train_sa(X, seed=config.SEED, verbose=True)
        print(f"  epochs run: {info['epochs_run']}")
        finalize_method("sa", res, labels, latent, extra=info)
        runtimes["SA"] = info["runtime_s"]
        del latent; gc.collect()

    # ---- Runtime summary (merged with any previous run) -------------------
    config.FIGURES_DIR.mkdir(exist_ok=True)
    rt_path = config.FIGURES_DIR / "runtime_summary.csv"
    if rt_path.exists():
        prev = pd.read_csv(rt_path, index_col=0)["runtime_s"].to_dict()
        prev.update(runtimes)
        runtimes = prev
    rt = pd.DataFrame({"runtime_s": runtimes})
    rt["runtime_min"] = (rt["runtime_s"] / 60).round(2)
    if "PCA" in rt.index:
        rt["relative_to_PCA"] = (rt["runtime_s"] / rt.loc["PCA", "runtime_s"]).round(1)
    rt.to_csv(rt_path)
    print("\n" + rt.round(2).to_string())
    print(f"\nDONE ({', '.join(sorted(want))}) in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
