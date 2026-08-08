import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from src import config
from src import plotting as P
from src.pipeline import train_som
from src.preprocessing import load_and_preprocess_enmap
from src.seeding import seed_everything

GRIDS = [(2, 2), (2, 3), (3, 3), (4, 4), (5, 5)]


def main():
    P.use_paper_style()
    seed_everything()
    res = load_and_preprocess_enmap(config.ENMAP_TIF, standardize=False)
    X = res.data_valid

   
    from src.postprocess import labels_to_spatial, majority_filter
    fig, axes = plt.subplots(2, 3, figsize=(P.COL2, 6.4), constrained_layout=True)
    axes = axes.ravel()
    for ax, grid in zip(axes, GRIDS):
        seed_everything()
        labels, _, info, _ = train_som(X, seed=config.SEED, grid=grid)
        m = majority_filter(labels_to_spatial(labels, res.valid_mask_full,
                                               res.rows, res.cols),
                             config.MAJORITY_FILTER_SIZE)
        nc = grid[0] * grid[1]
        colors = P.CLUSTER4 if nc <= 4 else P.SOM9[:nc] if nc <= 9 else None
        cmap = P.discrete_cmap(colors) if colors else plt.get_cmap("viridis", nc)
        ax.imshow(m, cmap=cmap, extent=res.extent, interpolation="nearest")
        ax.set_title(f"{grid[0]}x{grid[1]} ({nc} clusters), TE={info['Topographic Error']:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].axis("off")
    P.save(fig, config.FIGURES_DIR / "fig_som_grid_maps.png")
    print("  fig_som_grid_maps.png")

  
    from src.pipeline import train_pca_scores
    scores, _ = train_pca_scores(X, seed=config.SEED)
    ks = list(range(2, 11))
    inertias = [KMeans(k, random_state=config.SEED,
                       n_init=config.KMEANS_N_INIT).fit(scores).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(P.COL1, 2.6), constrained_layout=True)
    ax.plot(ks, inertias, "o-", color=P.CLUSTER4[0], markersize=4)
    ax.axvline(config.KMEANS_K, color=P.MUTED, ls="--", lw=0.9)
    ax.annotate(f"k={config.KMEANS_K}", (config.KMEANS_K, inertias[config.KMEANS_K - 2]),
                textcoords="offset points", xytext=(8, 8), fontsize=7, color=P.MUTED)
    ax.set_xlabel("Number of clusters $k$"); ax.set_ylabel("Inertia")
    ax.set_title("Elbow criterion (PCA + $k$-means)")
    ax.grid(alpha=0.4); ax.set_axisbelow(True)
    P.save(fig, config.FIGURES_DIR / "fig_pca_elbow.png")
    print("  fig_pca_elbow.png")


if __name__ == "__main__":
    main()
