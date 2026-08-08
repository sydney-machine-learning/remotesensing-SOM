import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config
from .som_metrics import som_quality


def _kmeans(features, k, seed):
    return KMeans(n_clusters=k, random_state=seed,
                  n_init=config.KMEANS_N_INIT).fit_predict(features)


def train_som(X, seed=config.SEED, grid=None, sigma=None, learning_rate=None,
              epochs=None, verbose=False, legacy_iterations=None):
   
    from minisom import MiniSom

    grid = grid or config.SOM_GRID
    sigma = config.SOM_SIGMA if sigma is None else sigma
    learning_rate = config.SOM_LEARNING_RATE if learning_rate is None else learning_rate
    epochs = config.SOM_EPOCHS if epochs is None else epochs

    t0 = time.time()
    rows_g, cols_g = grid
    som = MiniSom(rows_g, cols_g, X.shape[1], sigma=sigma,
                  learning_rate=learning_rate,
                  neighborhood_function="gaussian", random_seed=seed)
    som.random_weights_init(X)
    if legacy_iterations:
        # Reproduces the paper exactly, including the sampling bug.
        som.train_batch(X, num_iteration=legacy_iterations, verbose=verbose)
        budget = {"legacy_iterations": legacy_iterations}
    else:
        som.train(X, num_iteration=epochs, use_epochs=True,
                  random_order=True, verbose=verbose)
        budget = {"epochs": epochs, "updates": int(epochs * len(X))}

    win = np.array([som.winner(x) for x in X])
    labels = win[:, 0] * cols_g + win[:, 1]

    info = {"runtime_s": time.time() - t0, "grid": f"{rows_g}x{cols_g}",
            "sigma": sigma, "learning_rate": learning_rate,
            "n_clusters": int(len(np.unique(labels)))}
    info.update(budget)
    info.update(som_quality(som, X, seed=seed))
    return labels, X, info, som


def train_pca_scores(X_raw, seed=config.SEED, n_components=None, standardize=None):
  
    n_components = n_components or config.PCA_N_COMPONENTS
    standardize = config.PCA_STANDARDIZE if standardize is None else standardize
    t0 = time.time()
    if standardize:
        Xs = StandardScaler().fit_transform(X_raw)
    else:
        Xs = X_raw
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(Xs).astype(np.float32)
    if standardize:
        del Xs
    info = {"runtime_s": time.time() - t0,
            "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
            # kept so the explained-variance figure can be redrawn without refitting
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
    return scores, info


def train_pca(X_raw, seed=config.SEED, k=None, scores=None):
    k = k or config.KMEANS_K
    info = {}
    if scores is None:
        scores, info = train_pca_scores(X_raw, seed=seed)
    t0 = time.time()
    labels = _kmeans(scores, k, seed)
    info["kmeans_runtime_s"] = time.time() - t0
    info["n_clusters"] = int(len(np.unique(labels)))
    return labels, scores, info


def train_ca(X, seed=config.SEED, k=None, epochs=20, verbose=False):
    """Canonical autoencoder (219->6->4->2) + k-means on the 2-D latent."""
    from .autoencoders import Autoencoder, encode_full, get_device, train_autoencoder

    k = k or config.KMEANS_K
    t0 = time.time()
    device = get_device()
    ae = Autoencoder(X.shape[1], (6, 4, 2), l1_lambda=1e-4)
    hist = train_autoencoder(ae, X, epochs=epochs, batch_size=64, learning_rate=1e-4,
                             seed=seed, device=device, name="CA", verbose=verbose)
    latent = encode_full(ae, X, device).astype(np.float32)
    train_s = time.time() - t0

    labels = _kmeans(latent, k, seed)
    info = {"runtime_s": train_s, "epochs_run": hist["epochs_run"],
            "stopped_early": hist["stopped_early"], "final_loss": hist["loss"][-1],
            "n_clusters": int(len(np.unique(labels)))}
    return labels, latent, info


def train_sa(X, seed=config.SEED, k=None, verbose=False):
    """Stacked autoencoder (3 stages) + k-means on the final 2-D latent."""
    from .autoencoders import build_stacked_features, get_device

    k = k or config.KMEANS_K
    t0 = time.time()
    latent, hists = build_stacked_features(X, seed=seed, device=get_device(),
                                           verbose=verbose)
    latent = latent.astype(np.float32)
    train_s = time.time() - t0

    labels = _kmeans(latent, k, seed)
    info = {"runtime_s": train_s,
            "epochs_run": {s: h["epochs_run"] for s, h in hists.items()},
            "n_clusters": int(len(np.unique(labels)))}
    return labels, latent, info
