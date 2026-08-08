import json

import numpy as np

from .config import FIGURES_DIR, MAJORITY_FILTER_SIZE, OUTPUTS_DIR, SEED
from .metrics import intrinsic_metrics
from .postprocess import labels_to_spatial, majority_filter


def finalize_method(name: str, result, labels_raw: np.ndarray,
                    features_for_intrinsic: np.ndarray,
                    filter_size: int = MAJORITY_FILTER_SIZE, verbose: bool = True,
                    save_features: bool = True, extra: dict = None):
  
    OUTPUTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    labels_raw = np.asarray(labels_raw).astype(int)

  
    spatial_raw = labels_to_spatial(labels_raw, result.valid_mask_full,
                                    result.rows, result.cols)
    spatial_filt = majority_filter(spatial_raw, size=filter_size)

   
    intr = intrinsic_metrics(features_for_intrinsic, labels_raw, seed=SEED)

   
    np.save(OUTPUTS_DIR / f"{name}_labels_spatial.npy", spatial_filt)
    np.save(OUTPUTS_DIR / f"{name}_labels_valid.npy", labels_raw)

   
    if save_features:
        feats = np.asarray(features_for_intrinsic)
        if feats.shape[1] <= 32:
            np.save(OUTPUTS_DIR / f"{name}_features.npy", feats.astype(np.float32))
        elif verbose:
            print(f"[{name}] features are {feats.shape[1]}-D (raw data) - not saved; "
                  f"reproduce via preprocessing")

    payload = {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                   else float(v)) for k, v in intr.items()}
    if extra:
        payload.update(extra)
    with open(OUTPUTS_DIR / f"{name}_intrinsic.json", "w") as f:
        json.dump(payload, f, indent=2)

    if verbose:
        uniq, counts = np.unique(labels_raw, return_counts=True)
        print(f"[{name}] {len(uniq)} clusters, {len(labels_raw):,} valid pixels")
        for u, c in zip(uniq, counts):
            print(f"   cluster {u}: {c:,} ({100 * c / len(labels_raw):.1f}%)")
        print(f"[{name}] intrinsic: "
              f"Silhouette={intr['Silhouette']:.4f}  "
              f"CH={intr['CH Index']:.1f}  DB={intr['DB Index']:.4f}")
        print(f"[{name}] saved -> outputs/{name}_labels_spatial.npy, "
              f"{name}_labels_valid.npy, {name}_intrinsic.json")

    return spatial_filt
