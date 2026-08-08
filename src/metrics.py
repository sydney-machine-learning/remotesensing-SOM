from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    cohen_kappa_score,
    completeness_score,
    confusion_matrix,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

from .config import SEED, TARGET_CLASS


def contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (matrix, classes, clusters) with matrix[i, j] = count of pixels
    of true class ``classes[i]`` assigned to ``clusters[j]``."""
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    cls_idx = {c: i for i, c in enumerate(classes)}
    clu_idx = {c: j for j, c in enumerate(clusters)}
    m = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[cls_idx[t], clu_idx[p]] += 1
    return m, classes, clusters


def map_clusters_to_classes(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, int]:
    """Majority-vote mapping: each cluster -> the true class it overlaps most.

    Many-to-one (several clusters may map to the same class), which is the
    standard "cluster then label" scheme for remote-sensing accuracy assessment
    and is appropriate when there are more clusters than reference classes.
    """
    m, classes, clusters = contingency_matrix(y_true, y_pred)
    dominant = m.argmax(axis=0)  # index into classes, per cluster
    return {int(clusters[j]): int(classes[dominant[j]]) for j in range(len(clusters))}


def apply_mapping(y_pred: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    out = np.empty_like(y_pred)
    for k, v in mapping.items():
        out[y_pred == k] = v
    return out


def external_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     target_class: int = TARGET_CLASS) -> Dict[str, float]:
    """All ground-truth metrics for one method.

    ARI/NMI/homogeneity/completeness/V-measure are permutation-invariant and
    need no mapping. Overall accuracy, balanced accuracy, kappa and per-class
    recall use the majority-vote cluster->class mapping.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    mapping = map_clusters_to_classes(y_true, y_pred)
    y_mapped = apply_mapping(y_pred, mapping)

    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_mapped, labels=classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1)
    per_class_recall = {int(c): float(recall[i]) for i, c in enumerate(classes)}

    n_clusters = int(len(np.unique(y_pred)))

    return {
        # permutation-invariant
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "Homogeneity": float(homogeneity_score(y_true, y_pred)),
        "Completeness": float(completeness_score(y_true, y_pred)),
        "V-measure": float(v_measure_score(y_true, y_pred)),
        # mapping-based (fixed)
        "Overall Accuracy": float(accuracy_score(y_true, y_mapped)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_mapped)),
        "Kappa": float(cohen_kappa_score(y_true, y_mapped)),
        "Purity": float(_purity(y_true, y_pred)),
        # exploration-critical
        "Kanimblan Recall": float(per_class_recall.get(target_class, np.nan)),
        "n_clusters": n_clusters,
        "_per_class_recall": per_class_recall,
        "_mapping": mapping,
    }


def intrinsic_metrics(features: np.ndarray, labels: np.ndarray,
                      seed: int = SEED, sample_size: int = 10000) -> Dict[str, float]:
    """Silhouette (seeded sample), Calinski-Harabasz and Davies-Bouldin, computed
    on the given feature space. ``features`` and ``labels`` must be aligned and
    have the same length (the valid-pixel ordering)."""
    features = np.asarray(features)
    labels = np.asarray(labels).astype(int)
    if len(features) != len(labels):
        raise ValueError(
            f"features ({len(features)}) and labels ({len(labels)}) misaligned")

    n_labels = len(np.unique(labels))
    if n_labels < 2:
        return {"Silhouette": np.nan, "CH Index": np.nan, "DB Index": np.nan}

    rng = np.random.default_rng(seed)
    if len(features) > sample_size:
        idx = rng.choice(len(features), sample_size, replace=False)
        feat_s, lab_s = features[idx], labels[idx]
    else:
        feat_s, lab_s = features, labels

    out = {}
    try:
        out["Silhouette"] = float(silhouette_score(feat_s, lab_s))
    except Exception:
        out["Silhouette"] = np.nan
    try:
        out["CH Index"] = float(calinski_harabasz_score(features, labels))
    except Exception:
        out["CH Index"] = np.nan
    try:
        out["DB Index"] = float(davies_bouldin_score(features, labels))
    except Exception:
        out["DB Index"] = np.nan
    return out


def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m, _, _ = contingency_matrix(y_true, y_pred)
    return m.max(axis=0).sum() / m.sum()
