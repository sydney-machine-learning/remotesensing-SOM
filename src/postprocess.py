import numpy as np
from scipy.ndimage import uniform_filter


def labels_to_spatial(labels: np.ndarray, valid_mask_full: np.ndarray,
                      rows: int, cols: int) -> np.ndarray:
    """Scatter 1-D cluster labels back into a (rows, cols) map; invalid = NaN."""
    full = np.full(rows * cols, np.nan, dtype=float)
    full[valid_mask_full] = labels
    return full.reshape(rows, cols)


def majority_filter(label_image: np.ndarray, size: int = 7) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Filter size must be odd.")

    valid = ~np.isnan(label_image)
    if not valid.any():
        return label_image.copy()

    labels_present = np.unique(label_image[valid]).astype(int)
    labels_present.sort()

    h, w = label_image.shape
  
    best_count = np.zeros((h, w), dtype=np.float32)
    best_label = np.full((h, w), np.nan, dtype=float)

  
    valid_f = valid.astype(np.float32)
  
    area = size * size
    valid_counts = uniform_filter(valid_f, size=size, mode="constant") * area

    for lbl in labels_present:
        ind = (label_image == lbl).astype(np.float32)
        cnt = uniform_filter(ind, size=size, mode="constant") * area
        
        win = cnt > best_count
        best_count = np.where(win, cnt, best_count)
        best_label = np.where(win, float(lbl), best_label)

    # Pixels with no valid neighbour (isolated) or originally NaN -> NaN.
    out = np.where((valid_counts > 0) & valid, best_label, np.nan)
    # Preserve NaN for pixels that were invalid to begin with.
    out[~valid] = np.nan
    return out
