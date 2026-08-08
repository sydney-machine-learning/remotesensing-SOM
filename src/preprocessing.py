from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import ENMAP_TIF, FILL_VALUE, REFLECTANCE_SCALE


@dataclass
class PreprocessResult:
    data_valid: np.ndarray        
    valid_mask_full: np.ndarray  
    rows: int
    cols: int
    bands_original: int
    bands_with_data: List[int]    
    extent: Optional[list]       
    scaler: object = None         

    @property
    def n_valid_pixels(self) -> int:
        return int(self.data_valid.shape[0])

    @property
    def n_valid_bands(self) -> int:
        return int(self.data_valid.shape[1])


def preprocess_cube(cube: np.ndarray) -> "PreprocessResult":
    if cube.ndim != 3:
        raise ValueError(f"Expected (bands, rows, cols), got shape {cube.shape}")

    bands, rows, cols = cube.shape
   
    data_reshaped = cube.reshape(bands, -1).T

   
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            band_means = np.nanmean(data_reshaped, axis=0)
    bands_with_data = np.where(~np.isnan(band_means))[0]

   
    data_filled = data_reshaped[:, bands_with_data].copy()
    col_means = np.nanmean(data_filled, axis=0)
    for j in range(data_filled.shape[1]):
        col = data_filled[:, j]
        m = np.isnan(col)
        if m.any():
            col[m] = col_means[j]

  
    nan_per_pixel = np.isnan(data_reshaped).sum(axis=1)
    completely_invalid = nan_per_pixel == bands
    valid_mask_full = ~completely_invalid

    data_valid = data_filled[valid_mask_full].astype(np.float32)
    data_valid = np.ascontiguousarray(data_valid)

    if np.isnan(data_valid).any():
        raise ValueError("NaNs remain after preprocessing - check inputs.")

    return PreprocessResult(
        data_valid=data_valid,
        valid_mask_full=valid_mask_full,
        rows=rows,
        cols=cols,
        bands_original=bands,
        bands_with_data=[int(b) for b in bands_with_data],
        extent=None,
    )


def load_and_preprocess_enmap(path=ENMAP_TIF, standardize: bool = False, verbose: bool = True):

    import rioxarray as rxr  # imported lazily so the core is testable without it

    data = rxr.open_rasterio(str(path)).squeeze()
    # Canonical fill handling: replace the sensor no-data code, then scale.
    data = data.where(data != FILL_VALUE)
    data = data / REFLECTANCE_SCALE

    extent = [
        float(data.x.min().values), float(data.x.max().values),
        float(data.y.min().values), float(data.y.max().values),
    ]
    cube = data.values  # (bands, rows, cols)

    result = preprocess_cube(cube)
    result.extent = extent

    if standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        result.data_valid = scaler.fit_transform(result.data_valid).astype(np.float32)
        result.scaler = scaler

    if verbose:
        print(f"Loaded EnMAP: {result.bands_original} bands -> "
              f"{result.n_valid_bands} valid bands")
        print(f"Valid pixels: {result.n_valid_pixels:,} of "
              f"{result.rows * result.cols:,} "
              f"({100 * result.n_valid_pixels / (result.rows * result.cols):.1f}%)")
        print(f"Standardised: {standardize}")

    return result
