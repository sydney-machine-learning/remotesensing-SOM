
from typing import Dict

import numpy as np

from .config import SEED


def som_quality(som, data: np.ndarray, sample: int = 50_000,
                seed: int = SEED) -> Dict[str, float]:
   
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    if len(data) > sample:
        idx = rng.choice(len(data), sample, replace=False)
        d = data[idx]
    else:
        d = data

    return {
        "Quantization Error": float(som.quantization_error(d)),
        "Topographic Error": float(som.topographic_error(d)),
        "n_samples": int(len(d)),
    }


def describe_topographic_error(te: float) -> str:
    if te < 0.05:
        return "excellent topology preservation"
    if te < 0.10:
        return "good topology preservation"
    if te < 0.20:
        return "moderate topology preservation"
    return "poor topology preservation"
