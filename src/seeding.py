
import os
import random

import numpy as np

from .config import SEED


def seed_everything(seed: int = SEED) -> int:
    """Seed python, numpy and (if installed) torch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN (only affects GPU runs; harmless on CPU).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
