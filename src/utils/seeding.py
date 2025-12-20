from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
