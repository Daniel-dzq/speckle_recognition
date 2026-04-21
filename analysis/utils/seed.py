"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random
from typing import Optional


def seed_everything(seed: Optional[int]) -> Optional[int]:
    """Seed Python / NumPy / PyTorch (if available). Returns the applied seed."""
    if seed is None:
        return None
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - best effort
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
    return seed


__all__ = ["seed_everything"]
