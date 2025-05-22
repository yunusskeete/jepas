from pathlib import Path
from typing import Callable, Optional, Tuple, TypeVar

import torch

from model.seed import seed_everything
from utils.types import T


def save_model_weights(
    filepath: str,
    constructor: Callable[..., Tuple[T, ...]],
    seed: Optional[int] = None,
    **kwargs,
) -> T:
    if seed is not None:
        seed_everything(seed)

    model = constructor(**kwargs)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), filepath)

    print(f"✅ Initial weights saved to '{filepath}'")

    return model


def load_model_weights(
    filepath: str,
    constructor: Callable[..., Tuple[T, ...]],
    **kwargs,
) -> T:
    model = constructor(**kwargs)

    print(f"⏳ Loading initial weights from '{filepath}'")

    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)

    print(f"✅ Loaded initial weights from '{filepath}'")

    return model
