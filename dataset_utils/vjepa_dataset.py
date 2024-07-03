from pathlib import Path
from typing import List, Literal, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class VJEPADataset(Dataset):
    def __init__(self):
        super().__init__()
        pass

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index) -> torch.Tensor:
        raise NotImplementedError()
