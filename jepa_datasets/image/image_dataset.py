from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from configs import get_image_dataset_config

from .image_transforms import make_transforms

dataset_config = get_image_dataset_config()
# pylint: disable=dangerous-default-value


class ImageDataset(TorchDataset):
    """
    Wraps a PyTorch dataset for loading images from a directory, applying transformations, and returning tensors.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        stage: Literal["train", "test"],
        image_file_extensions: Union[str, List[str]] = dataset_config[
            "ACCEPTABLE_FILE_EXTENSIONS"
        ],
        shuffle: bool = dataset_config["SHUFFLE_DATASET"],
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()

        self.shuffle = shuffle
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        self.transform = transform or transforms.Compose(transforms=make_transforms())

        if not isinstance(dataset_path, Path):
            dataset_path: Path = Path(dataset_path)

        if not isinstance(image_file_extensions, list):
            image_file_extensions: List[str] = [image_file_extensions]

        # Define the path to the dataset based on the stage
        self.data_path: Path = dataset_path / ("train" if stage == "train" else "test")

        # Collect all image paths with the desired extensions
        self.image_paths: List[Path] = []
        for ext in image_file_extensions:
            self.image_paths.extend(self.data_path.rglob(f"*{ext}"))

        if self.shuffle:
            np.random.shuffle(self.image_paths)  # In place shuffle

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> torch.Tensor:
        # Load the image file
        image_path: Path = self.image_paths[index]
        image: Image = Image.open(image_path).convert("RGB")
        # Apply transformations
        image_tensor: torch.Tensor = self.transform(image)

        return image_tensor
