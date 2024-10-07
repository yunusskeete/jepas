from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# pylint: disable=redefined-outer-name


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        stage: Literal["train", "test"] = "train",
        image_file_extensions: Union[str, List[str]] = [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG",
        ],
        shuffle: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.shuffle = shuffle
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        self.transform = transform or transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),  # Normalize using ImageNet mean and std
            ]
        )

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


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        image_file_extensions: Union[str, List[str]] = [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG",
        ],
    ):
        super().__init__()

        if not isinstance(dataset_path, Path):
            dataset_path: Path = Path(dataset_path)

        if not isinstance(image_file_extensions, list):
            image_file_extensions: List[str] = [image_file_extensions]

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.image_file_extensions = image_file_extensions

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(
            dataset_path=self.dataset_path,
            stage="train",
            image_file_extensions=self.image_file_extensions,
            shuffle=self.shuffle,
        )
        self.val_dataset = ImageDataset(
            dataset_path=self.dataset_path,
            stage="val",
            image_file_extensions=self.image_file_extensions,
            shuffle=self.shuffle,
        )
        self.test_dataset = ImageDataset(
            dataset_path=self.dataset_path,
            stage="test",
            image_file_extensions=self.image_file_extensions,
            shuffle=self.shuffle,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    dataset_path: Path = Path(
        "./data/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
    ).resolve()  # Path to ImageNet dataset

    test_ijepa_dataset = ImageDataset(dataset_path, stage="test")

    test_ijepa_loader = DataLoader(test_ijepa_dataset, batch_size=32, shuffle=False)

    # Example of iterating through the test data
    for image in test_ijepa_loader:
        print(
            f"{image.shape=}"
        )  # Should print torch.Size([32, 3, img_height, img_width])
        break

    dataset = ImageDataModule(
        dataset_path=dataset_path, batch_size=32, pin_memory=False
    )
    dataset.setup()

    val_dataloader: DataLoader = dataset.val_dataloader()

    # Example of iterating through the validation data
    for image in val_dataloader:
        print(
            f"{image.shape=}"
        )  # Should print torch.Size([32, 3, img_height, img_width])
        break
