from pathlib import Path
from typing import Any, Dict, List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from configs import get_image_dataset_config

from .image_dataset import ImageDataset

dataset_config = get_image_dataset_config()


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
        shuffle: bool,
        image_file_extensions: Union[str, List[str]] = dataset_config[
            "ACCEPTABLE_FILE_EXTENSIONS"
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
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.image_file_extensions = image_file_extensions

        self.train_dataset: ImageDataset = None
        self.val_dataset: ImageDataset = None
        self.test_dataset: ImageDataset = None

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
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )


def create_image_datamodule(image_config: Dict[str, Any]) -> ImageDataModule:
    _dataset_config: Dict[str, Any] = image_config["dataset"]
    experiment_config: Dict[str, Any] = image_config["experiment"]

    return ImageDataModule(
        dataset_path=_dataset_config["DATASET_PATH"],
        batch_size=experiment_config["BATCH_SIZE"],
        num_workers=experiment_config["NUM_WORKERS"],
        pin_memory=experiment_config["PIN_MEMORY"],
        persistent_workers=experiment_config["PERSISTENT_WORKERS"],
        prefetch_factor=experiment_config["PREFETCH_FACTOR"],
        shuffle=_dataset_config["SHUFFLE_DATASET"],
    )
