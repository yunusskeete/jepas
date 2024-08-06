from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


class ImageFolder(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        dataset_path: Union[str, Path],
        stage: Literal["train", "test"] = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        if not isinstance(dataset_path, Path):
            dataset_path: Path = Path(dataset_path)

        # Define the path to the dataset based on the stage
        self.data_path: Path = dataset_path / ("train" if stage == "train" else "test")
        assert (
            self.data_path.exists()
        ), f"Dataset path does not exist: '{self.data_path}'"

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

        super(ImageFolder, self).__init__(root=self.data_path, transform=self.transform)

    def __getitem__(self, index: int):
        # Use the super class method to get the image and label
        image, _ = super(ImageFolder, self).__getitem__(index)

        return image


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        # world_size: int = 1,
        # rank: int = 0,
        # distributed: bool = False,
        # collate_fn: Optional[Callable] = None,
        # persistent_workers: bool = False,
    ):
        super().__init__()

        if not isinstance(dataset_path, Path):
            dataset_path: Path = Path(dataset_path)

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        # self.world_size = world_size
        # self.rank = rank
        # self.distributed = distributed
        # self.collate_fn = collate_fn
        # self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.train_dist_sampler = None
        self.val_dataset = None
        self.val_dist_sampler = None

    def setup(self, stage=None):
        self.train_dataset = ImageFolder(
            dataset_path=self.dataset_path,
            stage="train",
        )
        # if self.distributed:
        #     self.train_dist_sampler = DistributedSampler(
        #         dataset=self.train_dataset, num_replicas=self.world_size, rank=self.rank
        #     )

        self.val_dataset = ImageFolder(
            dataset_path=self.dataset_path,
            stage="val",
        )
        # if self.distributed:
        #     self.val_dist_sampler = DistributedSampler(
        #         dataset=self.val_dataset, num_replicas=self.world_size, rank=self.rank
        #     )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # sampler=self.train_dist_sampler,
            num_workers=self.num_workers,
            # collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            # persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # sampler=self.val_dist_sampler,
            num_workers=self.num_workers,
            # collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            # persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    dataset_path: Path = Path(
        "./data/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
    ).resolve()  # Path to ImageNet dataset

    test_ijepa_dataset = ImageFolder(dataset_path, stage="test")

    test_ijepa_loader = DataLoader(test_ijepa_dataset, batch_size=32, shuffle=False)

    # Example of iterating through the test data
    for batch_image_tensors in test_ijepa_loader:
        print(
            batch_image_tensors.shape
        )  # Should print torch.Size([32, 3, img_height, img_width])
        break

    dataset = ImageDataModule(
        dataset_path=dataset_path, batch_size=32, pin_memory=False
    )
    dataset.setup()

    val_dataloader = dataset.val_dataloader()

    # Example of iterating through the validation data
    for batch_image_tensors in val_dataloader:
        print(
            batch_image_tensors.shape
        )  # Should print torch.Size([32, 3, img_height, img_width])
        break
