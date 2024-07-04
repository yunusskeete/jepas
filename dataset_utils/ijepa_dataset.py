from pathlib import Path
from typing import List, Literal, Union

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
    ):
        super().__init__()
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        self.transform = transforms.Compose(
            [
                # transforms.RandomSizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(256),  # Resize to a standard size
                transforms.CenterCrop(224),  # Crop to the size required by most models
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

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> torch.Tensor:
        # Load the image file
        image_path: Path = self.image_paths[index]
        image: Image = Image.open(image_path).convert("RGB")
        # Apply transformations
        image_tensor: torch.Tensor = self.transform(image)

        return image_tensor


# import os
# from pathlib import Path
# from typing import Union, List
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# import concurrent.futures

# class ImageDataset(Dataset):
#     def __init__(
#         self,
#         dataset_path: Union[str, Path],
#         stage: str = "train",
#         image_file_extensions: Union[str, List[str]] = [
#             ".jpg",
#             ".jpeg",
#             ".png",
#             ".JPG",
#             ".JPEG",
#             ".PNG",
#         ],
#         num_workers: int = 4
#     ):
#         super().__init__()

#         self.transform = transforms.Compose(
#             [
#                 transforms.Resize(256),  # Resize to a standard size
#                 transforms.CenterCrop(224),  # Crop to the size required by most models
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225],
#                 ),  # Normalize using ImageNet mean and std
#             ]
#         )

#         if not isinstance(dataset_path, Path):
#             dataset_path = Path(dataset_path)

#         if not isinstance(image_file_extensions, list):
#             image_file_extensions = [image_file_extensions]

#         self.data_path = dataset_path / ("train" if stage == "train" else "test")
#         self.image_paths = self._collect_image_paths(self.data_path, image_file_extensions)
#         self.num_workers = num_workers

#     @staticmethod
#     def _collect_image_paths(data_path: Path, image_file_extensions: List[str]) -> List[Path]:
#         image_paths = []
#         for ext in image_file_extensions:
#             image_paths.extend(data_path.rglob(f"*{ext}"))
#         return image_paths

#     def __len__(self) -> int:
#         return len(self.image_paths)

#     def _load_image(self, image_path: Path) -> torch.Tensor:
#         image = Image.open(image_path).convert("RGB")
#         return self.transform(image)

#     def __getitem__(self, index) -> torch.Tensor:
#         image_path = self.image_paths[index]
#         image_tensor = self._load_image(image_path)
#         return image_tensor

#     def _load_images(self, indices: List[int]) -> List[torch.Tensor]:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#             images = list(executor.map(self.__getitem__, indices))
#         return images

#     def get_batch(self, indices: List[int]) -> List[torch.Tensor]:
#         return self._load_images(indices)


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

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(
            dataset_path=self.dataset_path,
            stage="train",
            image_file_extensions=self.image_file_extensions,
        )
        self.val_dataset = ImageDataset(
            dataset_path=self.dataset_path,
            stage="val",
            image_file_extensions=self.image_file_extensions,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
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
        print(image.shape)  # Should print torch.Size([32, 3, img_height, img_width])
        break

    dataset = ImageDataModule(
        dataset_path=dataset_path, batch_size=32, pin_memory=False
    )
    dataset.setup()

    val_dataloader: DataLoader = dataset.val_dataloader()

    # Example of iterating through the validation data
    for image in val_dataloader:
        print(image.shape)  # Should print torch.Size([32, 3, img_height, img_width])
        break
