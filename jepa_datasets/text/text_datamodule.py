from typing import Any, Callable, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader

from .pre_tokenised_text_dataset import PreTokenizedTextDataset
from .text_dataset import TextDataset


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        use_pre_tokenized_dataset: bool,
        dataset_id: str,
        un_tokenized_dataset_split: Optional[str],
        test_split: float,
        seed: int,
        shuffle: bool,
        collate_fn: Callable[..., Dict[str, torch.Tensor]],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
        train_fraction: float = 1.0,
    ):
        super().__init__()

        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.use_pre_tokenized_dataset = use_pre_tokenized_dataset
        self.dataset_id = dataset_id
        self.un_tokenized_dataset_split = un_tokenized_dataset_split
        self.test_split = test_split
        self.seed = seed
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        self.train_dataset: Union[PreTokenizedTextDataset, TextDataset] = None
        self.val_dataset: Union[PreTokenizedTextDataset, TextDataset] = None

    def prepare_data(self) -> None:
        if self.use_pre_tokenized_dataset:
            load_from_disk(self.dataset_id)
        else:
            load_dataset(self.dataset_id, split=self.un_tokenized_dataset_split)

    def setup(self, stage: Optional[str] = None) -> None:
        full_dataset: Dataset = (
            load_dataset(self.dataset_id, split=self.un_tokenized_dataset_split)
            if not self.use_pre_tokenized_dataset
            else load_from_disk(self.dataset_id)
        )

        splits: DatasetDict = full_dataset.train_test_split(
            test_size=self.test_split,
            seed=self.seed,
        )

        if self.train_fraction < 1.0:
            num_train = int(len(splits["train"]) * self.train_fraction)
            splits["train"] = splits["train"].select(range(num_train))

        dataset_cls = (
            PreTokenizedTextDataset if self.use_pre_tokenized_dataset else TextDataset
        )
        self.train_dataset = dataset_cls(splits["train"])
        self.val_dataset = dataset_cls(splits["test"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


def create_text_datamodule(
    text_config: Dict[str, Any], collate_fn: Callable[..., Dict[str, torch.Tensor]]
) -> TextDataModule:
    dataset_config: Dict[str, Any] = text_config["dataset"]
    experiment_config: Dict[str, Any] = text_config["experiment"]

    use_pre_tokenized_dataset: bool = dataset_config["USE_PRE_TOKENIZED_DATASET"]
    dataset_id: str = (
        dataset_config["TOKENIZED_DATASET_NAME"]
        if use_pre_tokenized_dataset
        else dataset_config["UNTOKENIZED_DATASET_NAME"]
    )

    return TextDataModule(
        collate_fn=collate_fn,
        use_pre_tokenized_dataset=use_pre_tokenized_dataset,
        dataset_id=dataset_id,
        un_tokenized_dataset_split=dataset_config["UNTOKENIZED_DATASET_SPLIT"],
        test_split=dataset_config["TEST_SPLIT"],
        shuffle=dataset_config["SHUFFLE_DATASET"],
        train_fraction=dataset_config["DATASET_TRAIN_FRACTION"],
        seed=experiment_config["SEED"],
        batch_size=experiment_config["BATCH_SIZE"],
        num_workers=experiment_config["NUM_WORKERS"],
        pin_memory=experiment_config["PIN_MEMORY"],
        persistent_workers=experiment_config["PERSISTENT_WORKERS"],
        prefetch_factor=experiment_config["PREFETCH_FACTOR"],
    )
