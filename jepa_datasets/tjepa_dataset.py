"""
Usage:
```bash
python -m jepa_datasets.tjepa_dataset
```
"""

from typing import Optional, Tuple

import datasets
import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer

# pylint: disable=redefined-outer-name


class HuggingfaceDatasetWrapper(Dataset):
    def __init__(
        self,
        hf_dataset: datasets.arrow_dataset.Dataset,
        tokeniser: Optional[BertTokenizer] = None,
        max_length: int = 512,
    ):
        """
        Args:
            hf_dataset: Huggingface dataset to wrap.
            tokeniser: Optional tokeniser to tokenize the text data.
            max_length: Maximum length of tokenized sequences (if tokeniser is used).
        """
        self.hf_dataset = hf_dataset
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self) -> int:
        # Return the number of samples in the dataset
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> str | Tuple[torch.Tensor, torch.Tensor]:
        data: str = self.hf_dataset[idx]["text"]

        if not self.tokeniser:
            return data

        encoding: transformers.tokenization_utils_base.BatchEncoding = self.tokeniser(
            data,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return encoding["input_ids"].squeeze(), encoding["attention_mask"].squeeze()


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hf_dataset: datasets.arrow_dataset.Dataset,
        tokeniser: BertTokenizer,
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = None,
        shuffle: bool = True,
    ):
        super().__init__()
        self.tokeniser = tokeniser
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle

        self.text_dataset = HuggingfaceDatasetWrapper(
            hf_dataset=hf_dataset,
            tokeniser=self.tokeniser,
            max_length=self.max_length,
        )

        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None

    def setup(self, stage=None):
        # Step 1: Define the split sizes for train, validation, and test sets
        dataset_size: int = len(self.text_dataset)
        train_size: int = int(0.8 * dataset_size)  # 80% for training
        val_size: int = int(0.1 * dataset_size)  # 10% for validation
        test_size: int = (
            dataset_size - train_size - val_size
        )  # Remaining 10% for testing

        # Step 2: Create indices for each split
        indices = torch.randperm(dataset_size).tolist()  # Shuffle the indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Step 3: Create Subset datasets for each split
        self.train_dataset = Subset(self.text_dataset, train_indices)
        self.val_dataset = Subset(self.text_dataset, val_indices)
        self.test_dataset = Subset(self.text_dataset, test_indices)

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


if __name__ == "__main__":
    from datasets import load_dataset

    hf_dataset: datasets.arrow_dataset.Dataset = load_dataset(
        "Skylion007/openwebtext", split="train"
    )

    tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")

    text_dataset = HuggingfaceDatasetWrapper(
        hf_dataset, tokeniser=tokeniser, max_length=512
    )

    full_tjepa_loader = DataLoader(text_dataset, batch_size=32, shuffle=False)

    # Example of iterating through the full dataset
    for sample in full_tjepa_loader:
        print(f"{len(sample)=}")  # Should print 2 (token_ids, attention_mask)
        print(
            f"{sample[0].shape=}"
        )  # Should print torch.Size([batch_size, max_length])
        break

    dataset = TextDataModule(
        hf_dataset=hf_dataset, tokeniser=tokeniser, batch_size=32, pin_memory=False
    )
    dataset.setup()

    test_dataloader: DataLoader = dataset.test_dataloader()

    # Example of iterating through the test data
    for sample in test_dataloader:
        print(f"{len(sample)=}")  # Should print 2 (token_ids, attention_mask)
        print(
            f"{sample[0].shape=}"
        )  # Should print torch.Size([batch_size, max_length])
        print(f"{sample[0][2:6]=}")
        print(f"{sample[1][2:6]=}")

        break

    # # longest_sample: int = 0
    # # for idx, sample in enumerate(hf_dataset):
    # #     # longest_sample = max(longest_sample, len(sample["text"]))

    # #     # if idx % 1_000_000 == 0:
    # #     #     print(f"{longest_sample=:,}")
    # #     #     print(f"{idx=:,}")
    # #     if len(sample["text"]) == 100_000:
    # #         print(f"{idx=:,}")

    # #         break

    # sample: str = hf_dataset[1606]["text"]
    # print(f"{len(sample)=}")
    # encoding = tokeniser(
    #     sample,
    #     return_tensors="pt",
    #     padding="max_length",
    #     truncation=True,
    #     max_length=10_000,
    # )
    # ids, mask = encoding["input_ids"].squeeze(), encoding["attention_mask"].squeeze()
    # print(f"{ids=}")
    # print(f"{mask=}")

    # # print(f"{longest_sample=:,}")
