from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from configs import get_text_dataset_config, get_text_experiment_config

dataset_config = get_text_dataset_config()

PAD_TOKEN_ID: int = dataset_config["PAD_TOKEN_ID"]


def dynamic_padding_collate(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of data by dynamically padding the sequences to match the
    maximum length of the sequences in the batch.

    Args:
        batch: A list of dictionaries, where each dictionary contains 'input_ids'
            and 'attention_mask' as keys with their corresponding tensor values.

    Returns:
        A dictionary containing padded 'input_ids' and 'attention_mask' tensors.
        The sequences are padded to the length of the longest sequence in the batch.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    # Pad dynamically to max length in batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=PAD_TOKEN_ID
    )

    return {"input_ids": input_ids, "attention_mask": attention_mask}


class PreTokenizedTextDataset:
    """
    Dataset that returns pre-tokenized tensors (input_ids and attention_mask).
    """

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }


if __name__ == "__main__":
    from datasets import load_from_disk
    from torch.utils.data import DataLoader

    experiment_config = get_text_experiment_config()

    # 1. Load full dataset
    full_dataset = load_from_disk(dataset_config["TOKENIZED_DATASET_NAME"])

    # 2. Split into train/val
    splits = full_dataset.train_test_split(
        test_size=dataset_config["TEST_SPLIT"],  # E.g. 2% validation, 98% train
        seed=experiment_config["SEED"],
    )

    # 3. Load datasets
    # Datasets
    train_dataset = PreTokenizedTextDataset(splits["train"])
    val_dataset = PreTokenizedTextDataset(splits["test"])

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=dynamic_padding_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, collate_fn=dynamic_padding_collate
    )

    print(f"{train_loader=}")
    print(f"{val_loader=}")

    # 4. Check dataloader
    for data_batch in train_loader:
        print(f"{data_batch=}")
        break
