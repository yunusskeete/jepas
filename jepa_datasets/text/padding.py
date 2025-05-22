from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from configs import get_text_dataset_config

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
