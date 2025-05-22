"""
Usage:
```bash
python -m jepa_datasets.tjepa_dataset
```
"""

from typing import Dict, List, Union

import torch

from .text.padding import dynamic_padding_collate
from .text.text_datamodule import PreTokenizedTextDataset, TextDataModule, TextDataset

if __name__ == "__main__":
    import gc

    from configs import get_text_dataset_config, get_text_experiment_config

    dataset_config = get_text_dataset_config()
    experiment_config = get_text_experiment_config()
    experiment_config["NUM_WORKERS"] = 0
    experiment_config["PREFETCH_FACTOR"] = None
    experiment_config["PERSISTENT_WORKERS"] = False
    experiment_config["PIN_MEMORY"] = False

    # 1. Load datamodule
    use_pre_tokenized_dataset: bool = dataset_config["USE_PRE_TOKENIZED_DATASET"]
    dataset_id: str = (
        dataset_config["TOKENIZED_DATASET_NAME"]
        if use_pre_tokenized_dataset
        else dataset_config["UNTOKENIZED_DATASET_NAME"]
    )
    datamodule = TextDataModule(
        use_pre_tokenized_dataset=use_pre_tokenized_dataset,
        dataset_id=dataset_id,
        un_tokenized_dataset_split=dataset_config["UNTOKENIZED_DATASET_SPLIT"],
        test_split=dataset_config["TEST_SPLIT"],
        seed=experiment_config["SEED"],
        shuffle=dataset_config["SHUFFLE_DATASET"],
        collate_fn=dynamic_padding_collate,
        batch_size=experiment_config["BATCH_SIZE"],
        num_workers=experiment_config["NUM_WORKERS"],
        pin_memory=experiment_config["PIN_MEMORY"],
        persistent_workers=experiment_config["PERSISTENT_WORKERS"],
        prefetch_factor=experiment_config["PREFETCH_FACTOR"],
        train_fraction=dataset_config["DATASET_TRAIN_FRACTION"],
    )
    print("✅ Datamodule loaded")

    # 2. Load dataset
    datamodule.setup()
    train_dataset: Union[PreTokenizedTextDataset, TextDataset] = (
        datamodule.train_dataset
    )
    val_dataset: Union[PreTokenizedTextDataset, TextDataset] = datamodule.val_dataset
    print("✅ Dataset loaded")

    # 3. Load dataloader
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    print("✅ Dataloader loaded")

    # 4. Print dataloader stats
    print(f"✅ Train dataloader size: {len(train_dataloader):,}")
    print(f"✅ Val dataloader size: {len(val_dataloader):,}")

    # 5. Check dataloader
    sample: Union[Dict[str, torch.Tensor], List[str]] = next(iter(train_dataloader))
    print(f"{len(sample)=}")  # Should print 2 (token_ids, attention_mask)
    print(
        f"{sample['input_ids'].shape=}"
    )  # Should print torch.Size([batch_size, max_length])

    sample: Union[Dict[str, torch.Tensor], List[str]] = next(iter(val_dataloader))
    print(f"{len(sample)=}")  # Should print 2 (token_ids, attention_mask)
    print(
        f"{sample['input_ids'].shape=}"
    )  # Should print torch.Size([batch_size, max_length])
    print("✅ Dataloader check passed")

    # 6. Cleanup
    del train_dataloader
    del val_dataloader
    gc.collect()
