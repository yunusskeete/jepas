from typing import Optional

import datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (  # ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BertTokenizer

from jepa_datasets import TextDataModule
from model import TJEPA

if __name__ == "__main__":
    import torch
    from datasets import load_dataset

    torch.set_float32_matmul_precision("medium")

    hf_dataset_name: str = "Skylion007/openwebtext"
    hf_dataset: datasets.arrow_dataset.Dataset = load_dataset(
        hf_dataset_name, split="train"
    )

    tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = TextDataModule(
        hf_dataset=hf_dataset,
        tokeniser=tokeniser,
        max_length=128,
        pin_memory=True,
        prefetch_factor=4,
        # batch_size=24,
    )

    model = TJEPA(
        lr=1e-3,
        # embed_dim=32,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name="t-jepa",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",  # 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16'
        max_epochs=15,
        gradient_clip_val=0.1,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
        val_check_interval=0.1,  # Run validation every 25% of an epoch
    )

    # Path to the checkpoint to resume from (use the latest checkpoint if available)
    checkpoint_path: Optional[str] = None

    trainer.fit(model, dataset, ckpt_path=checkpoint_path)
