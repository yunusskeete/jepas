from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (  # ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import ImageDataModule
from model import IJEPA

if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("medium")

    dataset_path: Path = Path(
        "./data/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
    ).resolve()  # Path to ImageNet dataset

    dataset = ImageDataModule(
        dataset_path=dataset_path,
        batch_size=128,
        pin_memory=True,
    )

    model = IJEPA(lr=4e-3)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name="i-jepa",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",  # 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16'
        max_epochs=15,
        gradient_clip_val=0.1,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
    )

    trainer.fit(model, dataset)
