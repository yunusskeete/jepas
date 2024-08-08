from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (  # ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)

from datasets import ImageDataModule
from model import IJEPA

if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("medium")

    dataset_path: Path = Path(
        "./data/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
    ).resolve()  # Path to ImageNet dataset

    dataset = ImageDataModule(
        dataset_path=dataset_path, batch_size=32, pin_memory=False
    )

    model = IJEPA()

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",  # 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16'
        max_epochs=15,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=0.1,
    )

    trainer.fit(model, dataset)
