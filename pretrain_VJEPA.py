from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (  # ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import VideoDataModule
from model import VJEPA

if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("medium")

    dataset_path: Path = Path(
        "/mnt/data/video/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    dataset = VideoDataModule(
        dataset_path=dataset_path,
        batch_size=8,
        frames_per_clip=16,
        pin_memory=True,
        prefetch_factor=4,
    )

    model = VJEPA(lr=1e-3, num_frames=dataset.frames_per_clip)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa",
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
