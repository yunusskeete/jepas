from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (  # ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger

from jepa_datasets import VideoDataModule
from model import VJEPA

if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("medium")

    dataset_path: Path = Path(
        "E:/ahmad/kinetics-dataset/smaller"
    ).resolve()  # Path to Kinetics dataset

    dataset = VideoDataModule(
        dataset_path=dataset_path,
        batch_size=4,
        frames_per_clip=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    model = VJEPA(
        lr=1e-3,
        num_frames=dataset.frames_per_clip,
        static_scene_temporal_reasoning=True,
    )

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
        # precision="16-true",  # 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16'
        max_epochs=1,
        gradient_clip_val=0.1,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
    )

    # Path to the checkpoint to resume from (use the latest checkpoint if available)
    checkpoint_path: Optional[str] = (
        # "D:/MDX/Thesis/lightning_logs/v-jepa/version_13/checkpoints/epoch=0-step=750.ckpt"
        None
    )

    trainer.fit(model, dataset, ckpt_path=checkpoint_path)

    checkpoint_path: Optional[str] = (
        "D:/MDX/Thesis/lightning_logs/v-jepa/version_16/checkpoints/epoch=0-step=750.ckpt"
        # None
    )

    trainer.validate(model, dataset, ckpt_path=checkpoint_path)
