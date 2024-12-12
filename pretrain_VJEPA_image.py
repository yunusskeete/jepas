import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger

from jepa_datasets import VideoDataModule
from model import VJEPA

if __name__ == "__main__":

    torch.set_float32_matmul_precision("medium")

    dataset_path: Path = Path(
        "/mnt/data/video/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    dataset_videos = VideoDataModule(
        dataset_path=dataset_path,
        # batch_size=12,
        batch_size=10,
        frames_per_clip=8,
        frame_step=8,
        pin_memory=True,
        prefetch_factor=4,
        num_clips=1,  # -1
        num_workers=4,
    )

    model = VJEPA(
        lr=1e-3,
        embed_dim=1024,
        enc_depth=24,
        num_heads=16,
        num_frames=dataset_videos.frames_per_clip,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/pretrain/images",
    )

    # Path to the checkpoint to resume from (use the latest checkpoint if available)
    checkpoint_path: Optional[str] = None

    if checkpoint_path is not None:
        model = VJEPA.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Define your checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/VJEPA/pretrain/images/",  # Directory to save checkpoints
        filename="epoch-{epoch:02d}-val_loss-{val_loss:.4f}",  # Naming scheme
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    trainer_images = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",  # 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16'
        max_epochs=6,
        gradient_clip_val=0.1,
        callbacks=[
            lr_monitor,
            model_summary,
            checkpoint_callback,
        ],
        logger=logger,
        val_check_interval=10_000,
    )

    model.phase = "images"

    trainer_images.fit(
        model=model,
        datamodule=dataset_videos,
        ckpt_path=checkpoint_path,
    )
