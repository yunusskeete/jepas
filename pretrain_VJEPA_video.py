import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (  # ModelCheckpoint,
    LearningRateMonitor,
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
        batch_size=16,
        frames_per_clip=8,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        prefetch_factor=4,
        frame_step=8,
        num_clips=-1,
    )

    model = VJEPA(lr=1e-3, num_frames=dataset_videos.frames_per_clip)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/pretrain/videos",
    )

    # Path to the checkpoint to resume from (use the latest checkpoint if available)
    checkpoint_path: Optional[str] = (
        # "D:/MDX/Thesis/suaijd/jepa/lightning_logs/v-jepa/pretrain/images/version_4/checkpoints/epoch=2-step=76500.ckpt"
        None
    )

    mid_epoch_checkpoint_path: Optional[str] = None

    if checkpoint_path is not None:
        model = VJEPA.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # print("STARTING VIDEOS")

    trainer_videos = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=3,
        gradient_clip_val=0.1,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
        profiler="advanced",
    )
    model.phase = "videos"

    trainer_videos.fit(
        model=model, datamodule=dataset_videos, ckpt_path=mid_epoch_checkpoint_path
    )
