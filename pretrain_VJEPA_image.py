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

    dataset_path: Path = Path(
        "E:/ahmad/kinetics-dataset/smaller"
    ).resolve()  # Path to Kinetics dataset

    dataset_videos = VideoDataModule(
        dataset_path=dataset_path,
        batch_size=4,
        frames_per_clip=8,
        frame_step=8,
        pin_memory=True,
        prefetch_factor=4,
        num_clips=-1,
        num_workers=4,
    )

    model = VJEPA(lr=1e-3, num_frames=dataset_videos.frames_per_clip)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/pretrain/images",
    )

    # Path to the checkpoint to resume from (use the latest checkpoint if available)
    checkpoint_path: Optional[str] = (
        # "D:/MDX/Thesis/lightning_logs/v-jepa/version_13/checkpoints/epoch=0-step=750.ckpt"
        None
    )

    print("STARTING IMAGES")

    trainer_images = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # precision="16-true",  # 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16'
        max_epochs=3,
        gradient_clip_val=0.1,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
    )

    model.phase = "images"

    trainer_images.fit(
        model,
        dataset_videos,
        ckpt_path=checkpoint_path,
    )
