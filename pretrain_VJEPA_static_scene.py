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
        "E:/ahmad/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    dataset_videos = VideoDataModule(
        dataset_path=dataset_path,
        batch_size=1,
        frames_per_clip=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    model = VJEPA(lr=1e-3, num_frames=dataset_videos.frames_per_clip)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/pretrain/static_scene",
    )

    # Path to the checkpoint to resume from (use the latest checkpoint if available)
    checkpoint_path: Optional[str] = (
        "D:/MDX/Thesis/suaijd/jepa/lightning_logs/v-jepa/pretrain/videos/version_0/checkpoints/epoch=0-step=21000.ckpt"
        # None
    )

    model = VJEPA.load_from_checkpoint(checkpoint_path=checkpoint_path)

    print("STARTING STATIC SCENES")

    trainer_static = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=3,
        gradient_clip_val=0.1,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
    )

    model.phase = "static_scene"

    trainer_static.fit(model, dataset_videos)
