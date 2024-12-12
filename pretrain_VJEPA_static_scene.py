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
    import time

    time.sleep(60 * 20)

    dataset_path: Path = Path(
        "/mnt/data/video/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    dataset_videos = VideoDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        frames_per_clip=8,
        pin_memory=True,
        prefetch_factor=4,
        frame_step=8,
    )

    model = VJEPA(lr=5e-4, num_frames=dataset_videos.frames_per_clip)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/pretrain/static_scene",
    )

    # Path to the checkpoint to resume from (use the latest checkpoint if available)
    checkpoint_path: Optional[str] = (
        "lightning_logs/v-jepa/pretrain/videos/version_3/checkpoints/epoch=1-step=30156.ckpt"
        # None
    )
    checkpoint_paths: Path = Path(
        "lightning_logs/v-jepa/pretrain/videos/version_3/checkpoints"
    )
    assert checkpoint_paths.exists, f"Checkpoints do not exist: '{checkpoint_paths}'"

    checkpoint: Path = [
        path for path in checkpoint_paths.glob("*.ckpt") if "epoch=2" in path.name
    ][0]
    assert checkpoint.exists, f"Checkpoint does not exist: '{checkpoint}'"
    checkpoint_path = str(checkpoint)

    model = VJEPA.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # print("STARTING STATIC SCENES")

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
