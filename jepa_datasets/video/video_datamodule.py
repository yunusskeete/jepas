from pathlib import Path
from typing import Any, Dict, List, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from configs import get_video_dataset_config

from .video_dataset import VideoDataset

dataset_config = get_video_dataset_config()


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        batch_size: int,
        frames_per_clip: int,
        frame_step: int,
        num_clips: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
        shuffle: bool,
        video_file_extensions: Union[str, List[str]] = dataset_config[
            "ACCEPTABLE_FILE_EXTENSIONS"
        ],
    ):
        super().__init__()

        if not isinstance(dataset_path, Path):
            dataset_path: Path = Path(dataset_path)

        if not isinstance(video_file_extensions, list):
            video_file_extensions: List[str] = [video_file_extensions]

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.video_file_extensions = video_file_extensions

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = VideoDataset(
            dataset_path=self.dataset_path,
            stage="train",
            video_file_extensions=self.video_file_extensions,
            frames_per_clip=self.frames_per_clip,
            frame_step=self.frame_step,
            num_clips=self.num_clips,
            shuffle=self.shuffle,
        )
        self.val_dataset = VideoDataset(
            dataset_path=self.dataset_path,
            stage="val",
            video_file_extensions=self.video_file_extensions,
            frames_per_clip=self.frames_per_clip,
            frame_step=self.frame_step,
            num_clips=self.num_clips,
            shuffle=self.shuffle,
        )
        self.test_dataset = VideoDataset(
            dataset_path=self.dataset_path,
            stage="test",
            video_file_extensions=self.video_file_extensions,
            frames_per_clip=self.frames_per_clip,
            frame_step=self.frame_step,
            num_clips=self.num_clips,
            shuffle=self.shuffle,
        )

    def video_collate_fn(self, batch):
        # Ensures all clips in the batch are of the same size
        batch = [torch.stack(clip) for clip in batch]  # Stack clips into tensors
        batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True
        )  # Pads sequences
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            # collate_fn=self.video_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            # collate_fn=self.video_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            # collate_fn=self.video_collate_fn,
        )


def create_video_datamodule(video_config: Dict[str, Any]) -> VideoDataModule:
    _dataset_config: Dict[str, Any] = video_config["dataset"]
    experiment_config: Dict[str, Any] = video_config["experiment"]

    return VideoDataModule(
        dataset_path=_dataset_config["DATASET_PATH"],
        batch_size=experiment_config["BATCH_SIZE"],
        frames_per_clip=experiment_config["FRAMES_PER_CLIP"],
        frame_step=experiment_config["FRAME_STEP"],
        num_clips=experiment_config["NUM_CLIPS"],
        num_workers=experiment_config["NUM_WORKERS"],
        pin_memory=experiment_config["PIN_MEMORY"],
        persistent_workers=experiment_config["PERSISTENT_WORKERS"],
        prefetch_factor=experiment_config["PREFETCH_FACTOR"],
        shuffle=_dataset_config["SHUFFLE_DATASET"],
    )
