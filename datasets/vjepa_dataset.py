"""
Usage:

```bash
python -m datasets.vjepa_dataset
```
"""

import os
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset_utils.video.transforms import make_transforms


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        stage: Literal["train", "val", "test"] = "train",
        video_file_extensions: Union[str, List[str]] = [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
        ],
        frames_per_clip: int = 16,
        frame_step: int = 4,
        num_clips=1,
        transform: Optional[transforms.Compose] = None,
        filter_short_videos: bool = False,
        filter_long_videos: int = int(10**9),  # bytes
        max_video_duration: Optional[float] = None,  # seconds
    ):
        super().__init__()

        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        self.transform = transform or make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=[3 / 4, 4 / 3],
            random_resize_scale=[0.3, 1.0],
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=224,
        )
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.max_video_duration = max_video_duration

        if not isinstance(dataset_path, Path):
            dataset_path: Path = Path(dataset_path)

        if not isinstance(video_file_extensions, list):
            video_file_extensions: List[str] = [video_file_extensions]

        # Define the path to the dataset based on the stage
        self.data_path: Path = dataset_path / stage
        # Collect all video paths with the desired extensions
        self.video_paths: List[Path] = []
        for ext in video_file_extensions:
            self.video_paths.extend(self.data_path.rglob(f"*{ext}"))

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Union[torch.Tensor, List[torch.Tensor]]:

        def split_into_clips(video: np.array) -> List[np.array]:
            """Split video into a list of clips"""
            return [
                video[i * self.frames_per_clip : (i + 1) * self.frames_per_clip]
                for i in range(self.num_clips)
            ]

        video_path: Path = self.video_paths[index]

        # Retrieve a random video clip if video clip cannot be retrieved
        while True:
            buffer_np: np.array
            buffer_np, _ = self.load_video_decord(video_path)
            if len(buffer_np) > 0:
                break
            index: int = np.random.randint(self.__len__())
            # pylint: disable=invalid-sequence-index
            video_path: Path = self.video_paths[index]

        if self.num_clips > 1:
            buffer_clips: List[np.array] = split_into_clips(buffer_np)

            if self.transform is not None:
                buffer_clips: List[np.array] = [
                    self.transform(clip) for clip in buffer_clips
                ]

            return buffer_clips

        if self.transform is not None:
            buffer: torch.Tensor = self.transform(buffer_np)

            return buffer

    def load_video_decord(self, video_path: Path) -> Tuple[np.array, np.array]:
        if not os.path.exists(video_path):
            warnings.warn(f"Video path not found: {video_path}")
            return [], None

        video_size = os.path.getsize(video_path)
        if video_size > self.filter_long_videos:
            warnings.warn(f"Skipping long video: {video_path}")
            return [], None

        try:
            vr: VideoReader = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as e:
            warnings.warn(f"Error loading video: {e}")
            return [], None

        max_frames: int = (
            int(self.max_video_duration * vr.get_avg_fps())
            if self.max_video_duration
            else 10**100
        )

        clip_len = self.frames_per_clip * self.frame_step
        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"Skipping short video: {video_path}")
            return [], None

        indices: np.array = np.arange(0, clip_len, self.frame_step)
        indices: np.array = np.clip(indices, 0, min(len(vr) - 1, max_frames))
        buffer: np.array = vr.get_batch(indices).asnumpy()

        return buffer, indices


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        batch_size: int = 16,
        frames_per_clip: int = 16,
        num_clips: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = None,
        shuffle: bool = True,
        video_file_extensions: Union[str, List[str]] = [
            ".mp4",
            ".avi",
            ".mpeg",
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
            num_clips=self.num_clips,
        )
        self.val_dataset = VideoDataset(
            dataset_path=self.dataset_path,
            stage="val",
            video_file_extensions=self.video_file_extensions,
            frames_per_clip=self.frames_per_clip,
            num_clips=self.num_clips,
        )
        self.test_dataset = VideoDataset(
            dataset_path=self.dataset_path,
            stage="test",
            video_file_extensions=self.video_file_extensions,
            frames_per_clip=self.frames_per_clip,
            num_clips=self.num_clips,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=self.shuffle,
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
        )


if __name__ == "__main__":
    dataset_path: Path = Path(
        "/mnt/data/video/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    test_vjepa_dataset = VideoDataset(dataset_path, stage="test")

    test_vjepa_loader = DataLoader(test_vjepa_dataset, batch_size=32, shuffle=False)

    # Example of iterating through the test data
    for video_clips in test_vjepa_loader:
        print(f"{len(video_clips)=}")  # Should print 1
        print(
            f"{video_clips[0].shape=}"
        )  # Should print torch.Size([batch_size, num_channels=3, clip_length, img_height, img_width])
        break

    dataset = VideoDataModule(
        dataset_path=dataset_path, batch_size=32, pin_memory=False
    )
    dataset.setup()

    test_dataloader: DataLoader = dataset.test_dataloader()

    # Example of iterating through the test data
    for video_clips in test_dataloader:
        print(f"{len(video_clips)=}")  # Should print 1
        print(
            f"{video_clips[0].shape=}"
        )  # Should print torch.Size([batch_size, num_channels=3, clip_length, img_height, img_width])
        break
