"""
Usage:
```bash
python -m jepa_datasets.vjepa_dataset
```
"""

import os
import warnings  # TODO: Change to logger
from pathlib import Path
from typing import List, Literal, Optional, Union
import numpy as np
import pytorch_lightning as pl
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset_utils.video.transforms import make_transforms


# pylint: disable=redefined-outer-name
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
        num_clips: int = 1,
        shuffle: bool = True,
        transform: Optional[transforms.Compose] = None,
        filter_short_videos: bool = False,
        filter_long_videos: int = int(10**9),  # bytes
        max_video_duration: Optional[float] = None,  # seconds
    ):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.shuffle = shuffle
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
        if self.shuffle:
            np.random.shuffle(self.video_paths)  # In place shuffle

    def __len__(self) -> int:
        return len(self.video_paths)

    def extract_video_clips(self, video: np.array) -> List[np.array]:
        """Split video into a list of clips with consistent frame lengths"""
        # Calculate the total number of full clips we can extract
        num_frames: int = video.shape[0]
        num_full_clips: int = num_frames // self.frames_per_clip
        # Only keep as many full clips as possible
        full_clips: List[np.array] = [
            video[i * self.frames_per_clip : (i + 1) * self.frames_per_clip]
            for i in range(num_full_clips)
        ]
        # If you want to limit the number of clips to self.num_clips
        return full_clips[: self.num_clips]

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        video_path: Path = self.video_paths[index]
        if os.path.getsize(video_path) > self.filter_long_videos:
            # Retry with a different video if this one is too large/long
            info: str = f"Skipping large/long video: {video_path.name}"
            warnings.warn(info)
            return self.__getitem__(np.random.randint(self.__len__()))
        batch_frames: np.array = self.load_batch_frames(video_path)
        if len(batch_frames) == 0:
            # Retry with a different video if this one fails
            info: str = f"Failed to load batch frames for video: {video_path.name=}"
            warnings.warn(info)
            return self.__getitem__(np.random.randint(self.__len__()))
        # Ensure the video has enough frames for the desired number of clips
        if len(batch_frames) < self.frames_per_clip * self.num_clips:
            info: str = f"Skipping short video: {video_path.name=}"
            warnings.warn(info)
            return self.__getitem__(np.random.randint(self.__len__()))
        # Extract clips
        batch_clips: List[np.array] = self.extract_video_clips(batch_frames)
        # Apply the transformation to each clip
        batch_clips: List[torch.Tensor] = [self.transform(clip) for clip in batch_clips]
        return batch_clips

    def load_batch_frames(self, video_path: Path) -> np.array:
        try:
            vr: VideoReader = VideoReader(str(video_path), ctx=cpu(0))
            clip_len: int = self.frames_per_clip * self.frame_step
            if self.filter_short_videos and len(vr) < clip_len:
                info: str = f"Skipping short video: {video_path.name=}"
                warnings.warn(info)  # TODO: Change to info log
                return np.array([])
            # Calculate the maximum number of frames based on the video duration and frame rate
            max_frames: int = (
                int(self.max_video_duration * vr.get_avg_fps())
                if self.max_video_duration
                else len(vr)
            )
            # Generate indices within the allowed frame range
            indices: np.array = np.arange(0, min(max_frames, len(vr)), self.frame_step)
            batch: np.array = vr.get_batch(indices).asnumpy()
            return batch
        except Exception as e:  # pylint: disable=broad-exception-caught
            info: str = f"Error loading video: {e=}"
            warnings.warn(info)
            return np.array([])


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        batch_size: int = 16,
        frames_per_clip: int = 16,
        frame_step: int = 4,
        num_clips: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = None,
        shuffle: bool = False,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import ToPILImage

    # Define a dummy transform
    dummy_transform = transforms.Lambda(lambda x: x)
    # Initialize transform for tensor to PIL image
    to_pil = ToPILImage()
    dataset_path: Path = Path(
        "/mnt/data/video/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset
    test_vjepa_dataset = VideoDataset(
        dataset_path,
        stage="test",
        # transform=dummy_transform,
    )
    test_vjepa_loader = DataLoader(test_vjepa_dataset, batch_size=32, shuffle=False)
    # Example of iterating through the test data
    for video_clips in test_vjepa_loader:
        print(f"{type(video_clips)=}")
        print(f"{len(video_clips)=}")  # Should print 1
        print(
            f"{video_clips[0].shape=}"
        )  # Should print torch.Size([batch_size, num_channels=3, clip_length, img_height, img_width])
        break
    dataset = VideoDataModule(
        dataset_path=dataset_path, batch_size=32, pin_memory=False, frame_step=10
    )
    dataset.setup()
    test_dataloader: DataLoader = dataset.test_dataloader()
    # Example of iterating through the test data
    for clip_idx, video_clips in enumerate(test_dataloader):
        print(f"{type(video_clips)=}")
        print(f"{len(video_clips)=}")  # Should print 1
        print(
            f"{video_clips[0].shape=}"
        )  # Should print torch.Size([batch_size, num_channels=3, clip_length, img_height, img_width])
        for clip in video_clips[0]:
            clip: torch.Tensor = clip.permute(1, 0, 2, 3)
            print(
                f"{clip.shape=}"
            )  # Shape: (num_channels, clip_length, img_height, img_width)
            # Prepare the plot
            num_frames = clip.size(0)  # Number of frames (clip_length)
            fig, axes = plt.subplots(
                1, num_frames, figsize=(15, 5)
            )  # 1 row, `num_frames` columns
            # Iterate through frames in the clip (over clip_length)
            for frame_idx, (frame, ax) in enumerate(zip(clip, axes)):
                print(f"{frame.shape=}")
                frame_img = to_pil(frame)  # Convert tensor to PIL image
                # Display the frame on its subplot
                ax.imshow(frame_img)
                ax.set_title(f"Frame {frame_idx}")
                ax.axis("off")
            # Adjust layout
            plt.tight_layout()
            plt.show()
            if clip_idx > 5:
                break
