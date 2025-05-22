import os
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from configs import get_video_dataset_config

dataset_config = get_video_dataset_config()

from .video_transforms import VideoTransform, make_transforms


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        stage: Literal["train", "val", "test"],
        frames_per_clip: int,
        frame_step: int,
        num_clips: int,
        shuffle: bool,
        filter_short_videos: bool = dataset_config["FILTER_SHORT_VIDEOS"],
        filter_long_videos: int = dataset_config["MAX_VIDEO_SIZE_B"],  # bytes
        video_file_extensions: Union[str, List[str]] = dataset_config[
            "ACCEPTABLE_FILE_EXTENSIONS"
        ],
        transform: Optional[VideoTransform] = None,
        max_video_duration: Optional[float] = dataset_config[
            "MAX_VIDEO_DURATION"
        ],  # seconds
    ):
        super().__init__()

        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.shuffle = shuffle
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        self.transform = transform or make_transforms()
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
                warnings.warn(info)

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
