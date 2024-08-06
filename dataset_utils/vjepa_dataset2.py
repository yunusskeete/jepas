import os
import pathlib
import warnings
from logging import getLogger
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.types import Number

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


_GLOBAL_SEED = 0
MIN_VIDEO_SIZE = 1 * 1024  # Minimum size of video to avoid hanging issue
logger = getLogger()


class VideoDataset(Dataset):
    def __init__(
        self,
        # data_paths: List[str],
        dataset_path: Union[str, Path],
        frames_per_clip: int = 16,
        frame_step: int = 4,
        num_clips: int = 1,
        transform: Optional[
            transforms.Compose
        ] = None,  # Apply transforms to individual clips within the buffer (e.g. random cropping)
        global_transform: Optional[
            transforms.Compose
        ] = None,  # Apply global transforms to entire buffer (e.g. normalisation) for efficiency gains
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,
        filter_short_videos: bool = False,
        filter_long_videos: int = int(10**9),
        duration: Optional[Number] = None,  # duration in seconds
    ):
        super().__init__()
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        self.global_transform = global_transform or transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),  # Normalize using ImageNet mean and std
            ]
        )
        self.transform = transform or transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
            ]
        )

        # self.data_paths = data_paths

        if not isinstance(dataset_path, Path):
            dataset_path: Path = Path(dataset_path)

        self.dataset_path = dataset_path
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.global_transform = global_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        # Load video paths and labels
        samples: List[str] = []
        labels: List[Any] = []
        self.num_samples_per_dataset: List[int] = []

        self.samples: List[str] = samples
        self.labels = labels

    def __getitem__(self, index):
        sample: str = self.samples[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video: bool = False
        while not loaded_video:
            buffer, clip_indices = self.load_video(sample)  # [T H W 3]

            loaded_video: bool = len(buffer) > 0
            if not loaded_video:
                index: int = np.random.randint(self.__len__())
                sample: str = self.samples[index]

        # Label/annotations for video
        label = self.labels[index]

        # Parse video into frames & apply data augmentations
        if self.global_transform is not None:
            buffer = self.global_transform(buffer)

        buffer = self._split_into_clips(buffer)

        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def __len__(self) -> int:
        return len(self.samples)

    def load_video(
        self, video_path: str
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Load video content using Decord.

        Args:
            video_path (str): Path to the video file.

        Returns:
            Tuple[np.ndarray, Optional[List[np.ndarray]]]: A tuple containing the video frames and indices of clips.
        """
        if not self._is_video_valid(video_path=video_path):
            return np.array([]), None

        try:
            video_reader = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        except Exception:
            return np.array([]), None

        frames_per_clip: int = self.frames_per_clip
        frame_step: int = self.frame_step

        if self.duration is not None:
            frame_step = self._calculate_frame_step(video_reader)

        clip_length: int = int(frames_per_clip * frame_step)

        if self.filter_short_videos and len(video_reader) < clip_length:
            warnings.warn(f"skipping video of length {len(video_reader)}")
            return np.array([]), None

        video_reader.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_length: int = len(video_reader) // self.num_clips

        all_indices: List[Any] = []
        clip_indices: List[np.ndarray] = []
        all_indices, clip_indices = self._sample_clips(
            video_reader=video_reader,
            partition_length=partition_length,
            clip_length=clip_length,
        )

        buffer: np.ndarray = video_reader.get_batch(all_indices).asnumpy()

        return buffer, clip_indices

    def _split_into_clips(self, video):
        """Split video into a list of clips"""
        fpc: int = self.frames_per_clip
        nc: int = self.num_clips

        return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

    def _is_video_valid(self, video_path: str) -> bool:
        if not os.path.exists(video_path):
            warnings.warn(f"video path not found {video_path=}")
            return False

        file_size: int = os.path.getsize(video_path)

        if file_size < MIN_VIDEO_SIZE:  # avoid hanging issue
            warnings.warn(f"video too short {video_path=}")
            return False

        if file_size > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {file_size=} (bytes)")
            return False

        return True

    def _calculate_frame_step(self, video_reader: VideoReader) -> int:
        try:
            fps = video_reader.get_avg_fps()
            return int(self.duration * fps / self.frames_per_clip)
        except Exception as e:
            warnings.warn(str(e))
            return self.frame_step

    def _sample_clips(
        self, video_reader: VideoReader, partition_length: int, clip_length: int
    ) -> Tuple[List[Any], List[np.ndarray]]:
        all_indices: List[Any] = []
        clip_indices: List[np.ndarray] = []

        for i in range(self.num_clips):
            if partition_length > clip_length:
                indices = self._sample_within_segment(i, partition_length, clip_length)
            else:
                indices = self._sample_with_overlap(
                    i, video_reader, partition_length, clip_length
                )
            clip_indices.append(indices)
            all_indices.extend(list(indices))

        return all_indices, clip_indices

    def _sample_within_segment(
        self, i: int, partition_length: int, clip_length: int
    ) -> np.ndarray:
        end_index: int = clip_length
        if self.random_clip_sampling:
            end_index = np.random.randint(clip_length, partition_length)

        start_index: int = end_index - clip_length

        indices: np.ndarray = np.linspace(
            start_index, end_index, num=self.frames_per_clip
        )
        indices = np.clip(indices, start_index, end_index - 1).astype(np.int64)
        indices += i * partition_length

        return indices

    def _sample_with_overlap(
        self, i: int, video_reader: VideoReader, partition_length: int, clip_length: int
    ) -> np.ndarray:
        indices: np.ndarray
        if not self.allow_clip_overlap:
            indices = self._sample_without_overlap(
                i=i, partition_length=partition_length
            )
        else:
            indices = self._sample_with_overlap_logic(
                i=i, video_reader=video_reader, clip_length=clip_length
            )

        return indices

    def _sample_without_overlap(self, i: int, partition_length: int) -> np.ndarray:
        indices: np.ndarray = np.linspace(
            0, partition_length, num=partition_length // self.frame_step
        )
        indices = np.concatenate(
            (
                indices,
                np.ones(self.frames_per_clip - partition_length // self.frame_step)
                * partition_length,
            )
        )
        indices = np.clip(indices, 0, partition_length - 1).astype(np.int64)

        indices += i * partition_length

        return indices

    def _sample_with_overlap_logic(
        self, i: int, video_reader: VideoReader, clip_length: int
    ) -> np.ndarray:
        sample_length: int = min(clip_length, len(video_reader)) - 1

        indices: np.ndarray = np.linspace(
            0, sample_length, num=sample_length // self.frame_step
        )
        indices = np.concatenate(
            (
                indices,
                np.ones(self.frames_per_clip - sample_length // self.frame_step)
                * sample_length,
            )
        )
        indices = np.clip(indices, 0, sample_length - 1).astype(np.int64)

        clip_step = 0
        if len(video_reader) > clip_length:
            clip_step = (len(video_reader) - clip_length) // (self.num_clips - 1)

        indices += i * clip_step

        return indices


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    global_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        global_transform=global_transform,
        transform=transform,
    )

    logger.info("VideoDataset dataset created")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    logger.info("VideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler
