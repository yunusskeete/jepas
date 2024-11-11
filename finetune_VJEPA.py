# pylint: disable=no-value-for-parameter
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import cv2
from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelSummary,
)

from utils.types import Number
from pretrain_VJEPA import VJEPA


class VJEPADataset(Dataset):
    def __init__(self, root_dir, num_frames=2, transform=None, image_transform=None):
        """
        Args:
            root_dir (str): Root directory containing subfolders of videos.
            num_frames (int): Number of frames to extract from each video.
            transform (callable, optional): Optional transform to be applied on video frames.
            image_transform (callable, optional): Optional transform for the single frame as an image.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.image_transform = image_transform
        self.video_paths = []

        # Traverse the subfolders and collect video paths
        for part_folder in os.listdir(root_dir):
            part_path = os.path.join(root_dir, part_folder)
            if os.path.isdir(part_path):
                for video_file in os.listdir(part_path):
                    if video_file.endswith(".mp4"):
                        self.video_paths.append(os.path.join(part_path, video_file))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Extract frames from the video
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        # Ensure we have enough frames; pad with the last frame if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        # Stack frames as a tensor (C, T, H, W)
        video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        print(f"{video_tensor.shape=}")

        # Select a single frame as an image (e.g., the first frame)
        single_frame = frames[0]
        if self.image_transform:
            single_frame = self.image_transform(single_frame)
        print(f"{single_frame.shape=}")

        return single_frame.unsqueeze(1), video_tensor


class D2VDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        transform,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        num_frames: int = 8,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.num_frames = num_frames

    def setup(self, stage=None):
        self.train_dataset = VJEPADataset(
            root_dir=self.dataset_path,
            num_frames=self.num_frames,
            transform=self.transform,
            image_transform=self.transform,
        )
        self.val_dataset = VJEPADataset(
            root_dir=self.dataset_path,
            num_frames=self.num_frames,
            transform=self.transform,
            image_transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )


class VJEPA_FT(pl.LightningModule):
    def __init__(
        self, pretrained_model_path, frame_count, lr=1e-4, weight_decay=0, drop_path=0.1
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set learning parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.frame_count = frame_count
        self.pretrained_model_path = pretrained_model_path

        self.target_aspect_ratio: float = 0.75
        self.target_scale_interval: float = 0.15
        self.context_aspect_ratio: Number = 1
        self.context_scale: float = 0.85

        # Load the pretrained IJEPA model for video-based architecture
        self.pretrained_model = VJEPA.load_from_checkpoint(
            checkpoint_path=self.pretrained_model_path
        )
        self.pretrained_model.layer_dropout = drop_path

        self.deconv = nn.ConvTranspose3d(
            in_channels=self.pretrained_model.embed_dim,
            out_channels=3,
            kernel_size=(
                self.pretrained_model.tubelet_size,
                self.pretrained_model.patch_size[0],
                self.pretrained_model.patch_size[1],
            ),
            stride=(
                self.pretrained_model.tubelet_size,
                self.pretrained_model.patch_size[0],
                self.pretrained_model.patch_size[1],
            ),
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pretrained_model.embed_dim),
            nn.Linear(
                self.pretrained_model.embed_dim, self.pretrained_model.num_patches
            ),
            nn.Unflatten(
                1,
                (
                    frame_count // self.pretrained_model.tubelet_size,
                    self.pretrained_model.img_size[0]
                    // self.pretrained_model.patch_size[0],
                    self.pretrained_model.img_size[0]
                    // self.pretrained_model.patch_size[0],
                ),
            ),
        )

        # Define loss for video generation (e.g., reconstruction)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # TODO -- Do I have to do the whole patching and encoding here again instead of being able to use the pretrained model?
        x = self.pretrained_model(
            x,
            self.target_aspect_ratio,
            self.target_scale_interval,
            self.context_aspect_ratio,
            self.context_scale,
            static_scene_temporal_reasoning=True,
        )
        # x = self.mlp_head(x)
        return x  # Output shape: (batch, frames, 3, 224, 224)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # TODO -- Do I add mask tokens instead of repeating the same frames and not use static pos emb
        stacked_img = x.repeat(1, 1, 8, 1, 1)
        y_student, target_blocks, context_blocks, target_patches, context_patches = (
            self(stacked_img)
        )

        num_target_blocks, batch_size, num_blocks, embed_dim = y_student.shape
        context_batch_size, context_num_blocks, context_embed_dim = context_blocks.shape

        num_blocks += context_num_blocks

        reconstructed_x = torch.zeros(batch_size, 784, embed_dim).cuda()

        # Place the patches back in the original tensor
        for target_block_idx in range(num_target_blocks):
            target_patches_for_block = target_patches[
                target_block_idx
            ]  # Indices used in the original process

            # Get the corresponding block from target_block
            block = y_student[
                target_block_idx
            ]  # Shape: (batch_size, target_block_size, embed_dim)

            # Place the block patches back into reconstructed_x
            reconstructed_x[:, target_patches_for_block, :] = block

        reconstructed_x[:, context_patches, :] = context_blocks

        print(f"{reconstructed_x.shape=}")

        y_student_reshaped = rearrange(
            reconstructed_x, "b (t h w) e -> b e t h w", t=4, h=14, w=14
        )
        y_student_original = self.deconv(y_student_reshaped)
        save_as_mp4(y_student_original)
        loss = self.criterion(y_student_original, y)
        accuracy = (y_student_original.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("val_loss", loss)
        print(f"TRAIN LOSS: {loss}")
        self.log("train_accuracy", accuracy)
        print(f"TRAIN ACCURACY: {accuracy}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        stacked_img = x.repeat(1, 1, 8, 1, 1)
        y_student, target_blocks, context_blocks, target_patches, context_patches = (
            self(stacked_img)
        )

        num_target_blocks, batch_size, num_blocks, embed_dim = y_student.shape
        context_batch_size, context_num_blocks, context_embed_dim = context_blocks.shape

        num_blocks += context_num_blocks

        reconstructed_x = torch.zeros(batch_size, 784, embed_dim).cuda()

        # Place the patches back in the original tensor
        for target_block_idx in range(num_target_blocks):
            target_patches_for_block = target_patches[
                target_block_idx
            ]  # Indices used in the original process

            # Get the corresponding block from target_block
            block = y_student[
                target_block_idx
            ]  # Shape: (batch_size, target_block_size, embed_dim)

            # Place the block patches back into reconstructed_x
            reconstructed_x[:, target_patches_for_block, :] = block

        reconstructed_x[:, context_patches, :] = context_blocks

        print(f"{reconstructed_x.shape=}")

        y_student_reshaped = rearrange(
            reconstructed_x, "b (t h w) e -> b e t h w", t=4, h=14, w=14
        )
        y_student_original = self.deconv(y_student_reshaped)
        save_as_mp4(y_student_original)
        loss = self.criterion(y_student_original, y)
        accuracy = (y_student_original.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("val_loss", loss)
        print(f"TRAIN LOSS: {loss}")
        self.log("train_accuracy", accuracy)
        print(f"TRAIN ACCURACY: {accuracy}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


def save_as_mp4(video: torch.Tensor):
    video = video[0]  # Now shape is (channels, num_frames, height, width)

    # Reorder to (num_frames, height, width, channels)
    video = video.permute(1, 2, 3, 0)
    video = video.detach().cpu().numpy()
    num_frames, height, width, channels = video.shape
    output_filename = "output_video.mp4"
    fps = 30  # Frames per second

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame_idx in range(num_frames):
        # Ensure the frame is in a format that OpenCV can handle (e.g., uint8)
        frame = (
            (video[frame_idx] * 255).clip(0, 255).astype(np.uint8)
        )  # Scale if needed

        # If the channels are in the wrong order (e.g., PyTorch's RGB format), convert to BGR
        if channels == 3:  # Assuming the format is (H, W, C) with C=3
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame)

    # Release the video writer
    out.release()

    print(f"Video saved as {output_filename}")


if __name__ == "__main__":

    dataset: Path = Path("E:/ahmad/kinetics-dataset/smaller/test").resolve()

    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    dataset_module = D2VDataModule(
        dataset_path=dataset, transform=transform, num_frames=8
    )

    model = VJEPA_FT(
        pretrained_model_path="lightning_logs/v-jepa/version_33/checkpoints/epoch=0-step=1000-v2.ckpt",
        frame_count=8,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # precision=16,
        max_epochs=1,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=0.1,
    )

    trainer.fit(model, dataset_module)
