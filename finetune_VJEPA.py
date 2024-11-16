# pylint: disable=no-value-for-parameter
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import cv2
from einops import rearrange
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pytorch_lightning.loggers import TensorBoardLogger


import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelSummary,
)

from utils.types import Number
from pretrain_VJEPA_static_scene import VJEPA


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


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

        try:
            # Extract frames from the video
            while len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            cap.release()
            single_frame: torch.Tensor = (
                torch.arange(start=0, end=3 * 224 * 224, step=1)
                .reshape(3, 224, 224)
                .float()
            )
            video_tensor: torch.Tensor = (
                torch.arange(start=0, end=3 * 8 * 224 * 224, step=1)
                .reshape(3, 8, 224, 224)
                .float()
            )
            return (
                single_frame.unsqueeze(1),
                video_tensor,
            )  # Return None or you can choose to return a default value

        cap.release()

        if len(frames) <= 0:
            single_frame: torch.Tensor = (
                torch.arange(start=0, end=3 * 224 * 224, step=1)
                .reshape(3, 224, 224)
                .float()
            )
            video_tensor: torch.Tensor = (
                torch.arange(start=0, end=3 * 8 * 224 * 224, step=1)
                .reshape(3, 8, 224, 224)
                .float()
            )
            return single_frame.unsqueeze(1), video_tensor

        # Ensure we have enough frames; pad with the last frame if necessary
        while len(frames) < self.num_frames and len(frames) > 0:
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
        self,
        pretrained_model_path,
        output_channels,
        output_height,
        output_width,
        frame_count,
        lr=1e-4,
        weight_decay=0,
        drop_path=0.1,
        num_decoder_layers=6,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set learning parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.frame_count = frame_count
        self.pretrained_model_path = pretrained_model_path
        self.drop_path = drop_path
        self.output_channels = output_channels
        self.output_height = output_height
        self.output_width = output_width

        self.target_aspect_ratio: float = 0.75
        self.target_scale_interval: float = 0.15
        self.context_aspect_ratio: Number = 1
        self.context_scale: float = 0.85
        self.patch_size = (2, 16, 16)

        # Load the pretrained IJEPA model for video-based architecture
        self.pretrained_model = VJEPA.load_from_checkpoint(self.pretrained_model_path)
        self.pretrained_model.mode = "test"
        self.pretrained_model.layer_dropout = self.drop_path

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pretrained_model.embed_dim),
            nn.Linear(self.pretrained_model.embed_dim, np.prod(self.patch_size)),
            nn.Unflatten(
                1,
                (
                    frame_count // self.patch_size[0],
                    output_height // self.patch_size[1],
                    output_width // self.patch_size[2],
                ),
            ),  # Reshape into patches
            LambdaLayer(lambda x: x.permute(0, 4, 1, 2, 3)),
            nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels=np.prod(
                    self.patch_size
                ),  # Input channels (from the patch size)
                out_channels=3,  # RGB output
                kernel_size=(1, 8, 8),  # Same as the Conv3D kernel
                stride=(1, 8, 8),  # Same stride as Conv3D
            ),
        )

        # define loss
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.pretrained_model(
            x,
            self.target_aspect_ratio,
            self.target_scale_interval,
            self.context_aspect_ratio,
            self.context_scale,
            static_scene_temporal_reasoning=True,
        )
        print(f"SHAPE BEFORE MLP: {x.shape=}")
        x = self.mlp_head(x)
        print(f"SHAPE AFTER MLP: {x.shape=}")

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        stacked_img = x.repeat(1, 1, 8, 1, 1)
        print(f"{stacked_img.shape=}")
        y_hat = self(stacked_img)
        print(f"{y_hat.shape=}")
        print(f"{y.shape=}")
        save_frames_to_folder(
            video_tensor=y_hat,
            original_tensor=y,
            folder_name="finetune1",
            batch_idx=batch_idx,
        )
        loss = self.criterion(y_hat, y)  # calculate loss
        accuracy = (
            (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        )  # calculate accuracy
        self.log("train_accuracy", accuracy)
        self.log("train_loss", loss)
        print("train_accuracy", accuracy)
        print("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        stacked_img = x.repeat(1, 1, 8, 1, 1)
        print(f"{stacked_img.shape=}")
        y_hat = self(stacked_img)
        print(f"{y_hat.shape=}")
        print(f"{y.shape=}")
        save_frames_to_folder(
            video_tensor=y_hat,
            original_tensor=y,
            folder_name="finetune1",
            batch_idx=batch_idx,
        )
        loss = self.criterion(y_hat, y)  # calculate loss
        accuracy = (
            (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        )  # calculate accuracy
        self.log("train_accuracy", accuracy)
        self.log("train_loss", loss)
        print("val_accuracy", accuracy)
        print("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


def save_frames_to_folder(video_tensor, original_tensor, folder_name, batch_idx):
    # Create folder structure with unique folder names
    base_folder = f"{folder_name}/{batch_idx}"
    pred_folder = os.path.join(base_folder, "pred")
    target_folder = os.path.join(base_folder, "target")

    # Create directories if they don't exist
    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)

    video_tensor = video_tensor.squeeze(
        0
    )  # Now target_tensor has shape [3, num_frames, height, width]
    original_tensor = original_tensor.squeeze(0)

    video_tensor = video_tensor.cpu()  # Move to CPU if on CUDA
    original_tensor = original_tensor.cpu()
    # Iterate over each frame
    for i in range(
        video_tensor.shape[1]
    ):  # target_tensor.shape[1] is the number of frames
        # Extract frames from target and original tensors
        target_frame = video_tensor[:, i, :, :]  # Shape [3, height, width]
        original_frame = original_tensor[:, i, :, :]  # Shape [3, height, width]

        target_frame = target_frame - target_frame.min()  # Shift the minimum value to 0
        target_frame = target_frame / target_frame.max()  # Normalize to [0, 1]
        target_frame = target_frame * 255  # Scale to [0, 255]
        target_frame = target_frame.byte()

        original_frame = (
            original_frame - original_frame.min()
        )  # Shift the minimum value to 0
        original_frame = original_frame / original_frame.max()  # Normalize to [0, 1]
        original_frame = original_frame * 255  # Scale to [0, 255]
        original_frame = original_frame.byte()

        # Convert the tensor to a PIL Image (from [C, H, W] to [H, W, C])
        target_frame_image = target_frame.permute(
            1, 2, 0
        ).byte()  # Shape [height, width, 3]
        original_frame_image = original_frame.permute(
            1, 2, 0
        ).byte()  # Shape [height, width, 3]

        # Convert to PIL Image and save them in respective subfolders
        target_pil_image = Image.fromarray(
            target_frame_image.numpy()
        )  # Convert to PIL Image
        original_pil_image = Image.fromarray(
            original_frame_image.numpy()
        )  # Convert to PIL Image

        # Save the frames as PNG images
        target_pil_image.save(
            os.path.join(pred_folder, f"frame_{i+1}.png")
        )  # Save target frame
        original_pil_image.save(
            os.path.join(target_folder, f"frame_{i+1}.png")
        )  # Save original frame

    print(
        f"Saved {video_tensor.shape[1]} frames to 'target' and 'original' subfolders under '{base_folder}'."
    )


if __name__ == "__main__":

    torch.cuda.empty_cache()
    dataset: Path = Path("E:/ahmad/kinetics-dataset/smaller/train").resolve()

    img_size: int = 224
    frame_count: int = 8

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    dataset_module = D2VDataModule(
        dataset_path=dataset, transform=transform, num_frames=frame_count
    )

    model = VJEPA_FT(
        pretrained_model_path="D:/MDX/Thesis/suaijd/jepa/lightning_logs/v-jepa/pretrain/static_scene/version_0/checkpoints/epoch=0-step=6000.ckpt",
        frame_count=frame_count,
        output_channels=3,
        output_height=img_size,
        output_width=img_size,
    )
    for name, param in model.pretrained_model.named_parameters():
        print(name, param.requires_grad)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/finetune/",
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
        gradient_clip_val=0.1,
    )

    trainer.fit(model, dataset_module)
