# pylint: disable=no-value-for-parameter
import os
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelSummary,
)
from jepa_datasets import VideoDataModule

from utils.types import Number
from pretrain_VJEPA_static_scene import VJEPA
from combine_Loss import TemporalConsistencyLoss


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


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
        self.patch_size = (4, 16, 16)

        # Load the pretrained IJEPA model for video-based architecture
        self.pretrained_model = VJEPA.load_from_checkpoint(self.pretrained_model_path)
        self.pretrained_model.mode = "test"
        self.pretrained_model.phase = "videos"
        self.pretrained_model.layer_dropout = self.drop_path
        self.average_pool = nn.AvgPool1d((self.pretrained_model.embed_dim), stride=1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pretrained_model.embed_dim),
            nn.Linear(self.pretrained_model.embed_dim, np.prod(self.patch_size)),
            nn.Unflatten(
                1,
                (
                    self.pretrained_model.num_frames
                    // self.pretrained_model.tubelet_size,
                    self.pretrained_model.img_size[0]
                    // self.pretrained_model.patch_size[0],
                    self.pretrained_model.img_size[1]
                    // self.pretrained_model.patch_size[1],
                ),
            ),  # Reshape into patches
            LambdaLayer(lambda x: x.permute(0, 4, 1, 2, 3)),
            # nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels=np.prod(
                    self.patch_size
                ),  # Input channels (from the patch size)
                out_channels=3,  # RGB output
                kernel_size=(
                    self.pretrained_model.tubelet_size,
                    self.pretrained_model.patch_size[0],
                    self.pretrained_model.patch_size[1],
                ),  # Same as the Conv3D kernel
                stride=(
                    self.pretrained_model.tubelet_size,
                    self.pretrained_model.patch_size[0],
                    self.pretrained_model.patch_size[1],
                ),  # Same stride as Conv3D
            ),
        )

        # define loss
        self.criterion = nn.MSELoss()

    def forward(self, x, random_t):
        x = self.pretrained_model(
            x=x,
            target_aspect_ratio=self.target_aspect_ratio,
            target_scale=self.target_scale_interval,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=self.context_scale,
            static_scene_temporal_reasoning=False,
            use_static_positional_embedding=False,
        )
        print(f"SHAPE BEFORE MLP: {x.shape=}")
        # x = self.average_pool(x)  # conduct average pool like in paper
        # # new shape = [batch_size, num_patches, 1]
        # x = x.squeeze(-1)  # [batch_size, num_patches]
        x = self.mlp_head(
            x
        )  # [batch_size, output_channels, frame_count, output_height, output_width]
        print(f"SHAPE AFTER MLP: {x.shape=}")

        return x

    def training_step(self, batch, batch_idx):
        clip: torch.Tensor
        running_loss = 0.0
        running_accuracy = 0.0
        for clip in batch:
            y = clip
            x = y[:, :, 0:1, :, :]
            stacked_img = x.repeat(1, 1, self.frame_count, 1, 1)
            print(f"{stacked_img.shape=}")
            y_hat = self(x=stacked_img, random_t=0)
            print(f"{y_hat.shape=}")
            print(f"{y.shape=}")
            save_frames_to_folder(
                video_tensor=y_hat,
                original_tensor=y,
                folder_name="finetune/video",
                batch_idx=batch_idx,
            )
            loss = self.criterion(y_hat, y)  # calculate loss
            accuracy = (
                (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
            )  # calculate accuracy
            running_loss += loss
            running_accuracy += accuracy

        running_accuracy /= len(batch)
        running_loss /= len(batch)
        self.log("train_accuracy", running_accuracy)
        self.log("train_loss", running_loss)
        print("train_accuracy", running_accuracy)
        print("train_loss", running_loss)
        return running_loss

    def validation_step(self, batch, batch_idx):
        clip: torch.Tensor
        running_loss = 0.0
        running_accuracy = 0.0
        for clip in batch:
            y = clip
            x = y[:, :, 0:1, :, :]
            stacked_img = x.repeat(1, 1, self.frame_count, 1, 1)
            print(f"{stacked_img.shape=}")
            y_hat = self(x=stacked_img, random_t=0)
            print(f"{y_hat.shape=}")
            print(f"{y.shape=}")
            save_frames_to_folder(
                video_tensor=y_hat,
                original_tensor=y,
                folder_name="finetune/video",
                batch_idx=batch_idx,
            )
            loss = self.criterion(y_hat, y)  # calculate loss
            accuracy = (
                (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
            )  # calculate accuracy
            running_loss += loss
            running_accuracy += accuracy

        running_accuracy /= len(batch)
        running_loss /= len(batch)
        self.log("val_accuracy", running_accuracy)
        self.log("val_loss", running_loss)
        print("val_accuracy", running_accuracy)
        print("val_loss", running_loss)
        return running_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


def save_frames_to_folder(video_tensor, original_tensor, folder_name, batch_idx):
    # Create folder structure with unique folder names
    if batch_idx % 5000 != 0:
        return

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
        if video_tensor.dim() == 5:
            video_tensor = video_tensor[0]
        if original_tensor.dim() == 5:
            original_tensor = original_tensor[0]
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
    dataset: Path = Path("E:/ahmad/kinetics-dataset/k400").resolve()

    img_size: int = 224
    frame_count: int = 8

    dataset_module = VideoDataModule(
        dataset_path=dataset,
        batch_size=2,
        frames_per_clip=frame_count,
        pin_memory=True,
        prefetch_factor=2,
        frame_step=8,
    )

    model = VJEPA_FT(
        pretrained_model_path="D:/MDX/Thesis/suaijd/jepa/lightning_logs/v-jepa/pretrain/videos/version_1/checkpoints/epoch=1-step=241258.ckpt",
        frame_count=frame_count,
        output_channels=3,
        output_height=img_size,
        output_width=img_size,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/finetune/videos/",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=3,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
        gradient_clip_val=0.1,
    )

    trainer.fit(model, dataset_module)
