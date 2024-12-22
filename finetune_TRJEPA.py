# pylint: disable=no-value-for-parameter
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from jepa_datasets import VideoDataModule
from pretrain_VJEPA_static_scene import VJEPA
from finetune_VJEPA import VJEPA_FT
from utils.types import Number


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class TRJEPA_FT(pl.LightningModule):
    def __init__(
        self,
        vjepa_model: VJEPA,
        finetune_vjepa_model: Optional[VJEPA_FT] = None,
        frame_count: int = 8,
        lr=1e-4,
        weight_decay=0,
        drop_path=0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vjepa_model"])

        # Set learning parameters
        self.weight_decay = weight_decay
        self.pretrained_model = vjepa_model
        self.finetune_vjepa_model = finetune_vjepa_model
        self.frame_count = frame_count
        self.lr = lr
        self.drop_path = drop_path

        self.channels = 3

        self.pretrained_model.mode = "test"
        self.pretrained_model.phase = "static_scene"
        self.pretrained_model.layer_dropout = self.drop_path
        self.pretrained_model.m = 1
        self.pretrained_model.momentum_limits = (1.0, 1.0)

        # Freeze all parameters in the pretrained model (backbone)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False  # Freeze the pretrained model's parameters

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.pretrained_model.embed_dim,
            num_heads=8,  # or suitable number of heads
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pretrained_model.embed_dim),
            nn.Linear(
                self.pretrained_model.embed_dim,
                (
                    self.pretrained_model.tubelet_size
                    * self.pretrained_model.patch_size[0]
                    * self.pretrained_model.patch_size[1]
                    * self.channels
                ),
            ),
            LambdaLayer(lambda x: x.view(x.size(0), -1)),
            LambdaLayer(
                lambda x: x.reshape(
                    x.size(0),
                    self.channels,
                    self.pretrained_model.num_frames,
                    self.pretrained_model.img_size[0],
                    self.pretrained_model.img_size[1],
                )
            ),
        )

        # define loss
        self.criterion = nn.MSELoss()

    def forward(self, x, random_t):
        ###########################
        # NOTE: this is effectively our video encoder
        x = self.pretrained_model.forward(
            x=x,
            target_aspect_ratio=self.pretrained_model.target_aspect_ratio,
            target_scale=self.pretrained_model.target_scale_interval,
            context_aspect_ratio=self.pretrained_model.context_aspect_ratio,
            context_scale=self.pretrained_model.context_scale,
            static_scene_temporal_reasoning=False,
            use_static_positional_embedding=True,
            random_t=random_t,
        )
        context, target = self.mask_frames(x)
        target_prediction = self.pretrained_model.predictor(
            context_encoding=context, target_masks=target
        )
        #########################

        prediction = torch.cat(
            (context, target_prediction), dim=1
        )  # (batch_size, num_context_patches + num_target_patches, embed_dim)

        # NOTE: If finetune_VJEPA is given then use mlp head from that else use our mlp head
        if self.finetune_vjepa_model is not None:
            temporal_output, _ = self.finetune_vjepa_model.temporal_attention(
                prediction, prediction, prediction
            )
            temporal_output = self.finetune_vjepa_model.mlp_head(
                temporal_output
            )  # [batch_size, output_channels, frame_count, output_height, output_width]
        else:
            temporal_output, _ = self.temporal_attention(
                prediction, prediction, prediction
            )
            temporal_output = self.mlp_head(
                temporal_output
            )  # [batch_size, output_channels, frame_count, output_height, output_width]

        return temporal_output

    def mask_frames(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masks all frames except the first one and returns two tensors:
        one for the first frame and another for the masked frames.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, embed_dim].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - first_frame_tensor: Tensor of the first frame with positional embeddings.
                - masked_frame_tensor: Tensor of the masked frames with mask tokens and adjusted positional embeddings.
        """
        batch_size, num_patches, embed_dim = x.shape

        # Calculate patches per frame
        patches_per_frame = int(
            (self.pretrained_model.img_size[0] / self.pretrained_model.patch_size[0])
            * (self.pretrained_model.img_size[1] / self.pretrained_model.patch_size[1])
        )

        # Indices for the first frame and masked frames
        first_frame_indices = torch.arange(patches_per_frame)
        masked_frame_indices = torch.arange(patches_per_frame, num_patches)

        # Extract positional embeddings for the first and masked frames
        first_frame_pos_embedding = self.pretrained_model.pos_embedding[
            :, first_frame_indices, :
        ]
        masked_frame_pos_embedding = (
            self.pretrained_model.pos_embedding[:, masked_frame_indices, :] + 1.0
        )  # Add offset for masked positional embeddings

        # First frame tensor: Keep original tensor values with positional embeddings
        first_frame_tensor = x[:, first_frame_indices, :] + first_frame_pos_embedding

        # Masked frame tensor: Replace with mask token and adjusted positional embeddings
        mask_token_repeated = self.pretrained_model.mask_token.expand(
            batch_size, masked_frame_indices.size(0), embed_dim
        )
        masked_frame_tensor = mask_token_repeated + masked_frame_pos_embedding

        return first_frame_tensor, masked_frame_tensor

    def training_step(self, batch, batch_idx):
        clip: torch.Tensor
        running_loss = 0.0
        running_accuracy = 0.0
        for clip in batch:
            y = clip
            x = y[:, :, 0:1, :, :]  # Get first frame and stack
            stacked_img = x.repeat(1, 1, self.frame_count, 1, 1)
            y_hat = self(x=stacked_img, random_t=0)
            # save every 5000th frame to disk
            save_frames_to_folder(
                video_tensor=y_hat,
                original_tensor=y,
                folder_name="finetune/static",
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
        return running_loss

    def validation_step(self, batch, batch_idx):
        clip: torch.Tensor
        running_loss = 0.0
        running_accuracy = 0.0
        for clip in batch:
            y = clip
            x = y[:, :, 0:1, :, :]
            stacked_img = x.repeat(1, 1, self.frame_count, 1, 1)
            y_hat = self(x=stacked_img, random_t=0)
            save_frames_to_folder(
                video_tensor=y_hat,
                original_tensor=y,
                folder_name="finetune/static",
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
    torch.set_float32_matmul_precision("medium")

    ##############################
    # Load Dataset
    ##############################
    dataset_path: Path = Path(
        "E:/ahmad/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    dataset = VideoDataModule(
        dataset_path=dataset_path,
        batch_size=8,
        frames_per_clip=8,
        num_workers=os.cpu_count() // 2,
        prefetch_factor=4,
        frame_step=8,
        pin_memory=True,
        num_clips=1,
    )

    ##############################
    # Load Pretrained models
    ##############################
    model = VJEPA.load_from_checkpoint(
        "D:/MDX/Thesis/new-jepa/jepa/lightning_logs/v-jepa/pretrain/static_scene/version_6/checkpoints/epoch=2-step=90474.ckpt"
    )

    finetune_vjepa_path: Optional[str] = None
    finetune_vjepa_model: Optional[VJEPA_FT] = None

    if finetune_vjepa_path is not None:
        finetune_vjepa_model = VJEPA_FT.load_from_checkpoint(finetune_vjepa_path)

    ##############################
    # Finetune initialisation
    ##############################
    finetune_model = TRJEPA_FT(
        vjepa_model=model,
        finetune_vjepa_model=finetune_vjepa_model,
        frame_count=dataset.frames_per_clip,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/finetune/static/",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        callbacks=[lr_monitor, model_summary],
        logger=logger,
        gradient_clip_val=0.1,
    )

    trainer.fit(finetune_model, dataset)
