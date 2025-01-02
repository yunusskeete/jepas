# pylint: disable=no-value-for-parameter
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from einops import rearrange

import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from model.predictor import Predictor

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
        weight_decay: int = 0,
        drop_path: float = 0.1,
        decoder_heads: int = 16,
        decoder_depth: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set learning parameters
        self.weight_decay = weight_decay
        self.pretrained_model = vjepa_model
        self.finetune_vjepa_model = finetune_vjepa_model
        self.frame_count = frame_count
        self.lr = lr
        self.drop_path = drop_path

        self.target_aspect_ratio: float = 0.75
        self.target_scale_interval: float = 0.15
        self.context_aspect_ratio: Number = 1
        self.context_scale: float = 0.85
        self.channels = 3

        self.pretrained_model.mode = "test"
        self.pretrained_model.phase = "static_scene"
        self.pretrained_model.layer_dropout = self.drop_path
        self.pretrained_model.m = 1
        self.pretrained_model.momentum_limits = (1.0, 1.0)

        # Freeze all parameters in the pretrained model (backbone)
        for param in self.pretrained_model.parameters():
            param.requires_grad = True  # Un-Freeze the pretrained model's parameters

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.pretrained_model.patch_embed.patch_shape[0],
                (
                    self.pretrained_model.patch_embed.patch_shape[1]
                    * self.pretrained_model.patch_embed.patch_shape[2]
                ),
                self.pretrained_model.embed_dim,
            )
        )

        self.predictor = Predictor(
            embed_dim=self.pretrained_model.embed_dim,
            num_heads=decoder_heads,
            depth=decoder_depth,
            layer_dropout=self.drop_path,
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
        x = self.pretrained_model(
            x=x,
            target_aspect_ratio=self.target_aspect_ratio,
            target_scale=self.target_scale_interval,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=self.context_scale,
            static_scene_temporal_reasoning=False,
            use_static_positional_embedding=True,
            random_t=random_t,
        )
        # Make masked target from 2-n and add stacked pos embedding to context for 1
        context, target_mask = self.mask_frames(x)
        x = self.predictor(target_masks=target_mask, context_encoding=context)
        #########################

        # NOTE: If finetune_VJEPA is given then use mlp head from that else use our mlp head
        if self.finetune_vjepa_model is not None:
            x = self.finetune_vjepa_model.mlp_head(
                x
            )  # [batch_size, output_channels, frame_count, output_height, output_width]
        else:
            x = self.mlp_head(
                x
            )  # [batch_size, output_channels, frame_count, output_height, output_width]

        return x

    def pseudo_3d_pos_embedding(self, random_t: int = 0):
        """
        Generate pseudo-3D positional embeddings for context and target frames.
        Args:
            random_t (int): Index of the frame to use as context.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Context positional embeddings and target positional embeddings.
        """
        t, h, w = self.pretrained_model.patch_embed.patch_shape

        # Positional embedding for the selected context frame
        single_pos_embedding_slice = self.pos_embedding[:, random_t, :, :]

        # Positional embeddings for the rest of the frames
        rest_frames = torch.cat(
            [
                self.pos_embedding[:, :random_t, :, :],
                self.pos_embedding[:, random_t + 1 :, :, :],
            ],
            dim=1,
        )

        # Expand context frame positional embedding to match all frames
        pos_emb_stacked = single_pos_embedding_slice.unsqueeze(1).repeat(1, t, 1, 1)

        # Reshape embeddings for context and target
        context_pos_embed = pos_emb_stacked
        target_pos_embed = self.pos_embedding

        return context_pos_embed, target_pos_embed

    def mask_frames(
        self, x: torch.Tensor, random_t: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masks all frames except the first one and returns two tensors:
        one for the first frame and another for the masked frames.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, embed_dim].
            random_t (int): Index of the frame to use as context.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Context tensor and target tensor.
        """
        batch_size, num_patches, embed_dim = x.shape

        # Generate positional embeddings
        context_pos_embed, target_pos_embed = self.pseudo_3d_pos_embedding(random_t)

        # Create a binary mask: 1 for masked positions, 0 otherwise
        mask = torch.randint(
            0,
            2,
            (batch_size, target_pos_embed.size(1) * target_pos_embed.size(2)),
            device=x.device,
        ).bool()

        # Expand mask to match the target tensor's shape
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        mask_expanded = rearrange(
            mask_expanded, "b (t n) e -> b t n e", t=target_pos_embed.size(1)
        )

        # Apply mask to the target tensor
        masked_target = torch.where(
            mask_expanded,
            target_pos_embed,
            torch.zeros_like(target_pos_embed),
        )

        x = rearrange(
            x,
            "b (t n) e -> b t n e",
            t=self.pretrained_model.patch_embed.patch_shape[0],
        )

        # Add positional embeddings to the input tensor
        context_tensor = x + context_pos_embed
        target_tensor = masked_target + target_pos_embed
        context_tensor = rearrange(context_tensor, "b t n e -> b (t n) e")
        target_tensor = rearrange(target_tensor, "b t n e -> b (t n) e")

        return context_tensor, target_tensor

    def remake_predicted_tensor(self, context, target, random_t: int = 0):
        t, h, w = self.pretrained_model.patch_embed.patch_shape
        context_reshape = rearrange(
            context,
            "b (t h w) e -> b e t h w",
            t=t,
            h=h,
            w=w,
        )

        context_slice = context_reshape[:, :, random_t, :, :].unsqueeze(2)
        context_remade = rearrange(context_slice, "b e t h w -> b (t h w) e")

        return torch.cat((context_remade, target), dim=1)

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
        self.log("val_accuracy", running_accuracy)
        self.log("val_loss", running_loss)
        return running_loss

    def configure_optimizers(
        self,
    ) -> Dict[str, Union[Callable, Dict[str, Union[str, Callable]]]]:
        optimizer: Callable = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler: Callable = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


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
        "E:/ahmad/kinetics-dataset/extrasmall"
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
    # Load Pretrained TRJEPA model
    ##############################
    # model = VJEPA.load_from_checkpoint(
    #     "D:/MDX/Thesis/new-jepa/jepa/lightning_logs/v-jepa/pretrain/static_scene/version_6/checkpoints/epoch=2-step=90474.ckpt"
    # )
    model = VJEPA(lr=1e-3, num_frames=dataset.frames_per_clip)

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

    finetune_trjepa_path: Optional[str] = (
        "D:/MDX/Thesis/new-jepa/jepa/lightning_logs/v-jepa/finetune/static/version_1/checkpoints/epoch=4-step=625.ckpt"
    )
    if finetune_trjepa_path is not None:
        finetune_model = TRJEPA_FT.load_from_checkpoint(
            finetune_trjepa_path,
            vjepa_model=model,
            finetune_vjepa_model=finetune_vjepa_model,
            frame_count=dataset.frames_per_clip,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    logger = TensorBoardLogger(
        "lightning_logs",
        name="v-jepa/finetune/tr/",
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
