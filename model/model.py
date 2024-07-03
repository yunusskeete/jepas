from typing import Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from dataset_utils import generate_context_patches, generate_target_patches
from utils.types import Number

from .base_model import JEPA_base
from .vit import vit_base, vit_huge, vit_large, vit_nano, vit_small, vit_tiny


class IJEPA(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        vit: Literal[
            vit_nano, vit_tiny, vit_small, vit_base, vit_large, vit_huge
        ] = vit_nano,
        decoder_depth: int = 6,
        lr: float = 1e-6,
        weight_decay: float = 0.05,
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        target_scale_interval: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Number = 1,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        num_target_blocks: int = 4,  # number of different target blocks
        m: float = 0.996,  # momentum
        m_start_end: Tuple[float, float] = (0.996, 1.0),
    ):
        super().__init__()
        self.save_hyperparameters()

        # define models
        self.vision_transformer = vit(img_size=img_size, patch_size=patch_size)
        self.model = JEPA_base(
            vision_transformer=self.vision_transformer,
            pred_depth=decoder_depth,
            num_target_blocks=num_target_blocks,
        )

        # define hyperparameters
        self.num_target_blocks = num_target_blocks
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale_interval = target_scale_interval
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.embed_dim = self.vision_transformer.embed_dim
        self.patch_dim = self.vision_transformer.patch_dim
        self.patch_size = self.vision_transformer.patch_size

        self.num_tokens = (img_size // patch_size) ** 2

        self.m_start_end = m_start_end

        # define loss
        self.criterion = nn.MSELoss()

    def forward(
        self,
        x: torch.Tensor,
        target_aspect_ratio: Number,
        target_scale: Number,
        context_aspect_ratio: Number,
        context_scale: Number,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_patches: List[List[int]]
        all_patches: List[int]
        target_patches, all_patches = generate_target_patches(
            patch_dim=self.patch_dim,
            aspect_ratio=target_aspect_ratio,
            scale=target_scale,
            num_target_blocks=self.num_target_blocks,
        )

        context_patches: List[int] = generate_context_patches(
            patch_dim=self.patch_dim,
            aspect_ratio=context_aspect_ratio,
            scale=context_scale,
            target_patches=all_patches,
        )

        return self.model(
            x=x,
            target_patches=target_patches,
            context_patches=context_patches,
        )

    def update_momentum(self, m: float) -> None:
        """
        Update the teacher model parameters using momentum.

        Args:
            m (float): Momentum coefficient for the exponential moving average update.
        """
        # Disable layers like dropout and batch normalization
        student_model: nn.Module = self.model.vision_transformer.encoder.eval()
        teacher_model: nn.Module = self.model.teacher_encoder.eval()

        """
        Manual parameter updates:
        Manually update the teacher's parameters using a momentum term, ensuring the teacher model's parameters are a smoothed version of the student model's parameters - thus reducing the noise and fluctuations in the learning process.
        This smoothing provides more consistent and stable targets for the student model to learn from, increasing training efficacy.
        Additionally, this decoupling permits more exploration in the student model without directly affecting the teacher model's parameters, preventing the teacher model from overfitting to the student model's instantaneous updates.
        """
        # Disable gradient computation
        with torch.no_grad():
            for student_param, teacher_param in zip(
                student_model.parameters(), teacher_model.parameters()
            ):
                teacher_param.data.mul_(other=m).add_(
                    other=student_param.data, alpha=1 - m
                )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Generate random target and context aspect ratio and scale
        target_aspect_ratio = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self(
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss = self.criterion(y_student, y_teacher)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Generate random target and context aspect ratio and scale
        target_aspect_ratio = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self(
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss = self.criterion(y_student, y_teacher)
        self.log("val_loss", loss)

        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
        # Generate random target and context aspect ratio
        target_aspect_ratio = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        self.model.mode = "test"  # Set model to test mode, therefore forward pass returns only full embedding using the student encoder

        return self(
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=1,
        )  # Return only student embedding

    def on_after_backward(self) -> None:
        self.update_momentum(self.m)
        self.m += (
            self.m_start_end[1] - self.m_start_end[0]
        ) / self.trainer.estimated_stepping_batches

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
