from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils.types import Number

from .base_model import JEPA_base


class IJEPA(JEPA_base, pl.LightningModule):
    def __init__(
        self,
        decoder_depth: int = 6,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        target_scale_interval: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Number = 1,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        num_target_blocks: int = 4,  # number of different target blocks per image
        m: float = 0.996,  # momentum
        m_start_end: Tuple[float, float] = (0.996, 1.0),  # momentum
        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        JEPA_base.__init__(
            self,
            decoder_depth=decoder_depth,
            num_target_blocks=num_target_blocks,
            **kwargs,
        )
        self.save_hyperparameters()

        # Define hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m  # momentum
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale_interval = target_scale_interval
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale

        # self.num_tokens = (img_size // patch_size) ** 2

        self.m_start_end = m_start_end

        # Define loss
        self.criterion = nn.MSELoss()

    def forward(
        self,
        x: torch.Tensor,
        target_aspect_ratio: float,
        target_scale: float,
        context_aspect_ratio: Number,
        context_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_patches: List[List[int]]
        all_patches: List[int]
        target_patches, all_patches = IJEPA.generate_target_patches(
            patch_dim=self.patch_dim,
            aspect_ratio=target_aspect_ratio,
            scale=target_scale,
            num_target_blocks=self.num_target_blocks,
        )

        context_patches: List[int] = IJEPA.generate_context_patches(
            patch_dim=self.patch_dim,
            aspect_ratio=context_aspect_ratio,
            scale=context_scale,
            target_patches=all_patches,
        )

        # NOTE: The input tensor has a temporal dimension in video mode,
        # therefore the target and context patches should also be temporally expanded in video mode
        return self.forward_base(
            x=x,  # (batch_size, channels, img_height, img_width) if not self.is_video else (batch_size, channels, time, height, width)
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
        student_model: nn.Module = self.encoder.eval()  # student encoder
        teacher_model: nn.Module = self.teacher_encoder.eval()

        # pylint: disable=pointless-string-statement
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
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        (
            y_student,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            y_teacher,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            x=batch,  # (batch_size, img_channels = RGB = 3, img_height, img_width)
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)  # TODO: Shape
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Generate random target and context aspect ratio and scale
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        (
            y_student,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            y_teacher,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)  # TODO: Shape
        self.log("val_loss", loss)

        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
        # Generate random target and context aspect ratio
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        self.mode = "test"  # Set model to test mode, therefore forward pass returns only full embedding using the student encoder

        return self(  # Return only student embedding
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=1,
        )  # (batch_size, num_patches, embed_dim)

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

    @staticmethod
    def randomly_select_starting_patch_for_block(
        patch_width: int,
        patch_height: int,
        block_width: int,
        block_height: int,
        seed: Optional[int] = None,
    ) -> int:
        """
        Randomly selects the starting position (on a linear index) of a patch within a given block.
        NOTE: Linear index coordinates map 2D pixel coordinates onto a 1D array index (flattened form).

        Parameters:
        patch_width (int): The width of the patch.
        patch_height (int): The height of the patch.
        block_width (int): The width of the block from which the patch is to be extracted.
        block_height (int): The height of the block from which the patch is to be extracted.
        seed (Optional[int]): An optional random seed for reproducibility.

        Returns:
        int: The starting position of the patch within the block, represented as a linear index.
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the random seed for reproducibility

        def random_int(limit: int) -> int:
            return torch.randint(0, limit, (1,)).item()

        max_y: int = patch_height - block_height + 1
        max_x: int = patch_width - block_width + 1

        start_y: int = random_int(max_y)
        start_x: int = random_int(max_x)

        # Convert the 2D coordinate to a linear index
        # x1y1, x2y1, x3y1, ...
        # x1y2, x2y2, x3y3, ...
        # ... , ... , ... , ...
        # <--- patch_width --->
        start_index = (
            start_y * patch_width  # index of row `start_y` in flattened (1D) form
        ) + start_x  # position in row

        return start_index

    @staticmethod
    def generate_target_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        num_target_blocks: int,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Generate target patches for each target block.

        Args:
            patch_dim (Tuple[int, int]): Dimensions of the patches (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.

        Returns:
            Tuple[List[List[int]], List[int]]:
                - target_patches: A list of lists containing indices of patches for each target block.
                - all_patches: A list of all unique patches used in target blocks.
        """
        # Extract patch dimensions
        patch_h, patch_w = patch_dim

        # Calculate the number of patches in the target block
        num_patches_block: int = int(patch_h * patch_w * scale)
        # pylint: disable=pointless-string-statement
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = (num_patches_block/aspect_ratio)**.5
        """

        # Calculate the height and width of the target block maintaining the aspect ratio
        block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w: int = int(aspect_ratio * block_h)

        # Initialize lists to hold target patches and all unique patches
        target_patches: List[List[int]] = []
        all_patches: List[int] = []

        # For each of the target blocks to generate
        for _ in range(
            num_target_blocks  # num_target_blocks if not self.is_video else self.tubelet_size * num_target_blocks
            # TODO: Briadcast these target blocks along the temporal dimension in VJEPA
        ):
            start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
                patch_width=patch_w,
                patch_height=patch_h,
                block_width=block_w,
                block_height=block_h,
            )

            # Initialize list to hold the patches for the target block
            patches: List[int] = []
            # Collect patches within the target block
            for i in range(block_h):
                for j in range(block_w):
                    patches.append(start_patch + i * patch_w + j)
                    if start_patch + i * patch_w + j not in all_patches:
                        all_patches.append(start_patch + i * patch_w + j)

            # Store the patches for the current target block
            target_patches.append(patches)

        return target_patches, all_patches

    @staticmethod
    def generate_context_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        target_patches: List[int],
    ) -> List[int]:
        """
        Generate a list of patch indices for the context block, excluding target patches.

        Args:
            patch_dim (Tuple[int, int]): Dimensions of the patches (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches (List[int]): List containing indices of target patches.

        Returns:
            List[int]: A list of patch indices for the context block excluding target patches.
        """
        # Extract patch dimensions
        patch_h, patch_w = patch_dim

        # Calculate the number of patches in the context block
        num_patches_block: int = int(patch_h * patch_w * scale)
        # pylint: disable=pointless-string-statement
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = (num_patches_block/aspect_ratio)**.5
        """

        # Calculate the height and width of the context block maintaining the aspect ratio
        block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w: int = int(aspect_ratio * block_h)

        # Randomly select the starting patch for the context block
        start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
            patch_width=patch_w,
            patch_height=patch_h,
            block_width=block_w,
            block_height=block_h,
        )

        return [
            start_patch + i * patch_w + j
            for i in range(block_h)
            for j in range(block_w)
            if start_patch + i * patch_w + j not in target_patches
        ]


class VJEPA(JEPA_base, pl.LightningModule):
    def __init__(
        self,
        decoder_depth: int = 6,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        target_scale_interval: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Number = 1,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        num_target_blocks: int = 4,  # number of different target blocks per image
        m: float = 0.996,  # momentum
        m_start_end: Tuple[float, float] = (0.996, 1.0),  # momentum
        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        JEPA_base.__init__(
            self,
            decoder_depth=decoder_depth,
            num_target_blocks=num_target_blocks,
            **kwargs,
        )
        self.save_hyperparameters()

        # Define hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m  # momentum
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale_interval = target_scale_interval
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale

        # self.num_tokens = (img_size // patch_size) ** 2

        self.m_start_end = m_start_end

        # Define loss
        self.criterion = nn.MSELoss()

    def forward(
        self,
        x: torch.Tensor,
        target_aspect_ratio: float,
        target_scale: float,
        context_aspect_ratio: Number,
        context_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_patches: List[List[int]]
        all_patches: List[int]
        target_patches, all_patches = VJEPA.generate_target_patches(
            patch_dim=self.patch_dim,
            aspect_ratio=target_aspect_ratio,
            scale=target_scale,
            num_target_blocks=self.num_target_blocks,  # num_target_blocks if not self.is_video else self.tubelet_size * num_target_blocks
        )

        context_patches: List[int] = VJEPA.generate_context_patches(
            patch_dim=self.patch_dim,
            aspect_ratio=context_aspect_ratio,
            scale=context_scale,
            target_patches=all_patches,
        )

        return self.forward_base(
            x=x,  # (batch_size, img_channels = RGB = 3, img_height, img_width)
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
        student_model: nn.Module = self.encoder.eval()  # student encoder
        teacher_model: nn.Module = self.teacher_encoder.eval()

        # pylint: disable=pointless-string-statement
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
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        (
            y_student,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            y_teacher,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            x=batch,  # (batch_size, img_channels = RGB = 3, img_height, img_width)
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)  # TODO: Shape
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Generate random target and context aspect ratio and scale
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        (
            y_student,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            y_teacher,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)  # TODO: Shape
        self.log("val_loss", loss)

        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
        # Generate random target and context aspect ratio
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        self.mode = "test"  # Set model to test mode, therefore forward pass returns only full embedding using the student encoder

        return self(  # Return only student embedding
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=1,
        )  # (batch_size, num_patches, embed_dim)

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

    @staticmethod
    def randomly_select_starting_patch_for_block(
        patch_width: int,
        patch_height: int,
        block_width: int,
        block_height: int,
        seed: Optional[int] = None,
    ) -> int:
        """
        Randomly selects the starting position of a patch within a given block.

        Parameters:
        patch_width (int): The width of the patch.
        patch_height (int): The height of the patch.
        block_width (int): The width of the block from which the patch is to be extracted.
        block_height (int): The height of the block from which the patch is to be extracted.
        seed (Optional[int]): An optional random seed for reproducibility.

        Returns:
        int: The starting position of the patch within the block, represented as a linear index.
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the random seed for reproducibility

        def random_coordinate(limit: int) -> int:
            return torch.randint(0, limit, (1,)).item()

        max_height: int = patch_height - block_height + 1
        max_width: int = patch_width - block_width + 1

        start_patch_height: int = random_coordinate(max_height)
        start_patch_width: int = random_coordinate(max_width)

        return start_patch_height * patch_width + start_patch_width

    @staticmethod
    def generate_target_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        num_target_blocks: int,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Generate target patches for each target block.

        Args:
            patch_dim (Tuple[int, int]): Dimensions of the patches (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.

        Returns:
            Tuple[List[List[int]], List[int]]:
                - target_patches: A list of lists containing indices of patches for each target block.
                - all_patches: A list of all unique patches used in target blocks.
        """
        # Extract patch dimensions
        patch_h, patch_w = patch_dim

        # Calculate the number of patches in the target block
        num_patches_block: int = int(patch_h * patch_w * scale)

        # pylint: disable=pointless-string-statement
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = (num_patches_block/aspect_ratio)**.5
        """
        # Calculate the height and width of the target block maintaining the aspect ratio
        block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w: int = int(aspect_ratio * block_h)

        # Initialize lists to hold target patches and all unique patches
        target_patches: List[List[int]] = []
        all_patches: List[int] = []

        # For each of the target blocks to generate
        for _ in range(
            num_target_blocks  # num_target_blocks if not self.is_video else self.tubelet_size * num_target_blocks
        ):
            start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
                patch_width=patch_w,
                patch_height=patch_h,
                block_width=block_w,
                block_height=block_h,
            )

            # Initialize list to hold the patches for the target block
            patches: List[int] = []
            # Collect patches within the target block
            for i in range(block_h):
                for j in range(block_w):
                    patches.append(start_patch + i * patch_w + j)
                    if start_patch + i * patch_w + j not in all_patches:
                        all_patches.append(start_patch + i * patch_w + j)

            # Store the patches for the current target block
            target_patches.append(patches)

        return target_patches, all_patches

    @staticmethod
    def generate_context_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        target_patches: List[int],
    ) -> List[int]:
        """
        Generate a list of patch indices for the context block, excluding target patches.

        Args:
            patch_dim (Tuple[int, int]): Dimensions of the patches (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches (List[int]): List containing indices of target patches.

        Returns:
            List[int]: A list of patch indices for the context block excluding target patches.
        """
        # Extract patch dimensions
        patch_h, patch_w = patch_dim

        # Calculate the number of patches in the context block
        num_patches_block: int = int(patch_h * patch_w * scale)
        # pylint: disable=pointless-string-statement
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = (num_patches_block/aspect_ratio)**.5
        """

        # Calculate the height and width of the context block maintaining the aspect ratio
        block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w: int = int(aspect_ratio * block_h)

        # Randomly select the starting patch for the context block
        start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
            patch_width=patch_w,
            patch_height=patch_h,
            block_width=block_w,
            block_height=block_h,
        )

        return [
            start_patch + i * patch_w + j
            for i in range(block_h)
            for j in range(block_w)
            if start_patch + i * patch_w + j not in target_patches
        ]
