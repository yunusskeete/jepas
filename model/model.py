from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils.types import Number

from .base_model import JEPA_base

# pylint: disable=pointless-string-statement


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
        num_target_blocks: int = 4,  # number of distinct target blocks per image
        m: float = 0.996,  # momentum
        momentum_limits: Tuple[float, float] = (0.996, 1.0),
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

        # Optimisation parameters
        self.momentum_limits = momentum_limits
        self.criterion = nn.MSELoss()

    @staticmethod
    def randomly_select_starting_patch_for_block(
        patch_dim: Tuple[int, int],
        block_dim: Tuple[int, int],
        seed: Optional[int] = None,
    ) -> int:
        """
        Randomly selects the patch defining the 2D block's starting position (on a linear index).

        Parameters:
        patch_dim (Tuple[int, int]): A tuple containing the number of patches in each dimension (width and height).
        block_dim (Tuple[int, int]): A tuple containing the number of patches in each dimension (width and height) of the block from which the patch is to be extracted.
        seed (Optional[int]): An optional random seed for reproducibility.

        Returns:
        int: The starting position of the patch within the block, represented as a linear index.

        NOTE:
        Patches are the basic (processing) units of the image (e.g. 16x16 pixels).
        Blocks are larger regions composed of multiple patches.
        In training, the model attempts to understand blocks within an image - ie. context blocks - by processing it one patch at a time,
        and uses this understanding is used to predict the structure and content of (the target blocks within) an image in a more abstract way.

        Linear index coordinates are used to define the starting patch for a block,
        and map 2D pixel coordinates onto a 1D array index (flattened form).
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the random seed for reproducibility

        def random_int(limit: int) -> int:
            return torch.randint(0, limit, (1,)).item()

        num_patches_h, num_patches_w = (
            patch_dim  # The number of patches in each dimension (width and height)
        )
        num_blocks_h, num_blocks_w = (
            block_dim  # The number of patches in each dimension (width and height)
        )

        max_start_index_h: int = num_patches_h - num_blocks_h + 1
        max_start_index_w: int = num_patches_w - num_blocks_w + 1
        assert all(
            (
                num_blocks_h <= num_patches_h,
                num_blocks_w <= num_patches_w,
            )
        ), f"Blocks cannot be smaller than patches along any dimension, but there were more blocks than patches along at least one dimension ({patch_dim=}, {block_dim=})"

        start_index_h: int = random_int(max_start_index_h)
        start_index_w: int = random_int(max_start_index_w)

        # Convert the 2D coordinate to a linear index
        # x1y1, x2y1, x3y1, ...
        # x1y2, x2y2, x3y3, ...
        # ... , ... , ... , ...
        # <--- patch_width --->
        start_index: int = (
            start_index_h
            * num_patches_w  # index of row `start_y` in flattened (1D) form
        ) + start_index_w  # position in row

        return start_index

    @staticmethod
    def generate_target_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        num_target_blocks: int,
    ) -> Tuple[List[List[int]], Set[int]]:
        """
        Generate (spatial) target patches for each 2D target block.

        Args:
            patch_dim (Tuple[int, int]): The number of patches in each dimension (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.

        Returns:
            Tuple[List[List[int]], Set[int]]:
                - target_patches: A list of lists containing indices of patches for each target block.
                - all_patches: A set of all unique patches used in target blocks.
        """
        # Extract the number of patches in each dimension
        num_patches_h, num_patches_w = patch_dim

        # Calculate the number of patches in the target block
        num_patches_block: int = int(num_patches_h * num_patches_w * scale)

        # Calculate the height and width of the target block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = sqrt(num_patches_block/aspect_ratio)
        """
        num_blocks_h: int = int(
            torch.sqrt(torch.tensor(num_patches_block / aspect_ratio))
        )
        num_blocks_w: int = int(aspect_ratio * num_blocks_h)

        block_dim: Tuple[int, int] = num_blocks_h, num_blocks_w

        # Initialize structures to hold target patches and all unique patches
        target_patches: List[List[int]] = []
        all_patches: Set[int] = set()  # Using a set for fast membership checks

        # For each of the target blocks to generate
        for _ in range(num_target_blocks):
            start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
                patch_dim=patch_dim,
                block_dim=block_dim,
            )

            # Initialize list to hold the patches for the target block
            patches: List[int] = []
            # Collect patches within the target block
            for h in range(num_blocks_h):
                for w in range(num_blocks_w):
                    patch_start_position: int = start_patch + h * num_patches_w + w

                    patches.append(patch_start_position)

                    # Only updated if the start position is not already present
                    all_patches.add(patch_start_position)

            # Store the patches for the current target block
            target_patches.append(patches)

        return target_patches, all_patches

    @staticmethod
    def generate_context_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        target_patches_to_exclude: Set[int],
    ) -> List[int]:
        """
        Generate a list of patch indices for the 2D context block, excluding target patches.

        Args:
            patch_dim (Tuple[int, int]): The number of patches in each dimension (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches_to_exclude (Set[int]): Set containing indices of target patches.

        Returns:
            List[int]: A list of patch indices for the context block excluding target patches.
        """
        # Extract the number of patches in each dimension
        num_patches_h, num_patches_w = patch_dim

        # Calculate the number of patches in the context block
        num_patches_block: int = int(num_patches_h * num_patches_w * scale)

        # Calculate the height and width of the context block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = (num_patches_block/aspect_ratio)**.5
        """
        num_blocks_h: int = int(
            torch.sqrt(torch.tensor(num_patches_block / aspect_ratio))
        )
        num_blocks_w: int = int(aspect_ratio * num_blocks_h)

        block_dim: Tuple[int, int] = num_blocks_h, num_blocks_w

        # Randomly select the starting patch for the context block
        start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
            patch_dim=patch_dim,
            block_dim=block_dim,
        )

        # Generate indices for the context block
        h_indices: np.array
        w_indices: np.array
        h_indices, w_indices = np.meshgrid(
            np.arange(num_blocks_h), np.arange(num_blocks_w), indexing="ij"
        )

        linear_indices: np.array = start_patch + (
            h_indices.flatten() * num_patches_w + w_indices.flatten()
        )

        # Exclude target patches
        context_patches: List[int] = np.setdiff1d(
            linear_indices, np.array(target_patches_to_exclude), assume_unique=True
        ).tolist()

        return context_patches

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        target_aspect_ratio: float,
        target_scale: float,
        context_aspect_ratio: Number,
        context_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_patches: List[List[int]]
        all_unique_target_patches: Set[int]
        target_patches, all_unique_target_patches = IJEPA.generate_target_patches(
            patch_dim=self.patch_embed.patch_shape,  # The number of patches in each dimension
            aspect_ratio=target_aspect_ratio,
            scale=target_scale,
            num_target_blocks=self.num_target_blocks,
        )

        context_patches: List[int] = IJEPA.generate_context_patches(
            patch_dim=self.patch_embed.patch_shape,
            aspect_ratio=context_aspect_ratio,
            scale=context_scale,
            target_patches_to_exclude=all_unique_target_patches,
        )

        return self.forward_base(
            x=x,  # (batch_size, channels, img_height, img_width)
            target_patches=target_patches,
            context_patches=context_patches,
        )

    def update_momentum(self, m: float) -> None:
        """
        Update the teacher model parameters using momentum.

        Args:
            m (float): Momentum coefficient for the exponential moving average update.
        """
        # Enable eval mode to disable layers like dropout and batch normalization
        student_model: nn.Module = self.encoder.eval()
        teacher_model: nn.Module = self.teacher_encoder.eval()

        """
        Manual parameter updates:
        Manually update the teacher's parameters using a momentum term, ensuring
        that the teacher's parameters are a smoothed version of the student's parameters,
        thus reducing the noise and fluctuations in the learning process.

        This smoothing provides more consistent and stable targets for the student to learn from,
        increasing training efficacy. Additionally, this decoupling permits more exploration in the
        student without directly affecting the teacher's parameters, preventing the student from
        overfitting to the techer's instantaneous updates.
        """
        # Disable gradient computation
        with torch.no_grad():
            for student_param, teacher_param in zip(
                student_model.parameters(), teacher_model.parameters()
            ):
                teacher_param.data.mul_(other=m).add_(
                    other=student_param.data, alpha=1 - m
                )

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
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
            x=batch,  # (batch_size, channels, img_height, img_width)
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("train_loss", loss)

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
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

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("val_loss", loss)

        return loss

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_
        dataloader_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        # Generate random target and context aspect ratio
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        self.mode = "test"

        return self(  # Return only student embedding using the student (ViT) encoder
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=1,
        )  # (batch_size, num_patches, embed_dim)

    def on_after_backward(self) -> None:
        self.update_momentum(self.m)
        self.m += (
            self.momentum_limits[1] - self.momentum_limits[0]
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
        num_target_blocks: int = 4,  # number of distinct target blocks per image
        m: float = 0.996,  # momentum
        momentum_limits: Tuple[float, float] = (0.996, 1.0),
        num_frames: int = 16,
        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        JEPA_base.__init__(
            self,
            decoder_depth=decoder_depth,
            num_target_blocks=num_target_blocks,
            num_frames=num_frames,
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

        # Optimisation parameters
        self.momentum_limits = momentum_limits
        self.criterion = nn.MSELoss()

    @staticmethod
    def randomly_select_starting_patch_for_block(
        patch_dim: Tuple[int, int, int],
        block_dim: Tuple[int, int, int],
        seed: Optional[int] = None,
    ) -> int:
        """
        Randomly selects the patch defining the 3D block's starting position (on a linear index).

        Parameters:
        patch_dim (Tuple[int, int, int]): A tuple containing the number of patches in each dimension (temporal dimension, width and height).
        block_dim (Tuple[int, int, int]): A tuple containing the number of patches in each dimension (temporal dimension, width and height) of the block from which the patch is to be extracted.
        seed (Optional[int]): An optional random seed for reproducibility.

        Returns:
        int: The starting position of the patch within the block, represented as a linear index.

        NOTE:
        Patches are the basic (processing) units of the video (e.g. num_framesx16x16 pixels).
        Blocks are larger regions composed of multiple patches.
        In training, the model attempts to understand blocks within an video - ie. context blocks - by processing it one patch at a time,
        and uses this understanding is used to predict the structure and content of (the target blocks within) a video in a more abstract way.

        Linear index coordinates are used to define the starting patch for a block,
        and map 3D pixel coordinates onto a 1D array index (flattened form).
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the random seed for reproducibility

        def random_int(limit: int) -> int:
            return torch.randint(0, limit, (1,)).item()

        num_patches_t, num_patches_h, num_patches_w = patch_dim
        num_blocks_t, num_blocks_h, num_blocks_w = block_dim
        assert all(
            (
                num_blocks_t <= num_patches_t,
                num_blocks_h <= num_patches_h,
                num_blocks_w <= num_patches_w,
            )
        ), f"Blocks cannot be smaller than patches along any dimension, but there were more blocks than patches along at least one dimension ({patch_dim=}, {block_dim=})"

        max_start_index_t: int = num_patches_t - num_blocks_t + 1
        max_start_index_h: int = num_patches_h - num_blocks_h + 1
        max_start_index_w: int = num_patches_w - num_blocks_w + 1

        start_index_t: int = random_int(max_start_index_t)
        start_index_h: int = random_int(max_start_index_h)
        start_index_w: int = random_int(max_start_index_w)

        # Convert the 2D coordinate to a linear index
        # x1y1, x2y1, x3y1, ...
        # x1y2, x2y2, x3y3, ...
        # ... , ... , ... , ...
        # <--- patch_width --->
        start_index: int = (
            (
                start_index_t * (num_patches_h * num_patches_w)
            )  # index through temporal dimension
            + (start_index_h * num_patches_w)  # index down rows
            + start_index_w  # index along columns
        )

        return start_index

    @staticmethod
    def generate_target_patches(
        patch_dim: Tuple[int, int, int],
        aspect_ratio: Number,
        scale: Number,
        num_target_blocks: int,
    ) -> Tuple[List[List[int]], Set[int]]:
        """
        Generate (spatio-temporal) target patches for each 3D target block.

        Args:
            patch_dim (Tuple[int, int, int]): The number of patches in each dimension (temporal, height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.

        Returns:
            Tuple[List[List[int]], Set[int]]:
                - target_patches: A list of lists containing indices of patches for each target block.
                - all_patches: A set of all unique patches used in target blocks.
        """
        # Extract the number of patches in each dimension
        num_patches_t, num_patches_h, num_patches_w = patch_dim

        # Calculate the number of patches in the target block
        num_patches_block: int = int(
            num_patches_t * num_patches_h * num_patches_w * scale
        )

        # Calculate the height and width of the target block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = t * h * (w) = t * h * (aspect_ratio * h) = aspect_ratio * t * h**2
        h = sqrt(num_patches_block/(aspect_ratio * t))
        """
        num_blocks_t: int = num_patches_t
        num_blocks_h: int = int(
            torch.sqrt(torch.tensor(num_patches_block / (aspect_ratio * num_blocks_t)))
        )
        num_blocks_w: int = int(aspect_ratio * num_blocks_h)

        block_dim: Tuple[int, int, int] = num_blocks_t, num_blocks_h, num_blocks_w

        # Initialize structures to hold target patches and all unique patches
        target_patches: List[List[int]] = []
        all_patches: Set[int] = set()  # Using a set for fast membership checks

        # For each of the target blocks to generate
        for _ in range(num_target_blocks):
            start_patch: int = VJEPA.randomly_select_starting_patch_for_block(
                patch_dim=patch_dim,
                block_dim=block_dim,
            )

            # Initialize list to hold the patches for the target block
            patches: List[int] = []
            # Collect patches within the target block
            for t in range(num_blocks_t):
                for h in range(num_blocks_h):
                    for w in range(num_blocks_w):
                        patch_start_position: int = (
                            start_patch
                            + (t * (num_patches_h * num_patches_w))
                            + (h * num_patches_w)
                            + w
                        )

                        patches.append(patch_start_position)

                        # Only updated if the start position is not already present
                        all_patches.add(patch_start_position)

            # Store the patches for the current target block
            target_patches.append(patches)

        return target_patches, all_patches

    @staticmethod
    def generate_context_patches(
        patch_dim: Tuple[int, int, int],
        aspect_ratio: Number,
        scale: Number,
        target_patches_to_exclude: Set[int],
    ) -> List[int]:
        """
        Generate a list of patch indices for the 3D context block, excluding target patches.

        Args:
            patch_dim (Tuple[int, int, int]): Dimensions of the patches (temporal, height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches_to_exclude (Set[int]): Set containing indices of target patches.

        Returns:
            List[int]: A list of patch indices for the context block excluding target patches.
        """
        # Extract the number of patches in each dimension
        num_patches_t, num_patches_h, num_patches_w = patch_dim

        # Calculate the number of patches in the context block
        num_patches_block: int = int(
            num_patches_t * num_patches_h * num_patches_w * scale
        )

        # Calculate the height and width of the context block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = t * h * (w) = t * h * (aspect_ratio * h) = aspect_ratio * t * h**2
        h = (num_patches_block/aspect_ratio * t)**.5
        """
        num_blocks_t: int = num_patches_t
        num_blocks_h: int = int(
            torch.sqrt(torch.tensor(num_patches_block / (aspect_ratio * num_patches_t)))
        )
        num_blocks_w: int = int(aspect_ratio * num_blocks_h)

        block_dim: Tuple[int, int] = num_blocks_t, num_blocks_h, num_blocks_w

        # Randomly select the starting patch for the context block
        start_patch: int = VJEPA.randomly_select_starting_patch_for_block(
            patch_dim=patch_dim,
            block_dim=block_dim,
        )

        # Generate indices for the context block
        t_indices: np.array
        h_indices: np.array
        w_indices: np.array
        t_indices, h_indices, w_indices = np.meshgrid(
            np.arange(num_blocks_t),
            np.arange(num_blocks_h),
            np.arange(num_blocks_w),
            indexing="ij",
        )

        linear_indices: np.array = start_patch + (
            t_indices.flatten() * (num_patches_h * num_patches_w)
            + h_indices.flatten() * num_patches_w
            + w_indices.flatten()
        )

        # Exclude target patches
        context_patches: List[int] = np.setdiff1d(
            linear_indices, np.array(target_patches_to_exclude), assume_unique=True
        ).tolist()

        return context_patches

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        target_aspect_ratio: float,
        target_scale: float,
        context_aspect_ratio: Number,
        context_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_patches: List[List[int]]
        all_unique_target_patches: Set[int]
        target_patches, all_unique_target_patches = VJEPA.generate_target_patches(
            patch_dim=self.patch_embed.patch_shape,
            aspect_ratio=target_aspect_ratio,
            scale=target_scale,
            num_target_blocks=self.num_target_blocks,
        )

        context_patches: List[int] = VJEPA.generate_context_patches(
            patch_dim=self.patch_embed.patch_shape,
            aspect_ratio=context_aspect_ratio,
            scale=context_scale,
            target_patches_to_exclude=all_unique_target_patches,
        )

        # x: torch.Tensor = (
        #     x.permute(  # (batch_size, time, channels, img_height, img_width)
        #         0, 2, 1, 3, 4
        #     )
        # )  # (batch_size, channels, time, height, width)

        return self.forward_base(
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
        # Enable eval mode to disable layers like dropout and batch normalization
        student_model: nn.Module = self.encoder.eval()
        teacher_model: nn.Module = self.teacher_encoder.eval()

        """
        Manual parameter updates:
        Manually update the teacher's parameters using a momentum term, ensuring
        that the teacher's parameters are a smoothed version of the student's parameters,
        thus reducing the noise and fluctuations in the learning process.

        This smoothing provides more consistent and stable targets for the student to learn from,
        increasing training efficacy. Additionally, this decoupling permits more exploration in the
        student without directly affecting the teacher's parameters, preventing the student from
        overfitting to the techer's instantaneous updates.
        """
        # Disable gradient computation
        with torch.no_grad():
            for student_param, teacher_param in zip(
                student_model.parameters(), teacher_model.parameters()
            ):
                teacher_param.data.mul_(other=m).add_(
                    other=student_param.data, alpha=1 - m
                )

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
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
            x=batch,  # (batch_size, time, channels, img_height, img_width)
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("train_loss", loss)

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
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

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("val_loss", loss)

        return loss

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        _summary_

        Parameters
        ----------
        batch : torch.Tensor
            _description_
        batch_idx : int
            _description_
        dataloader_idx : int
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        # Generate random target and context aspect ratio
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )

        self.mode = "test"

        return self(  # Return only student embedding using the student (ViT) encoder
            x=batch,
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=1,
        )  # (batch_size, num_patches, embed_dim)

    def on_after_backward(self) -> None:
        self.update_momentum(self.m)
        self.m += (
            self.momentum_limits[1] - self.momentum_limits[0]
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
