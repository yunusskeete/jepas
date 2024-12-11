import copy
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertTokenizer
from x_transformers import Encoder
from x_transformers.x_transformers import ScaledSinusoidalEmbedding

from utils.types import Number

from .base_model import JEPA_base
from .predictor import Predictor

# pylint: disable=pointless-string-statement

BERT_MODEL_NAME: str = "bert-base-uncased"
PRETRAINED_TEXT_ENCODER: bool = True


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
        testing_purposes_only: bool = False,
        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        JEPA_base.__init__(
            self,
            decoder_depth=decoder_depth,
            num_target_blocks=num_target_blocks,
            **kwargs,
        )
        if not testing_purposes_only:
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
        seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], Set[int]]:
        """
        Generate (spatial) target patches for each 2D target block.

        Args:
            patch_dim (Tuple[int, int]): The number of patches in each dimension (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.
            seed (Optional[int]): An optional random seed for reproducibility.

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

        _target_patches: List[List[int]] = []
        _all_patches: Set[int] = set()  # Using a set for fast membership checks

        # For each of the target blocks to generate
        for target_block_idx in range(num_target_blocks):
            start_patch: int = IJEPA.randomly_select_starting_patch_for_block(
                patch_dim=patch_dim,
                block_dim=block_dim,
                seed=target_block_idx * seed if seed is not None else None,
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

            # Generate all patch indices in block using tensor operations
            h = torch.arange(num_blocks_h)
            w = torch.arange(num_blocks_w)
            hw_grid = torch.cartesian_prod(
                h, w
            )  # Efficiently generates all combinations of h, w

            block_patch_indices = start_patch + (
                hw_grid[:, 0] * num_patches_w + hw_grid[:, 1]
            )

            _target_patches.append(block_patch_indices.tolist())
            _all_patches.update(block_patch_indices.tolist())

        assert len(target_patches) == len(_target_patches)
        assert len(all_patches) == len(_all_patches)

        assert target_patches == _target_patches
        assert all_patches == _all_patches

        return target_patches, all_patches

    @staticmethod
    def generate_context_patches(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        target_patches_to_exclude: Set[int],
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a list of patch indices for the 2D context block, excluding target patches.

        Args:
            patch_dim (Tuple[int, int]): The number of patches in each dimension (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches_to_exclude (Set[int]): Set containing indices of target patches.
            seed (Optional[int]): An optional random seed for reproducibility.

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
            seed=seed,
        )

        context_patches_set: Set[int] = set()
        # Collect patches within the context block
        for h in range(num_blocks_h):
            for w in range(num_blocks_w):
                patch_start_position: int = start_patch + h * num_patches_w + w
                context_patches_set.add(patch_start_position)

        # Exclude the target patches
        context_patches: List[int] = list(
            context_patches_set.difference(target_patches_to_exclude)
        )

        h = torch.arange(num_blocks_h)
        w = torch.arange(num_blocks_w)
        hw_grid = torch.cartesian_prod(h, w)

        _context_patches_tensor: torch.Tensor = start_patch + (
            +hw_grid[:, 0] * num_patches_w + hw_grid[:, 1]
        )

        _context_patches_set = set(_context_patches_tensor.tolist())

        _context_patches: List[int] = list(
            _context_patches_set.difference(target_patches_to_exclude)
        )

        assert len(context_patches) == len(_context_patches)
        assert context_patches == _context_patches

        # print("Assertions passed")

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
        num_frames: int = 8,
        testing_purposes_only: bool = False,
        mid_epoch_savepoint: int = 10000,
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
        if not testing_purposes_only:
            self.save_hyperparameters()

        # Define hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.mid_epoch_savepoint = mid_epoch_savepoint
        self.phase = "videos"
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
        seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], Set[int]]:
        """
        Generate (spatio-temporal) target patches for each 3D target block.

        Args:
            patch_dim (Tuple[int, int, int]): The number of patches in each dimension (temporal, height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.
            seed (Optional[int]): An optional random seed for reproducibility.

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

        _target_patches: List[List[int]] = []
        _all_patches: Set[int] = set()  # Using a set for fast membership checks

        # For each of the target blocks to generate
        for target_block_idx in range(num_target_blocks):
            start_patch: int = VJEPA.randomly_select_starting_patch_for_block(
                patch_dim=patch_dim,
                block_dim=block_dim,
                seed=target_block_idx * seed if seed is not None else None,
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

            # Generate all patch indices in block using tensor operations
            t = torch.arange(num_patches_t)
            h = torch.arange(num_blocks_h)
            w = torch.arange(num_blocks_w)
            thw_grid = torch.cartesian_prod(
                t, h, w
            )  # Efficiently generates all combinations of t, h, w

            block_patch_indices = start_patch + (
                thw_grid[:, 0] * (num_patches_h * num_patches_w)
                + thw_grid[:, 1] * num_patches_w
                + thw_grid[:, 2]
            )

            _target_patches.append(block_patch_indices.tolist())
            _all_patches.update(block_patch_indices.tolist())

        assert len(target_patches) == len(_target_patches)
        assert len(all_patches) == len(_all_patches)

        assert target_patches == _target_patches
        assert all_patches == _all_patches

        # print("Assertions passed")

        return target_patches, all_patches

    @staticmethod
    def generate_context_patches(
        patch_dim: Tuple[int, int, int],
        aspect_ratio: Number,
        scale: Number,
        target_patches_to_exclude: Set[int],
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a list of patch indices for the 3D context block, excluding target patches.

        Args:
            patch_dim (Tuple[int, int, int]): Dimensions of the patches (temporal, height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches_to_exclude (Set[int]): Set containing indices of target patches.
            seed (Optional[int]): An optional random seed for reproducibility.

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
            seed=seed,
        )

        context_patches_set: Set[int] = set()
        # Collect patches within the context block
        for t in range(num_blocks_t):
            for h in range(num_blocks_h):
                for w in range(num_blocks_w):
                    patch_start_position: int = (
                        start_patch
                        + (t * (num_patches_h * num_patches_w))
                        + (h * num_patches_w)
                        + w
                    )

                    context_patches_set.add(patch_start_position)

        # Store the patches for the current context block
        context_patches = list(
            context_patches_set.difference(target_patches_to_exclude)
        )

        t = torch.arange(num_patches_t)
        h = torch.arange(num_blocks_h)
        w = torch.arange(num_blocks_w)
        thw_grid = torch.cartesian_prod(t, h, w)

        _context_patches_tensor: torch.Tensor = start_patch + (
            thw_grid[:, 0] * (num_patches_h * num_patches_w)
            + thw_grid[:, 1] * num_patches_w
            + thw_grid[:, 2]
        )

        _context_patches_set = set(_context_patches_tensor.tolist())

        _context_patches: List[int] = list(
            _context_patches_set.difference(target_patches_to_exclude)
        )

        assert len(context_patches) == len(_context_patches)
        assert context_patches == _context_patches

        # print("Assertions passed")

        return context_patches

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        target_aspect_ratio: float,
        target_scale: float,
        context_aspect_ratio: Number,
        context_scale: float,
        static_scene_temporal_reasoning: bool = False,
        use_static_positional_embedding: bool = False,
        random_t: int = 0,
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

        return self.forward_base(
            x=x,
            target_patches=target_patches,
            context_patches=context_patches,
            static_scene_temporal_reasoning=static_scene_temporal_reasoning,
            use_static_positional_embedding=use_static_positional_embedding,
            random_t=random_t,
        )

    def forward_video(self, batch, static_scene_temporal_reasoning, running_loss):
        clip: torch.Tensor
        for clip in batch:
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
                x=clip,  # (batch_size, time, channels, img_height, img_width)
                target_aspect_ratio=target_aspect_ratio,
                target_scale=target_scale,
                context_aspect_ratio=self.context_aspect_ratio,
                context_scale=context_scale,
                static_scene_temporal_reasoning=static_scene_temporal_reasoning,
            )

            # Compute the loss for the current clip
            loss: torch.Tensor = self.criterion(y_student, y_teacher)

            # Accumulate the loss
            running_loss += loss

        # Normalize the running loss by the batch size
        running_loss /= len(batch)

    def forward_image(self, batch, running_loss, original_clip):
        _, _, num_frames, _, _ = original_clip.shape  # 'time' is the third dimension

        # Generate new videos by repeating each frame
        new_videos = []
        for frame_idx in range(num_frames):
            # Extract the current frame (shape: [batch_size, in_channels, 1, height, width])
            frame = original_clip[
                :, :, frame_idx : frame_idx + 1, :, :
            ]  # Select frame along 'time'

            # Repeat the frame to create a video of original length (shape: [batch_size, in_channels, num_frames, height, width])
            repeated_video = frame.repeat(
                1, 1, num_frames, 1, 1
            )  # Repeat across 'time' dimension
            new_videos.append(repeated_video)

        # Generate random target and context aspect ratio and scale
        target_aspect_ratio: float = np.random.uniform(
            self.target_aspect_ratio[0], self.target_aspect_ratio[1]
        )
        target_scale: float = np.random.uniform(
            low=self.target_scale_interval[0],
            high=self.target_scale_interval[1],
        )
        context_scale: float = np.random.uniform(
            self.context_scale[0], self.context_scale[1]
        )

        # Forward pass with the frame-stacked video clip
        (
            y_student,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            y_teacher,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        ) = self(
            x=new_videos[0],  # Frame-stacked video clip
            target_aspect_ratio=target_aspect_ratio,
            target_scale=target_scale,
            context_aspect_ratio=self.context_aspect_ratio,
            context_scale=context_scale,
            use_static_positional_embedding=True,
        )

        # Compute the loss for the current clip
        loss: torch.Tensor = self.criterion(y_student, y_teacher)

        # Accumulate the loss
        running_loss += loss

        # Normalize the running loss by the total number of processed videos
        running_loss /= len(batch) * num_frames

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
        batch: List[torch.Tensor],  # clips
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Multi-data training step that processes different types of data with different logic.

        Parameters
        ----------
        batch : Any
            A batch of data from the dataloader.
        batch_idx : int
            Index of the batch in the current epoch.
        dataloader_idx : int, optional
            Index of the dataloader (for multiple data sources), by default 0

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
        """
        # Save a checkpoint every N batches (e.g., every 100 batches)
        if batch_idx % self.mid_epoch_savepoint == 0:
            self.save_mid_epoch_checkpoint(batch_idx)

        if self.phase == "images":
            # Logic for the first dataset (e.g., image clips)
            return self.training_step_images(batch, batch_idx)
        elif self.phase == "videos":
            # Logic for the second dataset (e.g., videos)
            return self.training_step_videos(
                batch, batch_idx, static_scene_temporal_reasoning=False
            )
        elif self.phase == "static_scene":
            # Logic for the third dataset (e.g., static scene temporal reasoning)
            return self.training_step_videos(
                batch, batch_idx, static_scene_temporal_reasoning=True
            )
        else:
            raise ValueError(f"Unsupported mode: {self.phase}")

    def training_step_images(
        self,
        batch: List[torch.Tensor],  # clips
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Create new video clips from each frame of the original video, where each new clip
        is a single frame repeated to maintain the original video length.

        Parameters
        ----------
        batch : List[torch.Tensor]
            A list of tensors representing the batch of data (video clips).
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
        """
        running_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        for original_clip in batch:
            self.forward_image(batch, running_loss, original_clip)

            self.log("train_loss", running_loss)
            print(f"IMAGES {running_loss=}")

        return running_loss

    def training_step_videos(
        self,
        batch: List[torch.Tensor],  # clips
        batch_idx: int,
        static_scene_temporal_reasoning: bool = False,
    ) -> torch.Tensor:
        """
        Perform a training step for each video clip (tensor) in the batch of clips (list of tensors).

        Parameters
        ----------
        batch : List[torch.Tensor]
            A list of tensors representing the batch of data (video clips).
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
        """
        # Initialize the running loss to zero
        running_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        self.forward_video(batch, static_scene_temporal_reasoning, running_loss)

        self.log("train_loss", running_loss)
        # if static_scene_temporal_reasoning:
        #     print(f"STATIC SCENE: {running_loss=}")
        # else:
        #     print(f"VIDEOS: {running_loss=}")

        return running_loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Perform a validation step for each video clip (tensor) in the batch of clips (list of tensors).

        Parameters
        ----------
        batch : torch.Tensor
            A tensor representing the batch of data (video clips).
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
        """
        if self.phase == "images":
            # Logic for the first dataset (e.g., image clips)
            return self.val_step_images(batch, batch_idx)
        elif self.phase == "videos":
            # Logic for the second dataset (e.g., videos)
            return self.val_step_videos(
                batch, batch_idx, static_scene_temporal_reasoning=False
            )
        elif self.phase == "static_scene":
            # Logic for the third dataset (e.g., static scene temporal reasoning)
            return self.val_step_videos(
                batch, batch_idx, static_scene_temporal_reasoning=True
            )
        else:
            raise ValueError(f"Unsupported mode: {self.phase}")

    def val_step_images(
        self,
        batch: List[torch.Tensor],  # clips
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Create new video clips from each frame of the original video, where each new clip
        is a single frame repeated to maintain the original video length.

        Parameters
        ----------
        batch : List[torch.Tensor]
            A list of tensors representing the batch of data (video clips).
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
        """
        running_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        for original_clip in batch:
            # The original clip has shape [batch_size, in_channels, time, height, width]
            # original_clip = original_clip[0]
            self.forward_image(batch, running_loss, original_clip)

            self.log("val_loss", running_loss)
            print(f"IMAGES {running_loss=}")

        return running_loss

    def val_step_videos(
        self,
        batch: List[torch.Tensor],  # clips
        batch_idx: int,
        static_scene_temporal_reasoning: bool = False,
    ) -> torch.Tensor:
        """
        Perform a training step for each video clip (tensor) in the batch of clips (list of tensors).

        Parameters
        ----------
        batch : List[torch.Tensor]
            A list of tensors representing the batch of data (video clips).
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
        """
        # Initialize the running loss to zero
        running_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        self.forward_video(batch, static_scene_temporal_reasoning, running_loss)

        self.log("val_loss", running_loss)
        # if static_scene_temporal_reasoning:
        #     print(f"STATIC SCENE: {running_loss=}")
        # else:
        #     print(f"VIDEOS: {running_loss=}")

        return running_loss

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


class TJEPA(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int = 64,
        enc_depth: int = 8,
        num_heads: int = 8,
        layer_dropout: float = 0.0,
        decoder_depth: int = 6,  # TODO: Make underpowered to prevent collapse
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        target_prob_range: Tuple[float, float] = (
            0.15,
            0.35,
        ),  # used to generate the number of distinct target tokens
        m: float = 0.996,  # momentum
        momentum_limits: Tuple[float, float] = (0.996, 1.0),
        **kwargs,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()

        # Define hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m  # momentum
        self.target_prob_range = target_prob_range

        # Optimisation parameters
        self.momentum_limits = momentum_limits
        self.criterion = nn.MSELoss()

        self.tokeniser: BertTokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.vocab_size = self.tokeniser.vocab_size
        assert (
            self.tokeniser.pad_token_id == 0
        ), f"non-zero pad token id received: {self.tokeniser.pad_token_id=}"

        self.embed_dim = embed_dim
        self.end_depth = enc_depth
        self.num_heads = num_heads
        self.layer_dropout = layer_dropout
        self.target_prob_range = target_prob_range

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_dim)

        self.encoder = Encoder(
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth,
            layer_dropout=self.layer_dropout,
        )  # student encoder

        self.teacher_encoder = copy.deepcopy(
            self.encoder
        ).cuda()  # copy student encoder

        self.predictor = Predictor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            depth=decoder_depth,
            layer_dropout=self.layer_dropout,
        )

        self.pos_embedding = ScaledSinusoidalEmbedding(self.embed_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

    @staticmethod
    def generate_target_indices(
        sequence_batch: torch.Tensor,
        # attention_masks: torch.Tensor,
        target_prob_range: Tuple[float, float],
    ) -> List[List[int]]:
        """
        Generate target indices for a 1D sequence.

        Args:
            sequence_batch (torch.Tensor): The sequence of tokens to generate target tokens for.
            attention_masks (torch.Tensor): The attention masks corresponding to the sequence batch.
            target_prob_range (Tuple[float, float]): The range of probabilities of a given token being a target.

        Returns:
            List[List[int]]: A list of lists containing indices of target tokens.
        """
        target_prob: float = np.random.uniform(
            low=target_prob_range[0], high=target_prob_range[1]
        )

        target_indices: List[List[int]] = []

        for sequence in sequence_batch:
            sequence_length: int = torch.count_nonzero(
                sequence
            )  # NOTE: The tokeniser padding token is 0

            num_target_tokens: int = max(int(sequence_length * target_prob), 1)
            # Randomly select 'num_target_tokens' indices from the sequence
            indices: torch.Tensor = torch.randperm(sequence_length)[:num_target_tokens]
            target_indices.append(indices.tolist())

        return target_indices

    @staticmethod
    def generate_context_indices(
        sequence_batch: torch.Tensor,
        # attention_masks: torch.Tensor,
        target_prob_range: Tuple[float, float],
    ) -> List[List[int]]:
        """
        Generate context tokens for a 1D sequence.

        Args:
            sequence_batch (torch.Tensor): The sequence of tokens to generate target tokens for.
            attention_masks (torch.Tensor): The attention masks corresponding to the sequence batch.
            target_prob_range (Tuple[float, float]): The range of probabilities of a given token being a target.

        Returns:
            List[List[int]]: A list of lists containing indices of context tokens.
        """
        context_prob: float = 1 - np.random.uniform(
            low=target_prob_range[0], high=target_prob_range[1]
        )

        context_indices: List[List[int]] = []

        for sequence in sequence_batch:
            sequence_length: int = torch.count_nonzero(
                sequence
            )  # NOTE: The tokeniser padding token is 0

            num_context_tokens: int = max(int(sequence_length * context_prob), 1)
            # Randomly select 'num_context_tokens' indices from the sequence
            indices: torch.Tensor = torch.randperm(sequence_length)[:num_context_tokens]
            context_indices.append(indices.tolist())

        return context_indices

    def forward(
        self,
        x: torch.Tensor,
        # attention_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_indices: List[int] = TJEPA.generate_target_indices(
            sequence_batch=x,
            # attention_masks=attention_masks,
            target_prob_range=self.target_prob_range,
        )

        context_indices: List[int] = TJEPA.generate_context_indices(
            sequence_batch=x,
            # attention_masks=attention_masks,
            target_prob_range=self.target_prob_range,
        )

        x = self.embedding_layer(x)  # (batch_size, seq_length, embed_dim)

        x = x + self.pos_embedding(x)  # (batch_size, seq_length, embed_dim)

        target_embeddings: torch.Tensor = x[
            None, target_indices
        ]  # (batch_size, num_target_tokens, embed_dim)
        context_embeddings: torch.Tensor = x[
            None, context_indices
        ]  # (batch_size, num_context_tokens, embed_dim)

        target_encoding: torch.Tensor = self.teacher_encoder(
            target_embeddings
        )  # (batch_size, num_target_tokens, embed_dim)
        context_encoding: torch.Tensor = self.encoder(
            context_embeddings
        )  # (batch_size, num_context_tokens, embed_dim)

        batch_dim, num_patches, _ = target_embeddings.shape
        target_masks: torch.Tensor = self.mask_token.repeat(batch_dim, num_patches, 1)
        assert target_masks.shape == target_embeddings.shape

        return (
            self.predictor(
                context_encoding=context_encoding,
                target_masks=target_masks,
            ),
            target_encoding,
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
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Perform training step for batch of text sequences.

        Parameters
        ----------
        batch: torch.Tensor
            Batch of text sequences and attention masks to train on.
        batch_idx: int
            Index of the batch.
        dataloader_idx: int
            Index of the dataloader.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        (
            y_student,  # (batch_size, target_block_size, embed_dim)
            y_teacher,  # (batch_size, target_block_size, embed_dim)
        ) = self(
            x=batch[0],  # (batch_size, seq_length)
            # attention_masks=batch[1],  # (batch_size, seq_length)
        )

        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("train_loss", loss)

        return loss


# TODO: validation_step, predict_step, on_after_backward, configure_optimizers
