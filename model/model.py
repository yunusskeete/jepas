import copy
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertEmbeddings

from model.text import TextEncoder
from model.vision.base_model import JEPA_base
from utils.types import Number

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
        Perform a validation step for each image (tensor) in the batch of images (list of tensors).

        Parameters
        ----------
        batch : torch.Tensor
            A tensor representing the batch of data (images).
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The aggregated loss for the batch.
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
        testing_purposes_only: bool = False,
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
        batch: List[torch.Tensor],  # clips
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
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
            )

            # Compute the loss for the current clip
            loss: torch.Tensor = self.criterion(y_student, y_teacher)

            # Accumulate the loss
            running_loss += loss

        # Normalize the running loss by the batch size
        running_loss /= len(batch)

        self.log("train_loss", running_loss)

        return running_loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: List[torch.Tensor],  # clips
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Perform a validation step for each video clip (tensor) in the batch of clips (list of tensors).

        Parameters
        ----------
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
                x=batch,
                target_aspect_ratio=target_aspect_ratio,
                target_scale=target_scale,
                context_aspect_ratio=self.context_aspect_ratio,
                context_scale=context_scale,
            )

            # Compute the loss for the current clip
            loss: torch.Tensor = self.criterion(y_student, y_teacher)

            # Accumulate the loss
            running_loss += loss

        # Normalize the running loss by the batch size
        running_loss /= len(batch)

        self.log("val_loss", running_loss)

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
            A tensor representing the batch of data (video clip).
        batch_idx : int
            Index of the batch.
        dataloader_idx : int
            Index of the dataloader.

        Returns
        -------
        torch.Tensor
            The student embedding of the video clip.
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
        tokenizer: transformers.PreTrainedTokenizer,
        pos_embedding_layer: nn.Module,
        embedder: Union[BertEmbeddings, nn.Embedding],
        encoder: TextEncoder,
        decoder: nn.Module,
        max_length: int = 128,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lr_warmup_fraction: float = 0.01,
        using_pre_tokenized_dataset: bool = False,
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        target_scale_interval: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Union[int, float] = 1,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        m: float = 0.996,  # momentum
        momentum_limits: Tuple[float, float] = (0.996, 1.0),
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=["pos_embedding_layer", "embedder", "encoder", "decoder"]
        )

        # Define parameters
        self.max_length = max_length
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_warmup_fraction = lr_warmup_fraction
        self.using_pre_tokenized_dataset = using_pre_tokenized_dataset
        self.m = m  # momentum
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale_interval = target_scale_interval
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.embed_dim = (
            embedder.word_embeddings.embedding_dim
            if hasattr(embedder, "word_embeddings")
            else embedder.embedding_dim
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        # Optimisation parameters
        self.momentum_limits = momentum_limits
        self.criterion = nn.MSELoss()

        self.tokenizer = tokenizer
        self.pos_embedding_layer = pos_embedding_layer
        self.token_embedder = embedder
        self.encoder: TextEncoder = encoder  # student
        self.decoder = decoder

        self.teacher_encoder: TextEncoder = copy.deepcopy(self.encoder)
        # Unfreeze student
        self.encoder.unfreeze()
        # Freeze teacher (updated via EMA)
        self.teacher_encoder.freeze()

        self.total_steps: Optional[int] = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_loader = self.trainer.datamodule.train_dataloader()
        self.total_steps = len(train_loader) * self.trainer.max_epochs

    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of texts and return the token IDs and attention mask.

        Args:
            texts: A list of strings to tokenize.

        Returns:
            A tuple of two tensors: the first containing the input token IDs and the
            second containing the attention mask.
        """
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return encoding["input_ids"], encoding["attention_mask"]

    def _get_tokens_from_batch(
        self, batch: Union[Dict[str, torch.Tensor], List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a batch of raw texts or pre-tokenized tensors into a pair of tensors
        representing the input token IDs and attention mask.

        Args:
            batch: A list of strings or a dictionary of tensors containing pre-tokenized
                input IDs and attention masks.

        Returns:
            A tuple of two tensors: the first containing the input token IDs and the
            second containing the attention mask.
        """
        if self.using_pre_tokenized_dataset or isinstance(batch, dict):
            token_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

        elif isinstance(batch, list):
            token_ids, attention_mask = self._tokenize_batch(batch)
            token_ids, attention_mask = token_ids.to(self.device), attention_mask.to(
                self.device
            )

        else:
            raise ValueError("Unsupported batch type.")

        return token_ids, attention_mask

    def add_positional_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Update input tokens with positional embeddings.

        Args:
            token_ids: A tensor of shape `(batch_size, seq_len)` containing input tokens.

        Returns:
            A tensor of shape `(batch_size, seq_len, embed_dim)` containing the tokens with positional embeddings.
        """
        return tokens + self.pos_embedding_layer(tokens)

    def _embed_with_positional(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input token IDs with positional embeddings.

        Args:
            token_ids: A tensor of shape `(batch_size, seq_len)` containing input token IDs.

        Returns:
            A tensor of shape `(batch_size, seq_len, embed_dim)` containing the embedded token IDs with positional embeddings.
        """
        embeddings = self.token_embedder(token_ids)
        embeddings = self.add_positional_embedding(embeddings)

        return embeddings

    def get_target_context_probs(self) -> Tuple[float, float]:
        """
        Sample target and context probabilities.
        """
        target_scale = np.random.uniform(
            low=self.target_scale_interval[0], high=self.target_scale_interval[1]
        )
        context_scale = np.random.uniform(
            low=self.context_scale[0], high=self.context_scale[1]
        )

        return target_scale, context_scale

    def sample_target_context_indices(
        self, shape: torch.Size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample exactly a fixed number of target and context indices for each batch item,
        based on the target and context probabilities.

        Args:
            shape (torch.Size): The shape (batch_size, seq_len) for sampling.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                target_indices and context_indices, both of shape (batch_size, seq_len), dtype=bool.
        """
        batch_size, seq_len = shape
        target_scale, context_scale = self.get_target_context_probs()

        num_target: int = max(1, int(seq_len * target_scale))
        num_context: int = max(1, int(seq_len * context_scale))

        target_indices: torch.Tensor = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=self.device
        )
        context_indices: torch.Tensor = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=self.device
        )

        for i in range(batch_size):
            perm: torch.Tensor = torch.randperm(seq_len, device=self.device)

            target_idx: torch.Tensor = perm[:num_target]
            context_idx: torch.Tensor = perm[num_target : num_target + num_context]

            target_indices[i, target_idx] = True
            context_indices[i, context_idx] = True

        return target_indices, context_indices

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Carries out a single forward pass.

        Args:
            token_ids: A tensor of shape `(batch_size, seq_len)` containing input token IDs.
            attention_mask: A tensor of shape `(batch_size, seq_len)` containing the attention mask.

        Returns:
            A dictionary containing the loss, predictions, forecast, and backcast.
        """
        token_embeddings = self._embed_with_positional(token_ids)
        # # TODO: Handle positional embeddings
        # token_embeddings = self.add_positional_embedding(token_ids)
        # # embed_dim = token_embeddings.shape[-1]

        batch_size, seq_len = token_ids.shape
        batch_size, seq_len, embed_dim = token_embeddings.shape

        # Sample targets and context
        target_indices, context_indices = self.sample_target_context_indices(
            token_ids.shape
        )

        target_embeddings = (
            token_embeddings[target_indices]
            .contiguous()
            .view(batch_size, -1, embed_dim)
        )  # (batch_size, num_target_tokens, embed_dim)
        context_embeddings = (
            token_embeddings[context_indices]
            .contiguous()
            .view(batch_size, -1, embed_dim)
        )  # (batch_size, num_context_tokens, embed_dim)

        # Encode
        target_encodings = self.teacher_encoder(
            target_embeddings
        )  # "Ground truth" (Privileged hypothesis)
        context_encodings = self.encoder(
            context_embeddings
        )  # Context (Unprivileged hypothesis)

        # Masked targets
        target_masks = self.mask_token.expand(
            batch_size, target_encodings.shape[1], embed_dim  # num_patches
        )

        # Prepare decoder input
        decoder_input = torch.cat([context_encodings, target_masks], dim=1)

        # Decode
        decoded_output = self.decoder(decoder_input)

        # Take only predictions for target part
        pred_targets = decoded_output[:, -target_masks.shape[1] :, :]

        # Loss: predict target embeddings
        loss = self.criterion(pred_targets, target_embeddings)

        return {
            "loss": loss,
            "context": context_embeddings,
            "targets": target_embeddings,
            "predictions": pred_targets,
        }

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

    def on_after_backward(self) -> None:
        self.update_momentum(self.m)
        self.m += (
            self.momentum_limits[1] - self.momentum_limits[0]
        ) / self.trainer.estimated_stepping_batches

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Union[Dict[str, torch.Tensor], List[str]],
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Perform training step for batch of text sequences.

        Parameters
        ----------
        batch: Union[Dict[str, torch.Tensor], List[str]]
            A list of raw text strings or, if `using_pre_tokenized_dataset` is True, a dictionary
            of tensors containing pre-tokenized input IDs and attention masks.
        batch_idx: int
            Index of the batch.
        dataloader_idx: int
            Index of the dataloader.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        # Extract token IDs and attention masks
        token_ids, attention_mask = self._get_tokens_from_batch(batch)

        # Forward pass
        outputs: Dict[str, torch.Tensor] = self(token_ids, attention_mask)
        loss = outputs["loss"]

        self.log(
            "train_loss",
            loss,
            # prog_bar=True,
            # on_epoch=True,
            # on_step=True,
            # batch_size=len(batch),
        )

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        # Extract token IDs and attention masks
        token_ids, attention_mask = self._get_tokens_from_batch(batch)

        # Forward pass
        outputs: Dict[str, torch.Tensor] = self(token_ids, attention_mask)
        loss = outputs["loss"]

        self.log(
            "val_loss",
            loss,
            # prog_bar=True,
            # on_epoch=True,
            # on_step=True,
            # batch_size=len(batch),
        )

        return loss

    def test_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        # Extract token IDs and attention masks
        token_ids, attention_mask = self._get_tokens_from_batch(batch)

        # Forward pass
        outputs: Dict[str, torch.Tensor] = self(token_ids, attention_mask)
        loss = outputs["loss"]

        self.log(
            "test_loss",
            loss,
            # prog_bar=True,
            # on_epoch=True,
            # on_step=True,
            # batch_size=len(batch),
        )

        return loss

    def predict_step(  # pylint: disable=arguments-differ
        self,
        batch: torch.Tensor,
        batch_idx: int,  # pylint: disable=unused-argument
        dataloader_idx: int = 0,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        self.mode = "test"

        # Get input tokens and embeddings
        token_ids, _ = self._get_tokens_from_batch(batch)
        token_embeddings = self._embed_with_positional(token_ids)

        outputs = {"token_ids": token_ids}

        predictions = self.decoder(token_embeddings)
        outputs["predictions"] = predictions

        return outputs

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
