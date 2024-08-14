import copy
from typing import Any, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from utils.types import Number

from .predictor import Predictor
from .vit import VisionTransformer

# pylint: disable=pointless-string-statement


class JEPA_base(VisionTransformer):
    def __init__(
        self,
        decoder_depth: int,
        num_target_blocks: int = 4,
        mode: Literal["test", "train"] = "train",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_target_blocks = num_target_blocks
        self.mode = mode.lower()

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        self.post_enc_norm_jepa = (
            nn.LayerNorm(self.embed_dim) if self.post_enc_norm else nn.Identity()
        )

        self.teacher_encoder = copy.deepcopy(self.encoder).cuda()  # student encoder

        self.predictor = Predictor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            depth=decoder_depth,
        )

    @torch.no_grad()
    def get_target_blocks(
        self,
        x: torch.Tensor,
        target_patches: List[List[int]],
    ) -> torch.Tensor:
        """
        (Image/video invariant - just ensure that the `target_patches` are right)
        Generate target blocks from the input tensor using a target encoder.

        Args:
            x (torch.Tensor): Input tensor to be processed by the target encoder of shape (batch_size, num_patches, embed_dim).
            target_patches (List[List[int]]): A list of lists containing indices of target patches for each target block.

        Returns:
            torch.Tensor:
                - target_block: A tensor containing the generated target blocks of shape (num_target_blocks, batch_size, target_block_size, embed_dim).
        """
        # Ensure the target encoder is in evaluation mode
        target_encoder = self.teacher_encoder.eval()

        # Encode the input tensor
        x = target_encoder(
            x  # NOTE: `x` already contains positional encoding from `self.forward_vit()` pass
        )  # (batch_size, num_patches, embed_dim), where num_patches = (output_height * output_width) if not self.is_video else (output_t * output_height * output_width)
        x = self.post_enc_norm_jepa(x)  # (batch_size, num_patches, embed_dim)

        # Create a list to hold the target blocks
        target_blocks_list: List[torch.Tensor] = []

        for target_block_idx in range(self.num_target_blocks):
            target_patches_for_block: List[int] = target_patches[target_block_idx]

            # Assign the corresponding encoded values to the target block tensor
            target_blocks_list.append(
                x[
                    :,  # Include batch dim
                    target_patches_for_block,  # Include only this patch
                    :,  # Include all embed dim
                ]
            )

        # Stack the list of tensors along the new dimension (0)
        target_block: torch.Tensor = torch.stack(
            target_blocks_list, dim=0
        )  # (num_target_blocks, batch_size, target_block_size, embed_dim)

        return target_block.cuda()

    def get_context_block(
        self,
        x: torch.Tensor,
        context_patches: List[int],
    ) -> torch.Tensor:
        """
        (Image/video invariant - just ensure that the `context_patches` are right)
        Generate a context block from the input tensor, excluding target patches.

        Args:
            x (torch.Tensor): Input tensor to be processed of shape (batch_size, num_patches, embed_dim).
            context_patches (List[int]): List containing indices of context patches.

        Returns:
            torch.Tensor: A tensor containing the context block with target patches excluded.
        """
        # Return the context block tensor excluding target patches
        context_block: torch.Tensor = x[
            :,  # Include batch dim
            context_patches,  # Include only this patch
            :,  # Include all embed dim
        ]  # (batch_size, num_patches, embed_dim)

        return context_block

    def make_predictions(
        self,
        num_target_blocks: int,
        batch_dim: int,
        num_patches: int,
        embed_dim: int,
        target_patches: List[List[int]],
        context_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """(Image/video invariant)"""
        # Initialize tensor to hold prediction blocks
        prediction_blocks = torch.zeros(
            (num_target_blocks, batch_dim, num_patches, embed_dim)
        ).cuda()

        # Predict each target block separately using the context encoding and mask tokens
        for target_block_idx in range(num_target_blocks):
            """
            `target_masks` will be concatenated with `context_encoding` and decoded using a transformer decoder.
            The self attention mechanism in the decoder is applied to the concatenated sequence in successive blocks,
            capturing the dependencies within the target sequence (initialised, in `target_masks`, as ones),
            yielding an understanding of the context of the current token in relation to all other tokens.
            The prediction blocks are taken as the ouputs of the final decoder block corresponding to the target tokens
            (`target_masks`).
            """
            target_masks: torch.Tensor = self.mask_token.repeat(
                batch_dim, num_patches, 1
            )

            # The target tokens (initialised as `target_masks`) must contain positional information.
            # The `context_encoding` already contains positional encoding from `self.forward_vit()` pass,
            # thus we must add positional embeddings to the targets
            target_pos_embedding = self.pos_embedding[
                :,  # Include batch dim
                target_patches[target_block_idx],  # Include target patch only
                :,  # Include all embed dim
            ]
            target_masks = target_masks + target_pos_embedding

            # Generate prediction for the current target block
            prediction_block = self.predictor(
                context_encoding=context_encoding,
                target_masks=target_masks,
            )  # (batch_size, target_block_size, embed_dim)

            prediction_blocks[target_block_idx] = prediction_block

        return prediction_blocks

    def forward_base(
        self,
        x: torch.Tensor,
        target_patches: List[List[int]],
        context_patches: List[int],
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        (Image/video invariant - just ensure that the target/context patches are right)
        Forward pass for generating predictions and targets within the JEPA architecture.

        Args:
            x (torch.Tensor): Input tensor of shape: (batch_size, channels, img_height, img_width) if not self.is_video else (batch_size, time, channels, img_height, img_width).
            target_patches (List[List[int]]): A list of lists containing indices of patches for each target block.
            context_patches (List[int]): A list of patch indices for the context block excluding target patches.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If self.mode is "test":
                    torch.Tensor: Full embedding tensor from the student encoder.
                - If self.mode is not "test":
                    Tuple[torch.Tensor, torch.Tensor]:
                        - prediction_blocks: Predicted blocks based on the context encoding.
                        - target_blocks: Actual target blocks.
        """
        if self.is_video:
            assert (
                x.dim() == 5
            ), f"Video tensor does not have the correct number of dimensions (5), received tensor of shape '{x.shape}'"

            x: torch.Tensor = x.permute(
                0, 2, 1, 3, 4
            )  # (batch_size, channels, time, height, width)

        test_mode: bool = self.mode == "test"
        """
        Skip (student) encoder (`patch_embed_only=True`) during training
        as patch embeddings are jointly encoded by the student and the teacher
        encoder such that the student can learn to predict the teacher

        NOTE: Positional encoding applied to `x` during `self.forward_vit()`
        """
        x: torch.Tensor = self.forward_vit(x=x, patch_embed_only=not test_mode)
        batch_size, num_patches, embed_dim = (
            x.shape
        )  # where num_patches = (output_height * output_width) if not self.is_video else (output_t * output_height * output_width)

        # If in test mode, return the full embedding using the student encoder
        if test_mode:
            return x  # (batch_size, num_patches, embed_dim)
            
        # If not in test mode, generate target/context blocks, embed those using the student/teacher encoders,
        # and set up JEP (Joint Embedding Prediction) optimisation incentive

        ### Get target embeddings using the target encoder
        target_blocks: torch.Tensor = (
            self.get_target_blocks(  # NOTE: `target_blocks` contain positional information from `x`, which underwent the `self.forward_vit()` pass
                x=x,
                target_patches=target_patches,
            )
        )
        num_target_blocks, batch_size, target_block_size, embed_dim = (
            target_blocks.shape
        )

        ### Get context embeddings excluding the target patches
        context_block: torch.Tensor = (
            self.get_context_block(  # NOTE: `context_block` contains positional information from `x`, which underwent the `self.forward_vit()` pass
                x=x,
                context_patches=context_patches,
            )
        )
        batch_size, num_context_patches, embed_dim = context_block.shape

        context_encoding: torch.Tensor = (
            self.post_enc_norm_jepa(  # NOTE: `context_encoding` contains positional information from `x`, which underwent the `self.forward_vit()` pass
                self.encoder(  # student encoder (ViT)
                    x=x,
                )
            )
        )  # (batch_size, num_context_patches, embed_dim)
        batch_size, num_patches_enc, embed_dim = context_encoding.shape
        assert (
            num_context_patches == num_patches_enc
        ), f"The number of patches in the context_block ({num_context_patches}) does not equal the number of patches in the context_encoding ({num_patches_enc})."

        ### Make predictions using the decoder
        prediction_blocks = self.make_predictions(
            num_target_blocks=num_target_blocks,
            batch_dim=batch_size,
            num_patches=target_block_size,
            embed_dim=embed_dim,
            target_patches=target_patches,
            context_encoding=context_encoding,
        )  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        num_target_blocks, batch_size, num_prediction_blocks, embed_dim = (
            prediction_blocks.shape
        )
        assert (
            target_block_size == num_prediction_blocks
        ), f"The number of patches in the target_block ({target_block_size}) does not equal the number of patches in the prediction_blocks ({num_prediction_blocks})."

        return (
            prediction_blocks,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            target_blocks,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        )

    @staticmethod
    def generate_target_patches_2d(
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

        # Calculate the height and width of the target block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = sqrt(num_patches_block/aspect_ratio)
        """
        block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w: int = int(aspect_ratio * block_h)

        # Initialize lists to hold target patches and all unique patches
        target_patches: List[List[int]] = []
        all_patches: List[int] = []

        # For each of the target blocks to generate
        for _ in range(num_target_blocks):
            start_patch: int = JEPA_base.randomly_select_starting_patch_for_block(
                patch_width=patch_w,
                patch_height=patch_h,
                block_width=block_w,
                block_height=block_h,
            )

            # Initialize list to hold the patches for the target block
            patches: List[int] = []
            # Collect patches within the target block
            for h in range(block_h):
                for w in range(block_w):
                    patch_start_position: int = start_patch + h * patch_w + w
                    
                    patches.append(patch_start_position)
                    
                    if patch_start_position not in all_patches:
                        all_patches.append(patch_start_position)

            # Store the patches for the current target block
            target_patches.append(patches)

        return target_patches, all_patches

    @staticmethod
    def generate_target_patches_3d(
        patch_dim: Tuple[int, int, int],
        aspect_ratio: Number,
        scale: Number,
        num_target_blocks: int,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Generate target patches for each target block in 3D space (spatio-temporal).

        Args:
            patch_dim (Tuple[int, int, int]): Dimensions of the patches (temporal, height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.

        Returns:
            Tuple[List[List[int]], List[int]]:
                - target_patches: A list of lists containing indices of patches for each target block.
                - all_patches: A list of all unique patches used in target blocks.
        """
        # Extract patch dimensions
        patch_t, patch_h, patch_w = patch_dim

        # Calculate the number of patches in the target block
        num_patches_block: int = int(patch_t * patch_h * patch_w * scale)

        # Calculate the height and width of the target block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = sqrt(num_patches_block/aspect_ratio)
        """
        block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w: int = int(aspect_ratio * block_h)
        block_t: int = patch_t

        # Initialize lists to hold target patches and all unique patches
        target_patches: List[List[int]] = []
        all_patches: List[int] = []

        # For each of the target blocks to generate
        for _ in range(num_target_blocks):
            start_patch: int = JEPA_base.randomly_select_starting_patch_for_block_3d(
                patch_width=patch_w,
                patch_height=patch_h,
                block_width=block_w,
                block_height=block_h,
            )

            # Initialize list to hold the patches for the target block
            patches: List[int] = []
            # Collect patches within the target block
            for t in range(block_t):
                for h in range(block_h):
                    for w in range(block_w):
                        patches.append(start_patch + h * patch_w + w)
                        if start_patch + h * patch_w + w not in all_patches:
                            all_patches.append(start_patch + h * patch_w + w)

            # Store the patches for the current target block
            target_patches.append(patches)

        return target_patches, all_patches

    @staticmethod
    def generate_context_patches_2d(
        patch_dim: Tuple[int, int],
        aspect_ratio: Number,
        scale: Number,
        target_patches_to_exclude: List[int],
    ) -> List[int]:
        """
        Generate a list of patch indices for the context block, excluding target patches.

        Args:
            patch_dim (Tuple[int, int]): Dimensions of the patches (height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches_to_exclude (List[int]): List containing indices of target patches.

        Returns:
            List[int]: A list of patch indices for the context block excluding target patches.
        """
        # Extract patch dimensions
        patch_h, patch_w = patch_dim

        # Calculate the number of patches in the context block
        num_patches_block: int = int(patch_h * patch_w * scale)
        # Calculate the height and width of the context block maintaining the aspect ratio
        """
        aspect_ratio = w / h
        num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
        h = (num_patches_block/aspect_ratio)**.5
        """
        block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w: int = int(aspect_ratio * block_h)

        # Randomly select the starting patch for the context block
        start_patch: int = JEPA_base.randomly_select_starting_patch_for_block(
            patch_width=patch_w,
            patch_height=patch_h,
            block_width=block_w,
            block_height=block_h,
        )

        return [
            start_patch + h * patch_w + w
            for h in range(block_h)
            for w in range(block_w)
            if start_patch + h * patch_w + w not in target_patches_to_exclude
        ]

    @staticmethod
    def randomly_select_starting_patch_for_block_2d(
        patch_dim: Tuple[int, int],
        block_dim: Tuple[int, int],
        seed: Optional[int] = None,
    ) -> int:
        """
        Randomly selects the patch defining the block's starting position (on a linear index).

        Parameters:
        patch_dim (Tuple[int, int]): A tuple containing the width and height of the patch.
        block_dim (Tuple[int, int]): A tuple containing the width and height of the block from which the patch is to be extracted.
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

        patch_h, patch_w = patch_dim
        block_h, block_w = block_dim

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
