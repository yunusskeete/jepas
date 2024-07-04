import copy
from typing import Any, Callable, List, Literal, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from einops import repeat
from x_transformers import Encoder

from utils.tensors import validate_tensor_dimensions
from utils.types import Number

from .predictor import Predictor
from .vit import VisionTransformer


class JEPA_base(VisionTransformer):
    def __init__(
        self,
        decoder_depth: int,
        num_target_blocks: int = 4,
        mode: Literal["test", "train"] = "train",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_target_blocks = num_target_blocks  # Number of patches (Unique)
        self.mode = mode  # Unique

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))  # Unique
        nn.init.trunc_normal_(self.mask_token, 0.02)

        self.norm = nn.LayerNorm(self.embed_dim)

        self.teacher_encoder = copy.deepcopy(  # Unique
            self.encoder  # student encoder
        ).cuda()

        self.predictor = Predictor(  # Unique
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
        Generate target blocks from the input tensor using a target encoder.

        Args:
            x (torch.Tensor): Input tensor to be processed by the target encoder of shape (batch, num_patches, embed_dim).
            target_patches (List[List[ing]]): TODO

        Returns:
            torch.Tensor:
                - target_block: A tensor containing the generated target blocks.
        """
        # Ensure the target encoder is in evaluation mode
        target_encoder = self.teacher_encoder.eval()
        # Encode the input tensor
        x = target_encoder(x)  # (batch, num_patches, embed_dim)
        x = self.norm(x)  # (batch, num_patches, embed_dim)

        # Create a list to hold the target blocks
        target_blocks_list: List[torch.Tensor] = []

        for target_block_idx in range(self.num_target_blocks):
            patches = target_patches[target_block_idx]
            # Assign the corresponding encoded values to the target block tensor
            target_blocks_list.append(x[:, patches, :])

        # Stack the list of tensors along the new dimension (0)
        target_block: torch.Tensor = torch.stack(
            target_blocks_list, dim=0
        )  # (num_target_blocks, batch, target_block_size, embed_dim)

        return target_block.cuda()

    def get_context_block(
        self,
        x: torch.Tensor,
        context_patches: List[int],
    ) -> torch.Tensor:
        """
        Generate a context block from the input tensor, excluding target patches.

        Args:
            x (torch.Tensor): Input tensor to be processed of shape (batch, num_patches, embed_dim).
            context_patches (List[int]): List containing indices of context patches.

        Returns:
            torch.Tensor: A tensor containing the context block with target patches excluded.
        """
        # Return the context block tensor excluding target patches
        return x[:, context_patches, :]  # (batch, num_patches, embed_dim)

    def make_predictions(
        self,
        num_target_blocks: int,
        batch_dim: int,
        num_patches: int,
        embed_dim: int,
        target_patches: List[List[int]],
        context_encoding: torch.Tensor,
    ) -> torch.Tensor:
        # Initialize tensor to hold prediction blocks
        prediction_blocks = torch.zeros(
            (num_target_blocks, batch_dim, num_patches, embed_dim)
        ).cuda()  # TODO: Extend to temporal dimenion if is_video

        # Predict each target block separately using the context encoding and mask tokens
        for i in range(num_target_blocks):
            target_masks = self.mask_token.repeat(batch_dim, num_patches, 1)
            target_pos_embedding = self.pos_embedding[
                :, target_patches[i], :
            ]  # NOTE: This includes a temporal dimenion if is_video
            target_masks = target_masks + target_pos_embedding

            # TODO: Do we map input to predictor dimension?

            # Generate prediction for the current target block
            prediction_block = self.predictor(
                context_encoding, target_masks
            )  # (batch_size, target_block_size, embed_dim)
            prediction_blocks[i] = prediction_block

            # TODO: Do we Normalize and project predictor ouputs back to input dimension?

        return prediction_blocks

    def forward_base(
        self,
        x: torch.Tensor,
        target_patches: List[List[int]],
        context_patches: List[int],
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for generating predictions and targets within the JEPA architecture.

        Args:
            x (torch.Tensor): Input tensor (shape: (batch_size, img_channels = RGB = 3, img_height, img_width)).
            target_aspect_ratio (Number): Aspect ratio for the target blocks.
            target_scale (Number): Scale factor for the target blocks.
            context_aspect_ratio (Number): Aspect ratio for the context block.
            context_scale (Number): Scale factor for the context block.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If self.mode is "test":
                    torch.Tensor: Full embedding tensor from the student encoder.
                - If self.mode is not "test":
                    Tuple[torch.Tensor, torch.Tensor]:
                        - prediction_blocks: Predicted blocks based on the context encoding.
                        - target_blocks: Actual target blocks.
        """
        x: torch.Tensor = self.forward_vit(x, skip_encoder=True)
        b, n, e = x.shape  # (batch_size, num_patches, embed_dim)

        # If in test mode, return the full embedding using the student encoder
        if self.mode == "test":
            return x  # (batch_size, num_patches, embed_dim)

        # Get target embeddings using the target encoder
        target_blocks: torch.Tensor = self.get_target_blocks(
            x=x,
            target_patches=target_patches,
        )  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        m, b, n_t, e = target_blocks.shape
        # TODO: Extend to temporal dimenion if is_video

        # Get context embeddings excluding the target patches
        context_block: torch.Tensor = self.get_context_block(
            x=x,
            context_patches=context_patches,
        )  # (batch_size, num_context_patches, embed_dim)
        b, n_c, e = context_block.shape

        context_encoding: torch.Tensor = self.norm(
            self.encoder(context_block)  # student encoder
        )  # (batch_size, num_context_patches, embed_dim)
        b, n_e, e = context_encoding.shape

        assert (
            n_c == n_e
        ), f"The number of context patches in the context_block ({n_c}) does not equal the number of context patches in the context_encoding ({n_e})."

        prediction_blocks = self.make_predictions(
            num_target_blocks=m,
            batch_dim=b,
            num_patches=n_t,
            embed_dim=e,
            target_patches=target_patches,
            context_encoding=context_encoding,
        )  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        m, b, n_p, e = prediction_blocks.shape

        assert (
            n_t == n_p
        ), f"The number of context patches in the target_block ({n_t}) does not equal the number of context patches in the prediction_blocks ({n_p})."

        return (
            prediction_blocks,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
            target_blocks,  # (num_target_blocks, batch_size, target_block_size, embed_dim)
        )
