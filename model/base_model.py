import copy
from typing import Any, List, Literal, Optional, Set, Tuple, Union

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

        self.teacher_encoder = copy.deepcopy(
            self.encoder
        ).cuda()  # copy student encoder

        # TODO: To help prevent colapse and prioritise expressive representations
        # in the encoder, the decoder should be underpowered with respect to the encoder.
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
                    target_patches_for_block,  # Include only selected patches
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
        Generate a context block from the input tensor, excluding target patches.

        Args:
            x (torch.Tensor): Input tensor to be processed of shape (batch_size, num_patches, embed_dim).
            context_patches (List[int]): List containing indices of context patches.

        Returns:
            torch.Tensor: A tensor containing the context block with target patches excluded of shape (batch_size, num_context_patches, embed_dim).
        """
        # Return the context block tensor excluding target patches
        context_block: torch.Tensor = x[
            :,  # Include batch dim
            context_patches,  # Include only selected patches
            :,  # Include all embed dim
        ]  # (batch_size, num_context_patches, embed_dim)

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
        """
        (Image/video invariant)

        Generates predictions for target blocks in an image/video sequence using a transformer decoder.

        This function operates on context encoding obtained from a Vision Transformer (ViT) model, making predictions
        for specified target blocks. The target blocks are specified by `target_patches`, and the context encoding is
        enriched with positional information before being processed by a transformer decoder to produce the final predictions.

        Args:
            num_target_blocks (int): The number of target blocks to generate predictions for.
            batch_dim (int): The batch size dimension, representing the number of sequences being processed in parallel.
            num_patches (int): The number of patches in each target block.
            embed_dim (int): The embedding dimension for each patch.
            target_patches (List[List[int]]): A list of lists, where each inner list contains the patch indices corresponding
                                              to a target block.
            context_encoding (torch.Tensor): A tensor of shape (batch_dim, num_context_patches, embed_dim) containing the
                                             context encoding from the ViT model, enriched with positional encodings.

        Returns:
            torch.Tensor: A tensor of shape (num_target_blocks, batch_dim, num_patches, embed_dim) containing the predicted
                          embeddings for each target block. Each target block's prediction is based on the context encoding
                          and the masked target tokens, which are processed by a transformer decoder.
        """
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
        static_scene_temporal_reasoning: bool = False,
        use_static_positional_embedding: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        (Image/video invariant)

        Forward pass for generating predictions and targets within the JEPA architecture.

        Args:
            x (torch.Tensor): Input tensor of shape: (batch_size, channels, img_height, img_width) if not self.is_video else (batch_size, channels, time, img_height, img_width).
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
        test_mode: bool = self.mode == "test"
        """
        Skip (student) encoder (`patch_embed_only=True`) during training
        as patch embeddings are jointly encoded by the student and the teacher
        encoder such that the student can learn to predict the teacher

        NOTE: Positional encoding applied to `x` during `self.forward_vit()`
        """
        output = self.forward_vit(
            x=x,
            attention_mask=None,
            patch_embed_only=test_mode,
            static_scene_temporal_reasoning=static_scene_temporal_reasoning,
            use_static_positional_embedding=use_static_positional_embedding,
        )
        if static_scene_temporal_reasoning:
            x, x_stacked = output
        else:
            x, x_stacked = output, None

        batch_size, num_patches, embed_dim = (  # pylint: disable=unused-variable
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
                x=x_stacked if static_scene_temporal_reasoning else x,
                context_patches=context_patches,
            )
        )
        batch_size, num_context_patches, embed_dim = context_block.shape

        context_encoding: torch.Tensor = (
            self.post_enc_norm_jepa(  # NOTE: `context_encoding` contains positional information from `x`, which underwent the `self.forward_vit()` pass
                self.encoder(  # student encoder (ViT)
                    x=context_block,
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
            context_block,
            target_patches,
            context_patches,
        )

    @staticmethod
    def randomly_select_starting_patch_for_block(
        patch_dim: Union[Tuple[int, int], Tuple[int, int, int]],
        block_dim: Union[Tuple[int, int], Tuple[int, int, int]],
        seed: Optional[int] = None,
    ) -> int:
        """
        (Placeholder function)

        Randomly selects the patch defining the 2D/3D block's starting position (on a linear index).

        Parameters:
        patch_dim (Union[Tuple[int, int], Tuple[int, int, int]]): A tuple containing the number of patches in each dimension (width and height)/(temporal dimension, width and height).
        block_dim (Union[Tuple[int, int], Tuple[int, int, int]]): A tuple containing the number of patches in each dimension (width and height)/(temporal dimension, width and height) of the block from which the patch is to be extracted.
        seed (Optional[int]): An optional random seed for reproducibility.

        Returns:
        int: The starting position of the patch within the block, represented as a linear index.

        NOTE:
        Patches are the basic (processing) units of the image/video (e.g. 16x16/num_framesx16x16 pixels).
        Blocks are larger regions composed of multiple patches.
        In training, the model attempts to understand blocks within an image/video - ie. context blocks - by processing it one patch at a time,
        and uses this understanding is used to predict the structure and content of (the target blocks within) an image/video in a more abstract way.

        Linear index coordinates are used to define the starting patch for a block,
        and map 2D/3D pixel coordinates onto a 1D array index (flattened form).
        """
        raise NotImplementedError()

    @staticmethod
    def generate_target_patches(
        patch_dim: Union[Tuple[int, int], Tuple[int, int, int]],
        aspect_ratio: Number,
        scale: Number,
        num_target_blocks: int,
    ) -> Tuple[List[List[int]], Set[int]]:
        """
        (Placeholder function)

        Generate (spatial/spatio-temporal) target patches for each 2D/3D target block.

        Args:
            patch_dim (Union[Tuple[int, int], Tuple[int, int, int]]): The number of patches in each dimension (height, width)/(temporal, height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
            scale (Number): Scaling factor for the number of patches in the target block.
            num_target_blocks (int): Number of target blocks to generate.

        Returns:
            Tuple[List[List[int]], Set[int]]:
                - target_patches: A list of lists containing indices of patches for each target block.
                - all_patches: A set of all unique patches used in target blocks.
        """
        raise NotImplementedError()

    @staticmethod
    def generate_context_patches(
        patch_dim: Union[Tuple[int, int], Tuple[int, int, int]],
        aspect_ratio: Number,
        scale: Number,
        target_patches_to_exclude: Set[int],
    ) -> List[int]:
        """
        (Placeholder function)

        Generate a list of patch indices for the 2D/3D context block, excluding target patches.

        Args:
            patch_dim (Union[Tuple[int, int], Tuple[int, int, int]]): Dimensions of the patches (height, width)/(temporal, height, width).
            aspect_ratio (Number): Aspect ratio to be maintained for the context block.
            scale (Number): Scaling factor for the number of patches in the context block.
            target_patches_to_exclude (Set[int]): Set containing indices of target patches.

        Returns:
            List[int]: A list of patch indices for the context block excluding target patches.
        """
        raise NotImplementedError()
