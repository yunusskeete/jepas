from typing import List, Tuple

import torch
import torch.nn as nn
from x_transformers import Decoder

from utils.tensors import validate_tensor_dimensions


class Predictor(nn.Module):
    """
    Lightweight Predictor Module using VIT to predict target patches from context patches

    This module uses a transformer-based decoder to predict the embeddings of target patches from context patches.

    Args:
        embed_dim (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads in the transformer decoder.
        depth (int): Number of layers in the transformer decoder.
    """

    def __init__(self, embed_dim: int, num_heads: int, depth: int):
        super().__init__()
        # Initialize the transformer-based decoder
        self.decoder = Decoder(dim=embed_dim, depth=depth, heads=num_heads)

    def forward(
        self, context_encoding: torch.Tensor, target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to predict the embeddings of target patches from context patches.

        Args:
            context_encoding (torch.Tensor): Tensor containing the context patch embeddings of shape (batch_size, num_context_patches, embed_dim).
            target_masks (torch.Tensor): Tensor containing the target mask embeddings of shape (batch_size, num_target_patches, embed_dim).

        Returns:
            torch.Tensor: Tensor containing the predicted embeddings of the target patches of shape (batch_size, num_target_patches, embed_dim).
        """
        # Ensure the dimensions match
        validate_tensor_dimensions(
            size1=context_encoding.size(),
            size2=target_masks.size(),
            dimension_checks=[
                (
                    0,
                    "The batch dimension of the context encoding does not equal the batch dimension of the target masks",
                ),
                (
                    2,
                    "The embedding dimension of the context encoding does not equal the embedding dimension of the target masks",
                ),
            ],
        )

        # Concatenate the context encoding and the target masks
        x = torch.cat(
            (context_encoding, target_masks), dim=1
        )  # (batch_size, num_context_patches + num_target_patches, embed_dim)
        # Pass the concatenated tensor through the transformer decoder
        x = self.decoder(
            x
        )  # (batch_size, num_context_patches + num_target_patches, embed_dim)
        # Return the output corresponding to target tokens, i.e., the last len(target_masks) tokens
        prediction = x[
            :, -target_masks.shape[1] :, :
        ]  # (batch_size, num_target_patches, embed_dim)

        return prediction
