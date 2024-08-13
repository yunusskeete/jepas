from typing import Optional

import torch
import torch.nn as nn
from x_transformers import Decoder


class Predictor(nn.Module):
    """
    Lightweight Predictor Module using VIT to predict target patches from context patches

    This module uses a transformer-based decoder to predict the embeddings of target patches from context patches.

    Args:
        embed_dim (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads in the transformer decoder.
        depth (int): Number of layers in the transformer decoder.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        predictor_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        # Initialize the transformer-based decoder
        self.decoder = Decoder(dim=embed_dim, depth=depth, heads=num_heads)

        self.predictor_embed = (
            nn.Linear(embed_dim, predictor_embed_dim, bias=True)
            if predictor_embed_dim
            else nn.Identity()
        )

        self.predictor_norm = (
            nn.LayerNorm(predictor_embed_dim) if predictor_embed_dim else nn.Identity()
        )
        self.predictor_proj = (
            nn.Linear(predictor_embed_dim, embed_dim, bias=True)
            if predictor_embed_dim
            else nn.Identity()
        )

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

        NOTE:
        `target_masks` are concatenated with `context_encoding` and decoded using a transformer decoder.
        The self attention mechanism in the decoder is applied to the concatenated sequence in successive blocks,
        capturing the dependencies within the target sequence, yielding an understanding
        of the context of the current token in relation to all other tokens.
        The prediction blocks are taken as the ouputs of the final decoder block corresponding to the target tokens.
        """
        # Concatenate the context encoding and the target masks
        x = torch.cat(
            (context_encoding, target_masks), dim=1
        )  # (batch_size, num_context_patches + num_target_patches, embed_dim)

        # Map context tokens to the predictor dimension
        x = self.predictor_embed(x)

        # Pass the concatenated tensor through the transformer decoder
        x = self.decoder(x)  # (batch_size, predictor_embed_dim, embed_dim)

        # Normalise and project predictor ouputs back to the input dimension
        x = self.predictor_proj(
            self.predictor_norm(x)
        )  # (batch_size, num_context_patches + num_target_patches, embed_dim)

        # Return the output corresponding to target tokens, i.e., the last len(target_masks) tokens
        prediction = x[
            :, -target_masks.shape[1] :, :
        ]  # (batch_size, num_target_patches, embed_dim)

        return prediction
