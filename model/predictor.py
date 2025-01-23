from typing import Optional

import torch
import torch.nn as nn
from x_transformers import Decoder


class Predictor(nn.Module):
    """
    A transformer-based decoder to predict the embeddings of target patches from context patch embeddings.

    This module is designed to take in context embeddings (from patches in an image or video) and predict the embeddings
    for a set of target patches. The model uses a transformer decoder to leverage the relationships between the context
    patches and the target patches, ultimately producing refined embeddings for the target patches.

    Args:
        embed_dim (int):
            The dimensionality of the embeddings used throughout the model. This is the size of the feature vectors
            representing each patch (both context and target). The `embed_dim` determines the size of the input and
            output embeddings for the transformer decoder. It should match the dimension of the embeddings provided
            by the preceding model (e.g., a Vision Transformer or similar).

        num_heads (int):
            The number of attention heads in the transformer's multi-head self-attention mechanism. More heads allow
            the model to focus on different parts of the input data in parallel, potentially capturing more complex
            dependencies.

        depth (int):
            The number of layers (or blocks) in the transformer decoder. Each layer consists of a self-attention mechanism
            and a feed-forward neural network, followed by layer normalization. A deeper network can model more complex
            relationships but requires more computational resources.

        predictor_embed_dim (Optional[int]):
            An optional intermediate embedding dimension used to map the context and target embeddings to a different
            dimensionality before applying the transformer decoder. This dimension acts as a bottleneck or projection
            layer, allowing the model to operate in a potentially lower-dimensional space (if `predictor_embed_dim` is
            less than `embed_dim`). If not provided, the model will operate directly in the `embed_dim` space, meaning
            that the predictor dimension will be the same as the embedding dimension.

    Attributes:
        decoder (x_transformers.Decoder):
            The transformer decoder responsible for processing the concatenated context and target embeddings. It captures
            the dependencies within the target patches and between the target and context patches.

        predictor_embed (nn.Module):
            A linear layer that projects the input embeddings (`embed_dim`) to the `predictor_embed_dim`, if provided.
            If `predictor_embed_dim` is not specified, this layer is an identity mapping, leaving the embeddings
            unchanged.

        predictor_norm (nn.LayerNorm or nn.Identity):
            A layer normalization applied to the output of the transformer decoder, only if `predictor_embed_dim` is
            provided. Otherwise, this is an identity mapping.

        predictor_proj (nn.Module):
            A linear layer that projects the decoder's output back to the original embedding dimension (`embed_dim`),
            if `predictor_embed_dim` is provided. This ensures that the final output of the model has the same dimension
            as the input embeddings. If `predictor_embed_dim` is not provided, this is an identity mapping.

    Example:
        ```python
        predictor = Predictor(embed_dim=768, num_heads=8, depth=6, predictor_embed_dim=512)
        context_encoding = torch.randn(batch_size, num_context_patches, embed_dim)
        target_masks = torch.randn(batch_size, num_target_patches, embed_dim)
        predictions = predictor(context_encoding, target_masks)
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        layer_dropout: float = 0.0,
        predictor_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        # Initialize the transformer-based decoder
        self.decoder = Decoder(
            dim=embed_dim, depth=depth, heads=num_heads, layer_dropout=layer_dropout
        )

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
            :, -target_masks.shape[1] :, :  # Include entire batch
        ]  # (batch_size, num_target_patches, embed_dim)

        return prediction
