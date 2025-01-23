from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from utils.types import ensure_tuple


class TokenEmbed1D(nn.Module):
    """
    Token Embedding

    This module converts token ids into token embeddings.

    Args:
        num_embeddings (int): Number of tokens in the vocabulary.
        embed_dim (int): Dimension of the token embedding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to convert the input token ids into token embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_token_ids).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_token_ids, embed_dim).
        """
        x = self.embed(x)

        return x  # (b, n, e)


# Adapted from https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
class PatchEmbed2D(nn.Module):
    """
    Image to Patch Embedding

    This module converts an input image into a sequence of patch embeddings using a convolutional layer.

    Args:
        img_size (int | Tuple[int, int], optional): Size of the input image. Default is 224.
        patch_size (int | Tuple[int, int], optional): Size of each patch. Default is 16.
        in_chans (int, optional): Number of input channels. Default is 3 (RGB).
        embed_dim (int, optional): Dimension of the embedding space. Default is 64.
    """

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 64,
    ):
        super().__init__()
        # Ensure img_size and patch_size are tuples
        img_size: Tuple[int, int] = ensure_tuple(img_size)
        patch_size: Tuple[int, int] = ensure_tuple(patch_size)

        # Calculate the number of patches in each dimension
        self.patch_shape: Tuple[int, int] = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )

        # Convolutional layer to convert the image into patches
        self.conv = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,  # Same stride as the patch_size as to extract non-overlapping patches
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to convert the input image into patch embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_chans, height, width), where ins_chans is the number of channels in the image, typically 3 for RGB.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_dim).
        """
        # Apply the convolutional layer to get patches
        x = self.conv(x)
        # Flatten the patches into a sequence
        x = rearrange(x, "b e h w -> b (h w) e")

        return x  # (b, n, e)


class PatchEmbed3D(nn.Module):
    """
    Video to Patch Embedding

    This module converts an input image-sequence into a sequence of patch embeddings using a convolutional layer.

    Args:
        img_size (int | Tuple[int, int], optional): Size of the input image. Default is 224.
        patch_size (int | Tuple[int, int], optional): Size of each patch. Default is 16.
        tubelet_size (int | Tuple[int, int], optional): The number of consecutive frames considered together as a single "tubelet" for patch extraction.
        in_chans (int, optional): Number of input channels. Default is 3 (RGB).
        embed_dim (int, optional): Dimension of the embedding space. Default is 64.
    """

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        num_frames: int = 16,
        patch_size: int | Tuple[int, int] = 16,
        tubelet_size: int = 2,  # Temporal dimension
        in_chans: int = 3,
        embed_dim: int = 64,
    ):
        super().__init__()
        # Ensure img_size and patch_size are tuples
        img_size: Tuple[int, int] = ensure_tuple(img_size)
        patch_size: Tuple[int, int] = ensure_tuple(patch_size)

        self.patch_shape: Tuple[int, int, int] = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )

        # Convolutional layer to convert the video into patches
        self.conv = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(
                tubelet_size,
                patch_size[0],
                patch_size[1],
            ),
            stride=(
                tubelet_size,
                patch_size[0],
                patch_size[1],
            ),  # Same stride as the patch_size as to extract non-overlapping patches
        )

    def forward(self, x, **kwargs):
        """
        Forward pass to convert the input image-sequence into patch embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, tubelet_size, in_chans, height, width),
                              where tubelet_size if the temporal dimension of the input image-sequence
                              and ins_chans is the number of channels in the image, typically 3 for RGB.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_dim).
        """
        # Apply the convolutional layer to get patches
        x = self.conv(x)
        # Flatten the patches into a sequence
        x = rearrange(x, "b e t h w -> b (t h w) e")

        return x  # (b n e)
