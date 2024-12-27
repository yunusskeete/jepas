from typing import Any, Optional, Tuple, Union
from einops import rearrange, repeat
import math

import torch
import torch.nn as nn
from x_transformers import Encoder

from utils.types import ensure_tuple

from .patch_embed import PatchEmbed2D, PatchEmbed3D


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_frames: int = 1,
        tubelet_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
        enc_depth: int = 8,
        num_heads: int = 8,
        post_emb_norm: bool = True,
        post_enc_norm: bool = True,
        layer_dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.img_size = ensure_tuple(img_size)
        self.patch_size = ensure_tuple(patch_size)
        self.in_chans = in_chans

        self.num_frames = num_frames
        self.is_video = num_frames > 1
        self.tubelet_size = tubelet_size

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed: nn.Module = (
            PatchEmbed2D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            if not self.is_video
            else PatchEmbed3D(
                img_size=img_size,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        )
        self.num_patches: int = int(
            torch.prod(torch.Tensor(self.patch_embed.patch_shape)).item()
        )

        (
            self.stacked_pos_embedding,
            self.pos_embedding,
        ) = self.generate_new_positional_embeddings(
            seq_len=in_chans * num_frames * img_size * img_size, mode="train"
        )  # [1, (channels * frames * height * width)]

        self.post_emb_norm = post_emb_norm
        self.post_emb_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_emb_norm else nn.Identity()
        )

        self.layer_dropout = layer_dropout

        self.encoder = Encoder(  # student encoder
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth,
            layer_dropout=self.layer_dropout,
        )

        self.post_enc_norm = post_enc_norm
        self.post_enc_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_enc_norm else nn.Identity()
        )  # student encoder

    def generate_new_positional_embeddings(self, seq_len: int, mode: str = "train"):
        """
        Generate two new positional embeddings from the original positional embedding:
        1. Positional embedding of the first frame stacked `num_frames` times.
        2. Positional embedding of the rest of the frames (2-n).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Stacked positional embedding of the first frame, shape [1, num frames, embed_dim].
                - Positional embedding of frames 2 to n, shape [1, num_frames-1, embed_dim].
        """
        frame_pos_embedding = self.get_sinusoidal_positional_embedding(seq_len=seq_len)

        if mode == "test":
            # Positional embeddings for frames 2 to n
            n_positional_embedding = self.get_frames_1_to_n_positional_embedding(
                original_embedding=frame_pos_embedding,
                frames=self.num_frames,
                height=self.img_size[0],
                width=self.img_size[1],
                channels=self.in_chans,
            )

        stacked_first_frame_embedding = self.get_stacked_frame_positional_embedding(
            original_embedding=frame_pos_embedding,
            frames=self.num_frames,
            height=self.img_size[0],
            width=self.img_size[1],
            channels=self.in_chans,
        )

        frame_pos_embedding = (
            n_positional_embedding if mode == "test" else frame_pos_embedding
        )
        return stacked_first_frame_embedding, frame_pos_embedding

    def get_sinusoidal_positional_embedding(self, seq_len: int, device="cuda"):
        """
        Generate a sinusoidal positional embedding for a flattened video tensor.
        Each positional embedding is a single value (no embedding dimension).

        Args:
            seq_len (int): The sequence length, which is channels * frames * height * width.

        Returns:
            torch.Tensor: Sinusoidal positional embedding of shape [1, seq_len].
        """
        # Create a tensor of shape [seq_len, 1]
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(
            1
        )  # Shape: [seq_len, 1]

        # Define the dimension of the embedding
        dim = 1  # No embedding dimension as each positional embedding is a single value

        # Compute the sinusoidal functions (sine and cosine) for positional encoding
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float, device=device)
            * -(math.log(10000.0) / dim)
        )  # Shape: [1]

        # Apply sine and cosine to even and odd indices
        pos_embedding = torch.sin(position * div_term)  # Shape: [seq_len, 1]

        # Add an extra dimension for batch size (1)
        pos_embedding = pos_embedding.unsqueeze(0).squeeze(-1)  # Shape: [1, seq_len]

        return pos_embedding.to(device=device)

    def get_stacked_frame_positional_embedding(
        self, original_embedding, frames, height, width, channels
    ):
        """
        Generate a positional embedding for a stacked video tensor where the first frame is repeated.

        Args:
            original_embedding (torch.Tensor): The original positional embedding of shape [1, seq_len].
            frames (int): Number of frames in the video.
            height (int): Height of each frame.
            width (int): Width of each frame.
            channels (int): Number of channels in the video (e.g., RGB).

        Returns:
            torch.Tensor: New positional embedding for the stacked video tensor, shape [1, seq_len].
        """
        # Compute the sequence length (seq_len = channels * frames * height * width)
        seq_len = channels * frames * height * width
        _, original_seq_len = original_embedding.shape
        assert (
            original_seq_len == seq_len
        ), f"Sequence length of the original embedding ({original_seq_len}) does not match expected seq_len ({seq_len})."

        # Number of patches per frame (considering channels)
        num_patches = channels * height * width

        # Extract positional embedding for the first frame (the first `num_patches` elements)
        first_frame_embedding = original_embedding[
            :, :num_patches
        ]  # Shape: [1, num_patches]

        # Repeat the first frame embedding across the temporal dimension (frames)
        stacked_embedding = first_frame_embedding.repeat(
            1, frames
        )  # Shape: [1, frames * num_patches]

        return stacked_embedding

    def get_frames_1_to_n_positional_embedding(
        self, original_embedding, frames, height, width, channels
    ):
        """
        Extract positional embedding for frames 1 to n (excluding the first frame).

        Args:
            original_embedding (torch.Tensor): The original positional embedding of shape [1, seq_len].
            frames (int): Total number of frames in the video.
            height (int): Height of each frame.
            width (int): Width of each frame.
            channels (int): Number of channels in the video (e.g., RGB).

        Returns:
            torch.Tensor: Positional embedding for frames 1 to n, shape [1, seq_len'],
                            where seq_len' = (frames - 1) * height * width * channels.
        """
        # Compute the sequence length (seq_len = channels * frames * height * width)
        seq_len = channels * frames * height * width
        _, original_seq_len = original_embedding.shape
        assert (
            original_seq_len == seq_len
        ), f"Sequence length of the original embedding ({original_seq_len}) does not match expected seq_len ({seq_len})."

        # Number of patches per frame (considering channels)
        num_patches = channels * height * width

        # Slice the embedding for frames 1 to n (excluding the first frame)
        frames_1_to_n_embedding = original_embedding[
            :, num_patches:
        ]  # Shape: [1, (frames - 1) * num_patches]

        return frames_1_to_n_embedding

    def flatten_video_tensor(self, video_tensor):
        """
        Flatten a video tensor into a sequence format for positional embeddings.

        Args:
            video_tensor (torch.Tensor): Input tensor of shape [batch_size, channels, frames, height, width].

        Returns:
            torch.Tensor: Flattened tensor of shape [batch_size, seq_len, embed_dim].
            tuple: Original shape for unflattening later.
        """
        original_shape = video_tensor.shape  # Save original shape for unflattening
        batch_size, channels, frames, height, width = original_shape
        seq_len = channels * frames * height * width

        # Flatten the spatial-temporal dimensions into a sequence
        flattened = video_tensor.view(batch_size, seq_len)
        return flattened, original_shape

    def unflatten_video_tensor(self, flattened_tensor, original_shape):
        """
        Unflatten a flattened tensor back to the original video tensor shape.

        Args:
            flattened_tensor (torch.Tensor): Flattened tensor of shape [batch_size, seq_len].
            original_shape (tuple): The original shape of the video tensor.

        Returns:
            torch.Tensor: Unflattened tensor of shape [batch_size, channels, frames, height, width].
        """
        return flattened_tensor.view(original_shape)

    def forward_vit(
        self,
        x: torch.Tensor,
        x_stacked: Optional[torch.Tensor] = None,
        random_t: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        patch_embed_only: bool = False,
        static_scene_temporal_reasoning: bool = False,
        use_static_positional_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if static_scene_temporal_reasoning and x_stacked is not None:
            flattened_x_stacked, original_stacked_shape = self.flatten_video_tensor(
                video_tensor=x_stacked
            )
            flattened_x_stacked = flattened_x_stacked + self.stacked_pos_embedding
            x_stacked = self.unflatten_video_tensor(
                flattened_tensor=flattened_x_stacked,
                original_shape=original_stacked_shape,
            )
            x_stacked = self.patch_embed(x_stacked)
            x_stacked = self.post_emb_norm_vit(x_stacked)

        flattened_x, original_shape = self.flatten_video_tensor(video_tensor=x)
        if use_static_positional_embedding:
            flattened_x = flattened_x + self.stacked_pos_embedding
        else:
            # Add positional embeddings to the patch embeddings
            flattened_x = (
                flattened_x + self.pos_embedding
            )  # (batch, num_patches, embed_dim)

        x = self.unflatten_video_tensor(
            flattened_tensor=flattened_x, original_shape=original_shape
        )

        # Obtain patch embeddings from the input tensor
        x_embed = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        # Normalize the patch embeddings (if `self.post_emb_norm`)
        x_embed = self.post_emb_norm_vit(x_embed)  # (batch, num_patches, embed_dim)

        if patch_embed_only:
            if static_scene_temporal_reasoning:
                return x_embed, x_stacked
            else:
                return x_embed
        # Encode the patch embeddings using the student encoder
        x_embed = self.encoder(
            x_embed, attn_mask=attention_mask
        )  # (batch, num_patches, embed_dim)

        # Normalize the encoded patches (if `self.post_enc_norm`)
        x_embed = self.post_enc_norm_vit(x_embed)  # (batch, num_patches, embed_dim)

        if static_scene_temporal_reasoning:
            return x_embed, x_stacked
        else:
            return x_embed


def vit_nano(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=64,
        enc_depth=8,
        num_heads=8,
        **kwargs,
    )


def vit_tiny(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=192,
        enc_depth=18,
        num_heads=8,
        **kwargs,
    )


def vit_small(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=384,
        enc_depth=12,
        num_heads=8,
        **kwargs,
    )


def vit_base(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=768,
        enc_depth=12,
        num_heads=12,
        **kwargs,
    )


def vit_large(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=1024,
        enc_depth=24,
        num_heads=16,
        **kwargs,
    )


def vit_huge(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=1280,
        enc_depth=32,
        num_heads=16,
        **kwargs,
    )
