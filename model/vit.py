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
            mode="train"
        )  # [1, num patches, embed dim]

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

    def generate_new_positional_embeddings(self, mode: str = "train"):
        """
        Generate two new positional embeddings from the original positional embedding:
        1. Positional embedding of the first frame stacked `num_frames` times.
        2. Positional embedding of the rest of the frames (2-n).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Stacked positional embedding of the first frame, shape [1, num frames, embed_dim].
                - Positional embedding of frames 2 to n, shape [1, num_frames-1, embed_dim].
        """
        frame_pos_embedding = self.get_sinusoidal_positional_embedding(
            seq_len=self.num_frames, dim=self.embed_dim
        )
        # Extract the number of frames and embedding dimension
        batch_size, num_frames, embed_dim = frame_pos_embedding.shape

        if num_frames < 2:
            raise ValueError("Number of frames must be at least 2 to split embeddings.")

        # Positional embedding for the first frame, stacked `num_frames` times
        first_frame_embedding = frame_pos_embedding[
            :, 0:1, :
        ]  # Shape: [1, 1, embed_dim]
        stacked_first_frame_embedding = repeat(
            first_frame_embedding, "1 1 e -> 1 n e", n=num_frames
        )  # Shape: [1, num_frames, embed_dim]

        if mode == "test":
            # Positional embeddings for frames 2 to n
            frame_pos_embedding = frame_pos_embedding[
                :, 1:, :
            ]  # Shape: [1, num_frames-1, embed_dim]

        stacked_first_frame_embedding = self.get_positional_embedding_3d_convolution(
            num_patches=self.num_patches,
            batch_size=batch_size,
            frame_pos_embedding=stacked_first_frame_embedding,
        )
        nonstacked_frame_embeddings = self.get_positional_embedding_3d_convolution(
            num_patches=self.num_patches,
            batch_size=batch_size,
            frame_pos_embedding=frame_pos_embedding,
        )

        return stacked_first_frame_embedding, nonstacked_frame_embeddings

    def get_sinusoidal_positional_embedding(
        self, seq_len: int, dim: int, device="cuda"
    ):
        """
        Generates sinusoidal positional embeddings for a sequence.

        Each position is encoded as a vector of `dim` dimensions using sine and cosine functions,
        ensuring smooth, interpretable embeddings that generalize well to unseen sequences.

        Args:
            seq_len (int): Number of positions in the sequence.
            dim (int): Dimensionality of the embeddings.
            device (str): Device for the tensor ('cpu' or 'cuda').

        Returns:
            torch.Tensor: A tensor of shape (seq_len, dim) with sinusoidal embeddings.

        Example:
            >>> embedding = get_sinusoidal_embedding(seq_len=392, dim=512, device='gpu')
            >>> print(embedding.shape)
            torch.Size([1, 392, 512])
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float, device=device)
            * -(math.log(10000.0) / dim)
        )
        embedding = torch.zeros(seq_len, dim, device=device)
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding.unsqueeze(0)  # Add batch dimension

    def get_positional_embedding_3d_convolution(
        self,
        num_patches,
        batch_size,
        frame_pos_embedding,
    ):
        """
        Apply frame-level sinusoidal positional embedding to a patch tensor created by 3D convolution.

        Args:
            patch_tensor (torch.Tensor): Tensor from the pretrained model, shape [batch, num_patches, embed_dim].
            frame_pos_embedding (torch.Tensor): Frame sinusoidal positional embedding, shape [1, num_frames, embed_dim].

        Returns:
            torch.Tensor: positional embeddings, shape [batch, num_patches, embed_dim].
        """

        patch_shape = self.patch_embed.patch_shape
        kernel_t = self.tubelet_size
        stride_t = self.tubelet_size
        padding_t = 0

        t_p, h_p, w_p = patch_shape

        # Ensure the number of patches matches t_p * h_p * w_p
        if num_patches != t_p * h_p * w_p:
            raise ValueError(
                f"Mismatch between num_patches ({num_patches}) and t_p * h_p * w_p ({t_p} * {h_p} * {w_p})."
            )

        # Map each temporal patch to the corresponding frame index
        num_frames = frame_pos_embedding.size(1)
        frame_indices = torch.arange(t_p) * stride_t - padding_t + kernel_t // 2
        frame_indices = frame_indices.clamp(
            min=0, max=num_frames - 1
        )  # Ensure valid indices

        # Gather frame positional embeddings for the temporal patches
        temporal_pos_embedding = frame_pos_embedding[
            :, frame_indices, :
        ]  # Shape: [1, t_p, embed_dim]

        # Repeat temporal embeddings for all spatial patches in each temporal patch
        patches_per_temporal_patch = h_p * w_p
        expanded_pos_embedding = temporal_pos_embedding.repeat_interleave(
            patches_per_temporal_patch, dim=1
        )  # Shape: [1, num_patches, embed_dim]

        # Broadcast to batch size and add to the patch tensor
        expanded_pos_embedding = expanded_pos_embedding.expand(
            batch_size, -1, -1
        )  # Shape: [batch, num_patches, embed_dim]

        return expanded_pos_embedding

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

        # Obtain patch embeddings from the input tensor
        x_embed = self.patch_embed(x)  # (batch, num_patches, embed_dim)

        if static_scene_temporal_reasoning and x_stacked is not None:
            x_stacked = self.patch_embed(x_stacked)
            x_stacked = x_stacked + self.stacked_pos_embedding
            x_stacked = self.post_emb_norm_vit(x_stacked)

        if use_static_positional_embedding:
            x_embed = x_embed + self.stacked_pos_embedding
        else:
            # Add positional embeddings to the patch embeddings
            x_embed = x_embed + self.pos_embedding  # (batch, num_patches, embed_dim)

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
