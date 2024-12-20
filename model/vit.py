from typing import Any, Optional, Tuple, Union
from einops import rearrange

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

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        print(f"{self.pos_embedding.shape=}")
        self.stacked_pos_embedding = None

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

    def pseudo_3d_pos_embedding(self, patch_shape, random_t: int):
        """
        Generates a pseudo-3D positional embedding by reshaping and stacking a 2D positional embedding.

        Args:
            conv_t (int): Temporal dimension (time steps).
            conv_h (int): Height dimension.
            conv_w (int): Width dimension.
            random_t (int): Time index to extract a slice of the positional embedding.

        Raises:
            AssertionError: If the positional embedding slice is incorrect or the shape of the stacked embedding is wrong.

        Side Effects:
            Updates `self.stacked_pos_embedding` with the reshaped and stacked positional embedding.
        """
        t, h, w = patch_shape
        pos_emb_reshape = rearrange(
            self.pos_embedding,
            "b (t h w) e -> b e t h w",
            t=t,
            h=h,
            w=w,
        )

        single_pos_embedding_slice = pos_emb_reshape[:, :, random_t, :, :]

        assert torch.equal(
            single_pos_embedding_slice, pos_emb_reshape[:, :, random_t, :, :]
        ), "Not correct positional embedding slice"

        single_t_slice_reshaped = single_pos_embedding_slice.unsqueeze(2)

        pos_emb_stacked = single_t_slice_reshaped.repeat(1, 1, t, 1, 1)

        assert (
            pos_emb_stacked.shape == pos_emb_reshape.shape
        ), "Shape of stacked positional embedding not correct"

        self.stacked_pos_embedding = rearrange(
            pos_emb_stacked, "1 e t h w -> 1 (t h w) e"
        )

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

            self.pseudo_3d_pos_embedding(
                patch_shape=self.patch_embed.patch_shape,
                random_t=random_t,
            )
            x_stacked = x_stacked + self.stacked_pos_embedding
            x_stacked = self.post_emb_norm_vit(x_stacked)

        if use_static_positional_embedding:
            self.pseudo_3d_pos_embedding(
                patch_shape=self.patch_embed.patch_shape,
                random_t=random_t,
            )
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
