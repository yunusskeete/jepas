from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEncoder
from x_transformers import Encoder


class TextEncoder(nn.Module):
    def __init__(
        self,
        encoder: Union[BertEncoder, Encoder],
        is_bert: bool,
    ):
        super().__init__()

        self.is_bert = is_bert
        self.encoder = encoder

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)

        return out.last_hidden_state if self.is_bert else out


def create_text_encoder(
    is_bert: bool,
    bert_model: Optional[BertModel] = None,
    embed_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    layer_dropout: Optional[float] = None,
):
    if is_bert and bert_model is None:
        raise ValueError("bert_model must be provided if is_bert is True")

    if not is_bert and (
        embed_dim is None
        or num_heads is None
        or num_layers is None
        or layer_dropout is None
    ):
        raise ValueError(
            f"embed_dim, num_heads, num_layers, and layer_dropout must be provided if is_bert is False (Received: embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}, layer_dropout={layer_dropout})"
        )

    return (
        TextEncoder(
            encoder=bert_model.encoder,
            is_bert=is_bert,
        )
        if is_bert
        else TextEncoder(
            encoder=Encoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                layer_dropout=layer_dropout,
            ),
            is_bert=is_bert,
        )
    )
