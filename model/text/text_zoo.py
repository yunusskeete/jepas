from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings
from x_transformers import Decoder
from x_transformers.x_transformers import ScaledSinusoidalEmbedding

from configs import get_text_config, get_text_experiment_config
from model.model import TJEPA
from model.save_load_weights import load_model_weights, save_model_weights
from model.seed import seed_everything
from model.text.text_encoder import TextEncoder, create_text_encoder

experiment_config = get_text_experiment_config()

# pylint: disable=redefined-outer-name

MODEL_SIZES = {
    "nano": {"num_layers": 4, "num_heads": 4},
    "tiny": {"num_layers": 6, "num_heads": 8},
    "small": {"num_layers": 8, "num_heads": 12},
    "base": {"num_layers": 12, "num_heads": 12},
    "large": {"num_layers": 24, "num_heads": 16},
    "huge": {"num_layers": 36, "num_heads": 20},
    "gigantic": {"num_layers": 48, "num_heads": 24},
}


def get_model_config(size: str) -> Tuple[int, int]:
    cfg = MODEL_SIZES[size]

    return cfg["num_layers"], cfg["num_heads"]


def create_tjepa_model(
    num_layers: int,
    num_heads: int,
    num_layers_encoder: Optional[int] = None,
    num_heads_encoder: Optional[int] = None,
    layer_dropout_encoder: Optional[float] = None,
) -> TJEPA:
    text_config: Dict[str, Any] = get_text_config()

    # Load pretrained BERT tokeniser
    tokeniser = BertTokenizer.from_pretrained(text_config["model"]["BERT_MODEL_NAME"])
    bert_model: BertModel = BertModel.from_pretrained(
        text_config["model"]["BERT_MODEL_NAME"]
    )
    embed_dim: int = bert_model.config.hidden_size
    vocab_size: int = tokeniser.vocab_size

    pos_embedding_layer: nn.Module = ScaledSinusoidalEmbedding(embed_dim)

    embedder: Union[BertEmbeddings, nn.Embedding] = (
        bert_model.embeddings
        if text_config["model"]["USE_PRETRAINED_TEXT_EMBEDDINGS"]
        else nn.Embedding(vocab_size, embed_dim)
    )

    is_bert: bool = text_config["model"]["USE_PRETRAINED_TEXT_ENCODER"]
    encoder: TextEncoder = create_text_encoder(
        is_bert=is_bert,
        bert_model=bert_model,
        embed_dim=embed_dim,
        num_heads=num_heads_encoder,
        num_layers=num_layers_encoder,
        layer_dropout=layer_dropout_encoder,
    )

    decoder = Decoder(
        dim=embed_dim,
        depth=num_layers,
        heads=num_heads,
        layer_dropout=text_config["model"]["LAYER_DROPOUT"],
        # Causal attn
    )

    model = TJEPA(
        tokenizer=tokeniser,
        pos_embedding_layer=pos_embedding_layer,
        embedder=embedder,
        encoder=encoder,
        decoder=decoder,
        max_length=text_config["model"]["MAX_LENGTH"],
        lr=text_config["experiment"]["LR"],
        weight_decay=text_config["experiment"]["WEIGHT_DECAY"],
        lr_warmup_fraction=text_config["experiment"]["LR_WARMUP_FRACTION"],
        using_pre_tokenized_dataset=text_config["dataset"]["USE_PRE_TOKENIZED_DATASET"],
        target_aspect_ratio=text_config["model"]["TARGET_ASPECT_RATIO"],
        target_scale_interval=text_config["model"]["TARGET_SCALE_INTERVAL"],
        context_aspect_ratio=text_config["model"]["CONTEXT_ASPECT_RATIO"],
        context_scale=text_config["model"]["CONTEXT_SCALE"],
        m=text_config["model"]["M"],
        momentum_limits=text_config["model"]["MOMENTUM_LIMITS"],
    )

    return model, tokeniser, bert_model


def make_tjepa_builder(
    size: str,
) -> Callable[[Dict[str, Any], Optional[int]], Tuple[TJEPA, BertTokenizer, BertModel]]:
    def builder(
        seed: Optional[int] = None,
    ) -> Tuple[TJEPA, BertTokenizer, BertModel]:
        if seed is not None:
            seed_everything(seed)

        num_layers, num_heads = get_model_config(size)
        tjepa, _, _ = create_tjepa_model(
            num_layers=num_layers,
            num_heads=num_heads,
        )

        return tjepa

    builder.__name__ = f"tjepa_{size}_builder"

    return builder


tjepa_model_builders: Dict[
    str, Callable[..., Tuple[TJEPA, BertTokenizer, BertModel]]
] = {size: make_tjepa_builder(size) for size in MODEL_SIZES}


if __name__ == "__main__":
    from model.checksum import model_checksum
    from model.save_load_weights import load_model_weights, save_model_weights

    # 1. Save initial weights
    path_to_initial_weights: str = "./initial_weights/initial_tjepa_weights_tiny.pt"

    tjepa_tiny_builder: Callable[..., Tuple[TJEPA, BertTokenizer, BertModel]] = (
        tjepa_model_builders["tiny"]
    )

    b_tjepa_init: TJEPA = save_model_weights(
        filepath=path_to_initial_weights,
        constructor=tjepa_tiny_builder,
        seed=experiment_config["SEED"],
    )
    print(f"✅ Model saved to {path_to_initial_weights}")

    b_tjepa_loaded = load_model_weights(
        filepath=path_to_initial_weights,
        constructor=tjepa_tiny_builder,
    )
    print(f"✅ Model 1 loaded from {path_to_initial_weights}")

    u_tjepa_loaded = load_model_weights(
        filepath=path_to_initial_weights,
        constructor=tjepa_tiny_builder,
    )
    print(f"✅ Model 2 loaded from {path_to_initial_weights}")

    assert (
        (checksum := model_checksum(b_tjepa_init))
        == model_checksum(b_tjepa_loaded)
        == model_checksum(u_tjepa_loaded)
    )

    print(f"✅ Model checksums match: {checksum}")
