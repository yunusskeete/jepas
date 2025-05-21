from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers.models.bert import BertEmbeddings
from x_transformers import Decoder
from x_transformers.x_transformers import ScaledSinusoidalEmbedding

from configs import text_experiment_config as experiment_config
from model.model import TJEPA
from model.seed import seed_everything

from .text_encoder import TextEncoder, create_text_encoder

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
    return MODEL_SIZES[size]["num_layers"], MODEL_SIZES[size]["num_heads"]


def create_tjepa_model(
    text_config: Dict[str, Any],
    num_layers: int,
    num_heads: int,
    num_layers_encoder: Optional[int] = None,
    num_heads_encoder: Optional[int] = None,
    layer_dropout_encoder: Optional[float] = None,
) -> TJEPA:
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


def tjepa_nano(
    text_config: Dict[str, Any], seed: Optional[int] = None
) -> Tuple[TJEPA, BertTokenizer, BertModel]:
    if seed is not None:
        seed_everything(seed)

    num_layers, num_heads = get_model_config(size="nano")

    return create_tjepa_model(
        text_config=text_config,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def tjepa_tiny(
    text_config: Dict[str, Any], seed: Optional[int] = None
) -> Tuple[TJEPA, BertTokenizer, BertModel]:
    if seed is not None:
        seed_everything(seed)

    num_layers, num_heads = get_model_config(size="tiny")

    return create_tjepa_model(
        text_config=text_config,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def tjepa_small(
    text_config: Dict[str, Any], seed: Optional[int] = None
) -> Tuple[TJEPA, BertTokenizer, BertModel]:
    if seed is not None:
        seed_everything(seed)

    num_layers, num_heads = get_model_config(size="small")

    return create_tjepa_model(
        text_config=text_config,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def tjepa_base(
    text_config: Dict[str, Any], seed: Optional[int] = None
) -> Tuple[TJEPA, BertTokenizer, BertModel]:
    if seed is not None:
        seed_everything(seed)

    num_layers, num_heads = get_model_config(size="base")

    return create_tjepa_model(
        text_config=text_config,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def tjepa_large(
    text_config: Dict[str, Any], seed: Optional[int] = None
) -> Tuple[TJEPA, BertTokenizer, BertModel]:
    if seed is not None:
        seed_everything(seed)

    num_layers, num_heads = get_model_config(size="large")

    return create_tjepa_model(
        text_config=text_config,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def tjepa_huge(
    text_config: Dict[str, Any], seed: Optional[int] = None
) -> Tuple[TJEPA, BertTokenizer, BertModel]:
    if seed is not None:
        seed_everything(seed)

    num_layers, num_heads = get_model_config(size="huge")

    return create_tjepa_model(
        text_config=text_config,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def tjepa_gigantic(
    text_config: Dict[str, Any], seed: Optional[int] = None
) -> Tuple[TJEPA, BertTokenizer, BertModel]:
    if seed is not None:
        seed_everything(seed)

    num_layers, num_heads = get_model_config(size="gigantic")

    return create_tjepa_model(
        text_config=text_config,
        num_layers=num_layers,
        num_heads=num_heads,
    )


tjepa_model_builders: Dict[
    str,
    Callable[..., Tuple[TJEPA, BertTokenizer, BertModel]],
] = {
    "tiny": tjepa_tiny,
    "small": tjepa_small,
    "base": tjepa_base,
    "large": tjepa_large,
    "huge": tjepa_huge,
    "gigantic": tjepa_gigantic,
}


def save_initial_weights(
    filepath: str,
    constructor: Optional[
        Callable[
            ...,
            Tuple[TJEPA, BertTokenizer, BertModel],
        ]
    ] = tjepa_tiny,
    seed: Optional[int] = None,
) -> TJEPA:
    if seed is not None:
        seed_everything(seed)

    model, _, _ = constructor()
    torch.save(model.state_dict(), filepath)

    print(f"✅ Initial weights saved to '{filepath}'")

    return model


def load_initial_weights(
    filepath: str,
    constructor: Optional[
        Callable[
            ...,
            Tuple[TJEPA, BertTokenizer, BertModel],
        ]
    ] = tjepa_tiny,
) -> TJEPA:
    model, _, _ = constructor()

    print(f"⏳ Loading initial weights from '{filepath}'")

    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)

    print(f"✅ Loaded initial weights from '{filepath}'")

    return model


if __name__ == "__main__":
    import hashlib

    def checksum_model(model: torch.nn.Module) -> str:
        all_params = torch.cat(
            [p.detach().flatten().cpu() for p in model.parameters() if p.requires_grad]
        )

        return hashlib.md5(all_params.numpy().tobytes()).hexdigest()

    # 1. Save initial weights
    path_to_initial_weights: str = "./initial_weights/initial_model_weights_tiny.pt"

    b_tjepa_init: TJEPA = save_initial_weights(
        filepath=path_to_initial_weights,
        constructor=tjepa_tiny,
        seed=experiment_config["SEED"],
    )
    print(f"✅ Model saved to {path_to_initial_weights}")

    b_tjepa_loaded = load_initial_weights(
        filepath=path_to_initial_weights,
        constructor=tjepa_tiny,
    )
    print(f"✅ Model 1 loaded from {path_to_initial_weights}")

    u_tjepa_loaded = load_initial_weights(
        filepath=path_to_initial_weights,
        constructor=tjepa_tiny,
    )
    print(f"✅ Model 2 loaded from {path_to_initial_weights}")

    assert (
        (checksum := checksum_model(b_tjepa_init))
        == checksum_model(b_tjepa_loaded)
        == checksum_model(u_tjepa_loaded)
    )

    print(f"✅ Model checksums match: {checksum}")
