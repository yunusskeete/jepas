from typing import Any, Callable, Dict, Optional

import torch

from configs import get_image_experiment_config, get_image_model_config
from model.model import VJEPA
from model.seed import seed_everything
from model.vision import get_model_config as get_vit_config

experiment_config = get_image_experiment_config()
model_config = get_image_model_config()

MODEL_SIZES = {
    "nano": {"num_decoder_layers": 2},
    "tiny": {"num_decoder_layers": 4},
    "small": {"num_decoder_layers": 4},
    "base": {"num_decoder_layers": 8},
    "large": {"num_decoder_layers": 8},
    "huge": {"num_decoder_layers": 12},
    "gigantic": {"num_decoder_layers": 16},
}


def get_model_config(size: str) -> int:
    return MODEL_SIZES[size]["num_decoder_layers"]


def create_vjepa_model(vit_size: str) -> VJEPA:
    embed_dim, enc_depth, num_heads = get_vit_config(size=vit_size)
    num_layers_decoder: int = get_model_config(size=vit_size)

    return VJEPA(
        decoder_depth=num_layers_decoder,
        lr=experiment_config["LR"],
        weight_decay=experiment_config["WEIGHT_DECAY"],
        target_aspect_ratio=experiment_config["TARGET_ASPECT_RATIO"],
        target_scale_interval=experiment_config["TARGET_SCALE_INTERVAL"],
        context_aspect_ratio=experiment_config["CONTEXT_ASPECT_RATIO"],
        context_scale=experiment_config["CONTEXT_SCALE"],
        num_target_blocks=experiment_config["NUM_TARGET_BLOCKS"],
        m=experiment_config["M"],
        momentum_limits=experiment_config["MOMENTUM_LIMITS"],
        img_size=model_config["IMAGE_SIZE"],
        patch_size=model_config["PATCH_SIZE"],
        num_frames=model_config["NUM_FRAMES"],
        tubelet_size=model_config["TUBELET_SIZE"],
        in_chans=model_config["IN_CHANS"],
        embed_dim=embed_dim,
        enc_depth=enc_depth,
        num_heads=num_heads,
        post_emb_norm=model_config["POST_EMBED_NORM"],
        post_enc_norm=model_config["POST_ENCODE_NORM"],
        layer_dropout=model_config["LAYER_DROPOUT"],
    )


def vjepa_nano() -> VJEPA:
    return create_vjepa_model(vit_size="nano")


def vjepa_tiny() -> VJEPA:
    return create_vjepa_model(vit_size="tiny")


def vjepa_small() -> VJEPA:
    return create_vjepa_model(vit_size="small")


def vjepa_base() -> VJEPA:
    return create_vjepa_model(vit_size="base")


def vjepa_large() -> VJEPA:
    return create_vjepa_model(vit_size="large")


def vjepa_huge() -> VJEPA:
    return create_vjepa_model(vit_size="huge")


def vjepa_gigantic() -> VJEPA:
    return create_vjepa_model(vit_size="gigantic")


vjepa_model_builders: Dict[
    str,
    Callable[..., VJEPA],
] = {
    "tiny": vjepa_tiny,
    "small": vjepa_small,
    "base": vjepa_base,
    "large": vjepa_large,
    "huge": vjepa_huge,
    "gigantic": vjepa_gigantic,
}


def save_initial_weights(
    filepath: str,
    constructor: Optional[Callable[..., VJEPA]] = vjepa_tiny,
    seed: Optional[int] = None,
) -> VJEPA:
    if seed is not None:
        seed_everything(seed)

    model, _, _ = constructor()
    torch.save(model.state_dict(), filepath)

    print(f"✅ Initial weights saved to '{filepath}'")

    return model


def load_initial_weights(
    filepath: str,
    constructor: Optional[Callable[..., VJEPA]] = vjepa_tiny,
) -> VJEPA:
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

    b_tjepa_init: VJEPA = save_initial_weights(
        filepath=path_to_initial_weights,
        constructor=vjepa_tiny,
        seed=experiment_config["SEED"],
    )
    print(f"✅ Model saved to {path_to_initial_weights}")

    b_tjepa_loaded = load_initial_weights(
        filepath=path_to_initial_weights,
        constructor=vjepa_tiny,
    )
    print(f"✅ Model 1 loaded from {path_to_initial_weights}")

    u_tjepa_loaded = load_initial_weights(
        filepath=path_to_initial_weights,
        constructor=vjepa_tiny,
    )
    print(f"✅ Model 2 loaded from {path_to_initial_weights}")

    assert (
        (checksum := checksum_model(b_tjepa_init))
        == checksum_model(b_tjepa_loaded)
        == checksum_model(u_tjepa_loaded)
    )

    print(f"✅ Model checksums match: {checksum}")
