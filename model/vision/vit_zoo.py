from typing import Callable, Dict, Optional, Tuple

import torch

from configs import get_image_experiment_config
from model.seed import seed_everything

from .vit import VisionTransformer

experiment_config = get_image_experiment_config()

MODEL_SIZES = {
    "nano": {"embed_dim": 128, "num_layers": 4, "num_heads": 4},
    "tiny": {"embed_dim": 192, "num_layers": 8, "num_heads": 4},
    "small": {"embed_dim": 384, "num_layers": 12, "num_heads": 8},
    "base": {"embed_dim": 768, "num_layers": 18, "num_heads": 12},
    "large": {"embed_dim": 1024, "num_layers": 24, "num_heads": 16},
    "huge": {"embed_dim": 1280, "num_layers": 32, "num_heads": 20},
    "gigantic": {"embed_dim": 1536, "num_layers": 48, "num_heads": 24},
}


def get_model_config(size: str) -> Tuple[int, int, int]:
    return (
        MODEL_SIZES[size]["embed_dim"],
        MODEL_SIZES[size]["num_layers"],
        MODEL_SIZES[size]["num_heads"],
    )


def vit_nano(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    embed_dim, num_layers, num_heads = get_model_config(size="nano")

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=embed_dim,
        enc_depth=num_layers,
        num_heads=num_heads,
        **kwargs,
    )


def vit_tiny(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    embed_dim, num_layers, num_heads = get_model_config(size="tiny")

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=embed_dim,
        enc_depth=num_layers,
        num_heads=num_heads,
        **kwargs,
    )


def vit_small(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    embed_dim, num_layers, num_heads = get_model_config(size="small")

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=embed_dim,
        enc_depth=num_layers,
        num_heads=num_heads,
        **kwargs,
    )


def vit_base(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    embed_dim, num_layers, num_heads = get_model_config(size="base")

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=embed_dim,
        enc_depth=num_layers,
        num_heads=num_heads,
        **kwargs,
    )


def vit_large(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    embed_dim, num_layers, num_heads = get_model_config(size="large")

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=embed_dim,
        enc_depth=num_layers,
        num_heads=num_heads,
        **kwargs,
    )


def vit_huge(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    embed_dim, num_layers, num_heads = get_model_config(size="huge")

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=embed_dim,
        enc_depth=num_layers,
        num_heads=num_heads,
        **kwargs,
    )


def vit_gigantic(img_size, patch_size=16, in_chans=3, num_frames=1, **kwargs):
    embed_dim, num_layers, num_heads = get_model_config(size="gigantic")

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        embed_dim=embed_dim,
        enc_depth=num_layers,
        num_heads=num_heads,
        **kwargs,
    )


vit_model_builders: Dict[
    str,
    Callable[..., VisionTransformer],
] = {
    "tiny": vit_tiny,
    "small": vit_small,
    "base": vit_base,
    "large": vit_large,
    "huge": vit_huge,
    "gigantic": vit_gigantic,
}


def save_initial_weights(
    filepath: str,
    constructor: Optional[Callable[..., VisionTransformer]] = vit_tiny,
    seed: Optional[int] = None,
) -> VisionTransformer:
    if seed is not None:
        seed_everything(seed)

    model, _, _ = constructor()
    torch.save(model.state_dict(), filepath)

    print(f"✅ Initial weights saved to '{filepath}'")

    return model


def load_initial_weights(
    filepath: str,
    constructor: Optional[Callable[..., VisionTransformer]] = vit_tiny,
) -> VisionTransformer:
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

    b_vit_init: VisionTransformer = save_initial_weights(
        filepath=path_to_initial_weights,
        constructor=vit_tiny,
        seed=experiment_config["SEED"],
    )
    print(f"✅ Model saved to {path_to_initial_weights}")

    b_vit_loaded = load_initial_weights(
        filepath=path_to_initial_weights,
        constructor=vit_tiny,
    )
    print(f"✅ Model 1 loaded from {path_to_initial_weights}")

    u_vit_loaded = load_initial_weights(
        filepath=path_to_initial_weights,
        constructor=vit_tiny,
    )
    print(f"✅ Model 2 loaded from {path_to_initial_weights}")

    assert (
        (checksum := checksum_model(b_vit_init))
        == checksum_model(b_vit_loaded)
        == checksum_model(u_vit_loaded)
    )

    print(f"✅ Model checksums match: {checksum}")
