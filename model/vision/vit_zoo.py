from functools import partial
from typing import Callable, Dict, Tuple, Union

from configs import get_image_experiment_config
from model.vision.vit import VisionTransformer

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
    cfg = MODEL_SIZES[size]

    return (cfg["embed_dim"], cfg["num_layers"], cfg["num_heads"])


def make_vit_builder(size: str) -> Callable[..., VisionTransformer]:
    def builder(
        img_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        in_chans: int,
        num_frames: int,
        **kwargs,
    ) -> VisionTransformer:
        embed_dim, num_layers, num_heads = get_model_config(size)
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

    builder.__name__ = f"vit_{size}_builder"

    return builder


vit_model_builders: Dict[str, Callable[..., VisionTransformer]] = {
    size: make_vit_builder(size) for size in MODEL_SIZES
}


if __name__ == "__main__":
    from model.checksum import model_checksum
    from model.save_load_weights import load_model_weights, save_model_weights

    # 1. Save initial weights
    path_to_initial_weights: str = "./initial_weights/initial_vit_weights_tiny.pt"

    vit_tiny_builder: Callable[..., VisionTransformer] = vit_model_builders["tiny"]

    b_vit_init: VisionTransformer = save_model_weights(
        filepath=path_to_initial_weights,
        constructor=vit_tiny_builder,
        seed=experiment_config["SEED"],
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_frames=1,
    )
    print(f"✅ Model saved to {path_to_initial_weights}")

    b_vit_loaded = load_model_weights(
        filepath=path_to_initial_weights,
        constructor=vit_tiny_builder,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_frames=1,
    )
    print(f"✅ Model 1 loaded from {path_to_initial_weights}")

    u_vit_loaded = load_model_weights(
        filepath=path_to_initial_weights,
        constructor=vit_tiny_builder,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_frames=1,
    )
    print(f"✅ Model 2 loaded from {path_to_initial_weights}")

    assert (
        (checksum := model_checksum(b_vit_init))
        == model_checksum(b_vit_loaded)
        == model_checksum(u_vit_loaded)
    )

    print(f"✅ Model checksums match: {checksum}")
