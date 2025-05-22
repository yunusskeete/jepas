from fractions import Fraction

from configs import get_video_dataset_transforms_config

from .transforms import VideoTransform
from .transforms import make_transforms as _make_transforms

transforms_config = get_video_dataset_transforms_config()


def make_transforms() -> VideoTransform:
    print(transforms_config["RANDOM_RESIZE_ASPECT_RATIO"])
    return _make_transforms(
        random_horizontal_flip=transforms_config["RANDOM_HORIZONTAL_FLIP"],
        random_resize_aspect_ratio=(
            Fraction(x) for x in transforms_config["RANDOM_RESIZE_ASPECT_RATIO"]
        ),
        random_resize_scale=(
            Fraction(x) for x in transforms_config["RANDOM_RESIZE_SCALE"]
        ),
        reprob=transforms_config["REPROB"],
        auto_augment=transforms_config["AUTO_AUGMENT"],
        motion_shift=transforms_config["MOTION_SHIFT"],
        crop_size=transforms_config["CROP_SIZE"],
    )
