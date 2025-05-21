from fractions import Fraction
from typing import List, Optional

from torchvision import transforms

from configs import video_dataset_transforms_config as transforms_config

from .transforms import VideoTransform
from .transforms import make_transforms as _make_transforms


def make_transforms() -> VideoTransform:
    return _make_transforms(
        random_horizontal_flip=transforms_config["RANDOM_HORIZONTAL_FLIP"],
        random_resize_aspect_ratio=Fraction(
            transforms_config["RANDOM_RESIZE_ASPECT_RATIO"]
        ),
        random_resize_scale=transforms_config["RANDOM_RESIZE_SCALE"],
        reprob=transforms_config["REPROB"],
        auto_augment=transforms_config["AUTO_AUGMENT"],
        motion_shift=transforms_config["MOTION_SHIFT"],
        crop_size=transforms_config["CROP_SIZE"],
    )
