from typing import List, Optional

from torchvision import transforms

from configs import get_image_dataset_transforms_config

transforms_config = get_image_dataset_transforms_config()

transforms_pre_filtering: List[Optional[transforms.Compose]] = [
    (
        transforms.RandomResizedCrop(random_resize_crop)
        if (random_resize_crop := transforms_config.get("RANDOM_RESIZE_CROP"))
        else None
    ),
    (
        transforms.RandomHorizontalFlip()
        if transforms_config.get("RANDOM_HORIZONTAL_FLIP")
        else None
    ),
    transforms.ToTensor() if transforms_config.get("TO_TENSOR") else None,
    (  # Normalize using ImageNet mean and std
        transforms.Normalize(mean=mean, std=std)
        if (
            (mean := transforms_config.get("NORMALIZE", {}).get("MEAN"))
            and (std := transforms_config.get("NORMALIZE", {}).get("STD"))
        )
        else None
    ),
]


def make_transforms() -> List[transforms.Compose]:
    filtered_transforms: List[transforms.Compose] = [
        x for x in transforms_pre_filtering if x is not None
    ]

    return filtered_transforms
