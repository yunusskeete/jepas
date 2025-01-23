# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=line-too-long
# pylint: disable=unused-wildcard-import
from pathlib import Path
from typing import *

import numpy as np
import torch
from IPython.display import display
from PIL import Image, ImageDraw
from torchvision import transforms

from datasets.ijepa_dataset import ImageDataset
from datasets.vjepa_dataset import VideoDataset
from model import IJEPA, VJEPA
from model.patch_embed import PatchEmbed2D, PatchEmbed3D
from utils.types import Number

path_to_image_dataset: str = (
    "/home/yunusskeete/Documents/code/github/yunusskeete/jepa/ijepa/I-JEPA/data/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
)
image_dataset_path: Path = Path(path_to_image_dataset)
assert image_dataset_path.exists, f"Dataset does not exist: '{image_dataset_path}'"

path_to_video_dataset: str = "/mnt/data/video/kinetics-dataset/k400"
video_dataset_path: Path = Path(path_to_video_dataset)
assert video_dataset_path.exists, f"Dataset does not exist: '{video_dataset_path}'"

ids: VideoDataset = ImageDataset(dataset_path=image_dataset_path, stage="test")
vds: VideoDataset = VideoDataset(dataset_path=video_dataset_path, stage="test")

image_model = IJEPA(testing_purposes_only=True)
video_model = VJEPA(testing_purposes_only=True)

image: List[torch.Tensor] = ids[0]
print(f"{image.shape=}")  # (C, H, W)

clips: List[torch.Tensor] = vds[0]
print(f"{len(clips)=}")
clip: torch.Tensor = clips[0]
print(f"type(clip)={type(clip).__name__}")
print(f"{clip.shape=}")  # (C, T, H, W)
clip: torch.Tensor = clip.permute(1, 0, 2, 3)
print(f"{clip.shape=}")  # (T, C, H, W)


class Self:  # pylint: disable=function-redefined
    def __init__(
        self,
        patch_embed: Union[PatchEmbed2D, PatchEmbed3D],
        num_target_blocks: int,
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        target_scale_interval: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Number = 1,
        context_scale: Tuple[float, float] = (0.85, 1.0),
    ):
        self.patch_embed = patch_embed
        self.num_target_blocks = num_target_blocks
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale_interval = target_scale_interval
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale


self = Self(
    patch_embed=image_model.patch_embed, num_target_blocks=image_model.num_target_blocks
)


def unnormalise_image(image_tensor: torch.Tensor) -> np.array:
    # The normalization was done with mean and std as below:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # Step 1: Reverse the normalization
    unnormalised_image = image_tensor * std + mean
    # Step 2: Convert the tensor to a NumPy array
    unnormalised_image = unnormalised_image.numpy()
    # Step 3: Convert from (C, H, W) to (H, W, C) for PIL
    unnormalised_image = np.transpose(unnormalised_image, (1, 2, 0))
    # Step 4: Convert to a range of [0, 255] and to uint8 type
    unnormalised_image = (unnormalised_image * 255).astype(np.uint8)

    return unnormalised_image


# Function to convert patch index to pixel coordinates
def patch_idx_to_coords_2d(
    idx: int, patch_size: Tuple[int, int], image_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Args:
        idx (int): The patch index.
        patch_size (Tuple[int, int]): Size of the patch in (height, width).
        image_size (Tuple[int, int]): Size of the image in (height, width).

    Returns:
        Tuple[int, int]: The (height, width) pixel coordinates of the top-left corner of the patch.
    """

    # Number of patches along the height and width dimensions
    patches_per_row: int = image_size[1] // patch_size[1]  # Width patches (columns)

    # Calculate the position in the 2D grid of patches
    row = idx // patches_per_row  # Row in the grid
    col = idx % patches_per_row  # Column in the grid

    # Convert patch grid position to pixel coordinates
    height_coord = row * patch_size[0]
    width_coord = col * patch_size[1]

    return height_coord, width_coord


def visualise_patches_2d(self: Self, x: torch.Tensor):
    context_aspect_ratio = self.context_aspect_ratio

    # Generate random target and context aspect ratio and scale
    target_aspect_ratio: float = np.random.uniform(
        self.target_aspect_ratio[0], self.target_aspect_ratio[1]
    )
    target_scale: float = np.random.uniform(
        low=self.target_scale_interval[0], high=self.target_scale_interval[1]
    )

    context_scale: float = np.random.uniform(
        self.context_scale[0], self.context_scale[1]
    )

    print(f"{target_aspect_ratio=}")
    print(f"{target_scale=}")
    print(f"{context_scale=}")

    target_patches: List[List[int]]
    all_unique_target_patches: Set[int]
    target_patches, all_unique_target_patches = IJEPA.generate_target_patches(
        patch_dim=self.patch_embed.patch_shape,  # The number of patches in each dimension
        aspect_ratio=target_aspect_ratio,
        scale=target_scale,
        num_target_blocks=self.num_target_blocks,
    )

    context_patches: List[int] = IJEPA.generate_context_patches(
        patch_dim=self.patch_embed.patch_shape,
        aspect_ratio=context_aspect_ratio,
        scale=context_scale,
        target_patches_to_exclude=all_unique_target_patches,
    )

    assert all_unique_target_patches.isdisjoint(set(context_patches))

    print(f"{x.shape=}")  # torch.Size([3, 224, 224])

    unnormalised_image: np.array = unnormalise_image(image_tensor=x)

    # Step 5: Create a PIL image
    pil_image = Image.fromarray(unnormalised_image)

    draw = ImageDraw.Draw(pil_image)

    image_size: Tuple[int, int] = image.shape[-2], image.shape[-1]
    patch_size: Tuple[int, int] = self.patch_embed.patch_shape

    # Draw the context patches in blue
    for context_patch in context_patches:
        top_left = patch_idx_to_coords_2d(context_patch, patch_size, image_size)
        bottom_right = (top_left[0] + patch_size[0], top_left[1] + patch_size[1])
        draw.rectangle([top_left, bottom_right], outline="black", width=2, fill=None)

    # Draw the target patches in red
    for target_patch in all_unique_target_patches:
        top_left = patch_idx_to_coords_2d(target_patch, patch_size, image_size)
        bottom_right = (top_left[0] + patch_size[0], top_left[1] + patch_size[1])
        draw.rectangle([top_left, bottom_right], outline="green", width=2, fill=None)

    # Display the image
    display(pil_image)


# context_patches, target_patches, context_block, target_block =
visualise_patches_2d(self=self, x=image)
