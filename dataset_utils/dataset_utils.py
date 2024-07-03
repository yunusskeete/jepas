from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch

from utils.types import Number, ensure_tuple


def randomly_select_patch_start(
    patch_width: int,
    patch_height: int,
    block_width: int,
    block_height: int,
    seed: Optional[int] = None,
) -> int:
    if seed is not None:
        torch.manual_seed(seed)  # Set the random seed for reproducibility

    def random_coordinate(limit: int) -> int:
        return torch.randint(0, limit, (1,)).item()

    max_height = patch_height - block_height + 1
    max_width = patch_width - block_width + 1

    start_patch_height: int = random_coordinate(max_height)
    start_patch_width: int = random_coordinate(max_width)

    return start_patch_height * patch_width + start_patch_width


def generate_target_patches(
    patch_dim: Tuple[int, int],
    aspect_ratio: Number,
    scale: Number,
    num_target_blocks: int,
) -> Tuple[List[List[int]], List[int]]:
    """
    Generate target patches for each target block.

    Args:
        patch_dim (Tuple[int, int]): Dimensions of the patches (height, width).
        aspect_ratio (Number): Aspect ratio to be maintained for target blocks.
        scale (Number): Scaling factor for the number of patches in the target block.
        num_target_blocks (int): Number of target blocks to generate.

    Returns:
        Tuple[List[List[int]], List[int]]:
            - target_patches: A list of lists containing indices of patches for each target block.
            - all_patches: A list of all unique patches used in target blocks.
    """
    # Extract patch dimensions
    patch_h, patch_w = patch_dim

    # Calculate the number of patches in the target block
    num_patches_block: int = int(patch_h * patch_w * scale)
    # pylint: disable=pointless-string-statement
    """
    aspect_ratio = w / h
    num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
    h = (num_patches_block/aspect_ratio)**.5
    """

    # Calculate the height and width of the target block maintaining the aspect ratio
    block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
    block_w: int = int(aspect_ratio * block_h)

    # Initialize lists to hold target patches and all unique patches
    target_patches: List[List[int]] = []
    all_patches: List[int] = []

    # For each of the target blocks to generate
    for _ in range(num_target_blocks):
        start_patch: int = randomly_select_patch_start(
            patch_width=patch_w,
            patch_height=patch_h,
            block_width=block_w,
            block_height=block_h,
        )

        # Initialize list to hold the patches for the target block
        patches: List[int] = []
        # Collect patches within the target block
        for i in range(block_h):
            for j in range(block_w):
                patches.append(start_patch + i * patch_w + j)
                if start_patch + i * patch_w + j not in all_patches:
                    all_patches.append(start_patch + i * patch_w + j)

        # Store the patches for the current target block
        target_patches.append(patches)

    return target_patches, all_patches


def generate_context_patches(
    patch_dim: Tuple[int, int],
    aspect_ratio: Number,
    scale: Number,
    target_patches: List[int],
) -> List[int]:
    """
    Generate a list of patch indices for the context block, excluding target patches.

    Args:
        patch_dim (Tuple[int, int]): Dimensions of the patches (height, width).
        aspect_ratio (Number): Aspect ratio to be maintained for the context block.
        scale (Number): Scaling factor for the number of patches in the context block.
        target_patches (List[int]): List containing indices of target patches.

    Returns:
        List[int]: A list of patch indices for the context block excluding target patches.
    """
    # Extract patch dimensions
    patch_h, patch_w = patch_dim

    # Calculate the number of patches in the context block
    num_patches_block: int = int(patch_h * patch_w * scale)
    # pylint: disable=pointless-string-statement
    """
    aspect_ratio = w / h
    num_patches_block = h * (w) = h * (aspect_ratio * h) = aspect_ratio * h**2
    h = (num_patches_block/aspect_ratio)**.5
    """

    # Calculate the height and width of the context block maintaining the aspect ratio
    block_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
    block_w: int = int(aspect_ratio * block_h)

    # Randomly select the starting patch for the context block
    start_patch: int = randomly_select_patch_start(
        patch_width=patch_w,
        patch_height=patch_h,
        block_width=block_w,
        block_height=block_h,
    )

    return [
        start_patch + i * patch_w + j
        for i in range(block_h)
        for j in range(block_w)
        if start_patch + i * patch_w + j not in target_patches
    ]
