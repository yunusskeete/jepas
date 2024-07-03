from typing import List, Tuple, Union

import torch


def validate_tensor_dimensions(
    size1: torch.Size,
    size2: torch.Size,
    dimension_checks: List[Tuple[Union[int, Tuple[int, int], str]]],
) -> None:
    for idxs, error_message in dimension_checks:
        if not isinstance(idxs, (int, tuple)):
            raise ValueError(f"Invalid index received: {idxs}")

        if isinstance(idxs, int) or len(idxs) == 1:
            idx1, idx2 = idxs, idxs
        else:
            idx1, idx2 = idxs

        assert (
            size1[idx1] == size2[idx2]
        ), f"{error_message}: {size1[idx1]} != {size2[idx2]}"
