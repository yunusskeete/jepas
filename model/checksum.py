import hashlib

import torch


def model_checksum(model: torch.nn.Module) -> str:
    all_params = torch.cat(
        [p.detach().flatten().cpu() for p in model.parameters() if p.requires_grad]
    )

    return hashlib.md5(all_params.numpy().tobytes()).hexdigest()
