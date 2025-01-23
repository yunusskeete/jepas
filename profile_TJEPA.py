import datetime
from pathlib import Path
from typing import Optional

import datasets
import torch
import torch.profiler
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from jepa_datasets import HuggingfaceDatasetWrapper
from model import TJEPA

if __name__ == "__main__":
    hf_dataset_name: str = "Skylion007/openwebtext"
    hf_dataset: datasets.arrow_dataset.Dataset = load_dataset(
        hf_dataset_name, split="train"
    )

    tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")

    tjepa_dataset = HuggingfaceDatasetWrapper(
        hf_dataset,
        tokeniser=tokeniser,
        max_length=128,
        return_attn_masks=False,
    )

    print(f"{len(tjepa_dataset)=}")

    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    PREFETCH_FACTOR: Optional[int] = 4
    SHUFFLE: bool = False

    tjepa_dataloader = DataLoader(
        tjepa_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=SHUFFLE,
    )

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TJEPA(lr=3e-3).to(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        profile_memory=True,
        # with_stack=True,
    ) as prof:
        for step, batch in enumerate(tjepa_dataloader):
            if step >= 200:
                break
            batch = batch.to(device)
            model.training_step(batch=batch, batch_idx=step)
            prof.step()

    profiler_results = prof.key_averages().table(  # pylint: disable=invalid-name
        sort_by="cpu_time_total", row_limit=10
    )

    # Save the configurations and profiling results to a text file
    profiler_results_file: Path = Path(
        f"profiling/t-jepa/profiler_output-{datetime.datetime.utcnow().isoformat()}.txt"
    )
    profiler_results_file.parent.mkdir(exist_ok=True, parents=True)
    with open(
        profiler_results_file,
        "w",
        encoding="utf-8",
    ) as f:
        # Save the DataLoader configuration parameters
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
        f.write(f"PIN_MEMORY: {PIN_MEMORY}\n")
        f.write(f"PERSISTENT_WORKERS: {PERSISTENT_WORKERS}\n")
        f.write(f"PREFETCH_FACTOR: {PREFETCH_FACTOR}\n")
        f.write(f"SHUFFLE: {SHUFFLE}\n")
        f.write("\n")

        # Save the profiling results
        f.write(profiler_results)

    print(profiler_results)
