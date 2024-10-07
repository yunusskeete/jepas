import datetime
from pathlib import Path

import torch
import torch.profiler
from torch.utils.data import DataLoader

from datasets import VideoDataset
from model import VJEPA

if __name__ == "__main__":
    dataset_path: Path = Path(
        "/mnt/data/video/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    vjepa_dataset = VideoDataset(dataset_path, stage="test")
    print(f"{len(vjepa_dataset)=}")

    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    PREFETCH_FACTOR: int = 4
    SHUFFLE: bool = False

    vjepa_dataloader = DataLoader(
        vjepa_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=SHUFFLE,
    )

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VJEPA(lr=4e-3, num_frames=vjepa_dataset.frames_per_clip).to(device)

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
        for step, batch in enumerate(vjepa_dataloader):
            if step >= 20:
                break
            batch: torch.Tensor = batch.to(device)
            model.training_step(batch=batch, batch_idx=step)
            prof.step()

    profiler_results = prof.key_averages().table(  # pylint: disable=invalid-name
        sort_by="cpu_time_total", row_limit=10
    )

    # Save the configurations and profiling results to a text file
    profiler_results_file: Path = Path(
        f"profiling/v-jepa/profiler_output-{datetime.datetime.utcnow().isoformat()}.txt"
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
