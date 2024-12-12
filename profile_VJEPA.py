import datetime
from pathlib import Path

import torch
import torch.profiler
from torch.utils.data import DataLoader

from jepa_datasets import VideoDataset
from model import VJEPA

if __name__ == "__main__":
    dataset_path: Path = Path(
        "/mnt/data/video/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    PREFETCH_FACTOR: int = 4
    SHUFFLE: bool = False
    IMG_SIZE: int = 224
    FRAME_STEP: int = 8
    FRAMES_PER_CLIP: int = 8
    NUM_CLIPS: int = 2

    vjepa_dataset = VideoDataset(
        dataset_path=dataset_path,
        stage="test",
        num_clips=NUM_CLIPS,
        frame_step=FRAME_STEP,
        frames_per_clip=FRAMES_PER_CLIP,
    )
    print(f"{len(vjepa_dataset)=}")

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
            batch = [b.to(device) for b in batch]
            model.training_step(batch=batch, batch_idx=step)
            prof.step()

    profiler_results = prof.key_averages().table(  # pylint: disable=invalid-name
        sort_by="cpu_time_total", row_limit=10
    )

    timestamp = datetime.datetime.utcnow().isoformat().replace(":", "-")

    # Save the configurations and profiling results to a text file
    profiler_results_file: Path = Path(
        f"profiling/v-jepa/pretrain/profiler_output-{timestamp}.txt"
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
        f.write(f"NUM CLIPS: {NUM_CLIPS}\n")
        f.write(f"FRAME_STEP: {FRAME_STEP}\n")
        f.write(f"FRAMES_PER_CLIP: {FRAMES_PER_CLIP}\n")
        f.write("\n")

        # Save the profiling results
        f.write(profiler_results)

    print(profiler_results)
