import datetime
import os
from pathlib import Path

import torch
import torch.profiler
from torch.utils.data import DataLoader

from finetune_VJEPA import VJEPA_FT
from jepa_datasets import VideoDataset
from model import VJEPA

if __name__ == "__main__":
    dataset_path: Path = Path(
        "E:/ahmad/kinetics-dataset/k400"
    ).resolve()  # Path to Kinetics dataset

    BATCH_SIZE: int = 2
    NUM_WORKERS: int = os.cpu_count() // 2
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    PREFETCH_FACTOR: int = 4
    SHUFFLE: bool = False
    IMG_SIZE: int = 224
    FRAME_COUNT: int = 8
    NUM_CLIPS: int = 2

    vjepa_dataset = VideoDataset(
        dataset_path=dataset_path,
        stage="test",
        num_clips=NUM_CLIPS,
        frame_step=FRAME_COUNT,
        frames_per_clip=FRAME_COUNT,
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

    model = VJEPA_FT(
        pretrained_model_path="D:/MDX/Thesis/suaijd/jepa/lightning_logs/v-jepa/pretrain/videos/version_1/checkpoints/epoch=1-step=241258.ckpt",
        frame_count=FRAME_COUNT,
        output_channels=3,
        output_height=IMG_SIZE,
        output_width=IMG_SIZE,
    ).to(device)

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
        f"profiling/v-jepa/profiler_output-{timestamp}.txt"
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
        f.write("\n")

        # Save the profiling results
        f.write(profiler_results)

    print(profiler_results)
