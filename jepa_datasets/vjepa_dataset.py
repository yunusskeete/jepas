"""
Usage:
```bash
python -m jepa_datasets.vjepa_dataset
```
"""

from pathlib import Path

from torch.utils.data import DataLoader

if __name__ == "__main__":
    import gc

    from configs import (
        get_video_config,
        get_video_dataset_config,
        get_video_experiment_config,
    )
    from jepa_datasets.video import (
        VideoDataModule,
        VideoDataset,
        create_video_datamodule,
    )

    video_config = get_video_config()
    video_config["experiment"]["NUM_WORKERS"] = 0
    video_config["experiment"]["PREFETCH_FACTOR"] = None
    video_config["experiment"]["PERSISTENT_WORKERS"] = False
    video_config["experiment"]["PIN_MEMORY"] = False
    dataset_config = get_video_dataset_config()
    experiment_config = get_video_experiment_config()

    dataset_path: Path = Path(dataset_config["DATASET_PATH"]).resolve()

    # 1. Load test dataloader
    test_vjepa_loader = DataLoader(
        VideoDataset(
            dataset_path,
            stage="test",
            frames_per_clip=experiment_config["FRAMES_PER_CLIP"],
            frame_step=experiment_config["FRAME_STEP"],
            num_clips=experiment_config["NUM_CLIPS"],
            shuffle=False,
        ),
        batch_size=32,
        shuffle=False,
    )
    print("✅ Test dataloader loaded")

    # 2. Iterate through the test data
    video_clips = next(iter(test_vjepa_loader))
    print(f"{type(video_clips)=}")
    print(f"{len(video_clips)=}")  # Should print 1
    print(
        f"{video_clips[0].shape=}"
    )  # Should print torch.Size([batch_size, num_channels=3, clip_length, img_height, img_width])
    print("✅ Sample check passed")

    # 3. Load datamodule
    dataset: VideoDataModule = create_video_datamodule(video_config=video_config)
    dataset.setup()
    print("✅ Datamodule loaded")

    # 4. Load dataloader
    test_dataloader: DataLoader = dataset.test_dataloader()
    print("✅ Dataloader loaded")

    # 5. Iterate through the test data
    for video_clips in test_dataloader:
        print(f"{type(video_clips)=}")
        print(f"{len(video_clips)=}")  # Should print 1
        print(
            f"{video_clips[0].shape=}"
        )  # Should print torch.Size([batch_size, num_channels=3, clip_length, img_height, img_width])
        print("✅ Sample check passed")
        break

    # 6. Cleanup
    del test_vjepa_loader
    del dataset
    del test_dataloader
    gc.collect()
