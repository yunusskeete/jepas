"""
Usage:
```bash
python -m jepa_datasets.vjepa_dataset
```
"""

from pathlib import Path

from torch.utils.data import DataLoader

from configs import video_config
from configs import video_dataset_config as dataset_config
from configs import video_experiment_config as experiment_config
from jepa_datasets.video import VideoDataModule, VideoDataset, create_video_datamodule

if __name__ == "__main__":
    dataset_path: Path = Path(dataset_config["DATASET_PATH"]).resolve()

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

    # Example of iterating through the test data
    for video_clips in test_vjepa_loader:
        print(f"{type(video_clips)=}")
        print(f"{len(video_clips)=}")  # Should print 1
        print(
            f"{video_clips[0].shape=}"
        )  # Should print torch.Size([batch_size, num_channels=3, clip_length, img_height, img_width])
        break

    dataset: VideoDataModule = create_video_datamodule(video_config=video_config)
    dataset.setup()

    test_dataloader: DataLoader = dataset.test_dataloader()

    # Example of iterating through the test data
    for video_clips in test_dataloader:
        print(f"{type(video_clips)=}")
        print(f"{len(video_clips)=}")  # Should print 1
        print(
            f"{video_clips[0].shape=}"
        )  # Should print torch.Size([batch_size, num_channels=3, clip_length, img_height, img_width])
        break
