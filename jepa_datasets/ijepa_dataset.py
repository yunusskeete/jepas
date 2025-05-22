"""
Usage:
```bash
python -m jepa_datasets.ijepa_dataset
```
"""

from pathlib import Path

from torch.utils.data import DataLoader

if __name__ == "__main__":
    from configs import get_image_config, get_image_dataset_config
    from jepa_datasets.image import (
        ImageDataModule,
        ImageDataset,
        create_image_datamodule,
    )

    image_config = get_image_config()
    image_config["experiment"]["NUM_WORKERS"] = 0
    image_config["experiment"]["PREFETCH_FACTOR"] = None
    image_config["experiment"]["PERSISTENT_WORKERS"] = False
    image_config["experiment"]["PIN_MEMORY"] = False
    dataset_config = get_image_dataset_config()

    dataset_path: Path = Path(dataset_config["DATASET_PATH"]).resolve()

    test_ijepa_loader = DataLoader(
        ImageDataset(dataset_path, stage="test"), batch_size=32, shuffle=False
    )
    print("✅ Test dataloader loaded")

    # Example of iterating through the test data
    image = next(iter(test_ijepa_loader))
    print(f"{image.shape=}")  # Should print torch.Size([32, 3, img_height, img_width])
    print("✅ Sample check passed")

    datamodule: ImageDataModule = create_image_datamodule(image_config=image_config)
    datamodule.setup()
    print("✅ Datamodule loaded")

    val_dataloader: DataLoader = datamodule.val_dataloader()
    print("✅ Dataloader loaded")

    # Example of iterating through the validation data
    for image in val_dataloader:
        print(
            f"{image.shape=}"
        )  # Should print torch.Size([32, 3, img_height, img_width])
        print("✅ Sample check passed")
        break
