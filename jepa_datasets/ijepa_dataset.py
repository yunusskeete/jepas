"""
Usage:
```bash
python -m jepa_datasets.ijepa_dataset
```
"""

from pathlib import Path

from torch.utils.data import DataLoader

if __name__ == "__main__":
    import gc

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

    # 1. Load test dataset
    test_dataset: ImageDataset = ImageDataset(dataset_path, stage="test")
    print("✅ Test dataset loaded")

    # 2. Load test dataloader
    test_ijepa_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("✅ Test dataloader loaded")

    # 3. Iterate through the test data
    image = next(iter(test_ijepa_loader))
    print(f"{image.shape=}")  # Should print torch.Size([32, 3, img_height, img_width])
    print("✅ Sample check passed")

    # 4. Load datamodule
    datamodule: ImageDataModule = create_image_datamodule(image_config=image_config)
    print("✅ Datamodule loaded")

    # 5. Load validation dataloader
    datamodule.setup()

    val_dataloader: DataLoader = datamodule.val_dataloader()
    print("✅ Dataloader loaded")

    # 6. Iterate through the validation data
    for image in val_dataloader:
        print(
            f"{image.shape=}"
        )  # Should print torch.Size([32, 3, img_height, img_width])
        print("✅ Sample check passed")
        break

    # 7. Cleanup
    del test_dataset
    del test_ijepa_loader
    del datamodule
    del val_dataloader
    gc.collect()
