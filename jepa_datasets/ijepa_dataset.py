"""
Usage:
```bash
python -m jepa_datasets.ijepa_dataset
```
"""

from pathlib import Path

from torch.utils.data import DataLoader

from configs import image_config
from configs import image_dataset_config as dataset_config
from jepa_datasets.image import ImageDataModule, ImageDataset, create_image_datamodule

if __name__ == "__main__":
    dataset_path: Path = Path(dataset_config["DATASET_PATH"]).resolve()

    test_ijepa_loader = DataLoader(
        ImageDataset(dataset_path, stage="test"), batch_size=32, shuffle=False
    )

    # Example of iterating through the test data
    for image in test_ijepa_loader:
        print(
            f"{image.shape=}"
        )  # Should print torch.Size([32, 3, img_height, img_width])
        break

    datamodule: ImageDataModule = create_image_datamodule(image_config=image_config)
    datamodule.setup()

    val_dataloader: DataLoader = datamodule.val_dataloader()

    # Example of iterating through the validation data
    for image in val_dataloader:
        print(
            f"{image.shape=}"
        )  # Should print torch.Size([32, 3, img_height, img_width])
        break
