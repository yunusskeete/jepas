from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from configs import (
    get_text_config,
    get_text_dataset_config,
    get_text_experiment_config,
    get_text_runtime_config,
    get_text_tracking_config,
)
from jepa_datasets import (
    TextDataModule,
    create_text_datamodule,
    dynamic_padding_collate,
)
from model.text import tjepa_model_builders

# EXPERIMENT
experiment_config = get_text_experiment_config()
MODEL_NAME: str = experiment_config["MODEL_NAME"]
MODEL_SIZE: str = experiment_config["MODEL_SIZE"]
LR: float = experiment_config["LR"]
SEED: int = experiment_config["SEED"]
MAX_EPOCHS: int = experiment_config["MAX_EPOCHS"]
GRADIENT_CLIP_VAL: float = experiment_config["GRADIENT_CLIP_VAL"]
USE_ENCODER: bool = experiment_config["USE_ENCODER"]

# TRACKING
tracking_config = get_text_tracking_config()
LOG_DIR: str = tracking_config["LOG_DIR"]
LOGGING_INTERVAL: str = tracking_config["LOGGING_INTERVAL"]
TOK_K_CHECKPOINTS: int = tracking_config["TOK_K_CHECKPOINTS"]
CHECKPOINT_DIR: str = tracking_config["CHECKPOINT_DIR"]
CHECKPOINT_MONITOR: str = tracking_config["CHECKPOINT_MONITOR"]
CHECKPOINT_MODE: str = tracking_config["CHECKPOINT_MODE"]
VAL_CHECK_INTERVAL: float = tracking_config["VAL_CHECK_INTERVAL"]

# RUNTIME
runtime_config = get_text_runtime_config()
ACCELERATOR: str = runtime_config["ACCELERATOR"]
DEVICES: int = runtime_config["DEVICES"]
PRECISION: int = runtime_config["PRECISION"]

# DATASET
dataset_config = get_text_dataset_config()
DATASET_TRAIN_FRACTION: float = dataset_config["DATASET_TRAIN_FRACTION"]

if __name__ == "__main__":
    import gc

    import torch

    text_config = get_text_config()

    torch.set_float32_matmul_precision(runtime_config["FLOAT32_MATMUL_PRECISION"])

    # 1. Instantiate model with fixed initialisation
    model_id = f"{MODEL_SIZE}_{SEED}_{LR:.1e}-{MAX_EPOCHS}_{DATASET_TRAIN_FRACTION}-{'enc' if USE_ENCODER else 'emb'}"
    model, _, _ = tjepa_model_builders[MODEL_SIZE](
        text_config=text_config,
        seed=SEED,
    )
    print(f"✅ Model loaded: {model_id}")

    # 2. Load dataset
    datamodule: TextDataModule = create_text_datamodule(
        text_config=text_config, collate_fn=dynamic_padding_collate
    )
    print("✅ Dataset loaded")

    # 3. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=MODEL_NAME,
        monitor=CHECKPOINT_MONITOR,
        mode=CHECKPOINT_MODE,
        save_top_k=TOK_K_CHECKPOINTS,
    )
    lr_monitor = LearningRateMonitor(logging_interval=LOGGING_INTERVAL)

    # 4. Train
    logger = TensorBoardLogger(save_dir=LOG_DIR, name=MODEL_NAME, version=model_id)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=VAL_CHECK_INTERVAL,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)

    # 5. Save
    trainer.save_checkpoint(f"checkpoints/{MODEL_NAME}_{model_id}.ckpt")

    # 6. Test
    trainer.test(model, datamodule=datamodule)

    # 7. Cleanup
    del datamodule
    gc.collect()
