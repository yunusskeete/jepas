import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from configs import (
    get_video_experiment_config,
    get_video_runtime_config,
    get_video_tracking_config,
    video_config,
)
from jepa_datasets import VideoDataModule, create_video_datamodule
from model import VJEPA
from model.video import vjepa_model_builders

# EXPERIMENT
experiment_config = get_video_experiment_config()
MODEL_NAME: str = experiment_config["MODEL_NAME"]
MODEL_SIZE: str = experiment_config["MODEL_SIZE"]
LR: float = experiment_config["LR"]
SEED: int = experiment_config["SEED"]
MAX_EPOCHS: int = experiment_config["MAX_EPOCHS"]
GRADIENT_CLIP_VAL: float = experiment_config["GRADIENT_CLIP_VAL"]

# TRACKING
tracking_config = get_video_tracking_config()
LOG_DIR: str = tracking_config["LOG_DIR"]
LOGGING_INTERVAL: str = tracking_config["LOGGING_INTERVAL"]
TOK_K_CHECKPOINTS: int = tracking_config["TOK_K_CHECKPOINTS"]
CHECKPOINT_DIR: str = tracking_config["CHECKPOINT_DIR"]
CHECKPOINT_MONITOR: str = tracking_config["CHECKPOINT_MONITOR"]
CHECKPOINT_MODE: str = tracking_config["CHECKPOINT_MODE"]
VAL_CHECK_INTERVAL: float = tracking_config["VAL_CHECK_INTERVAL"]

# RUNTIME
runtime_config = get_video_runtime_config()
ACCELERATOR: str = runtime_config["ACCELERATOR"]
DEVICES: int = runtime_config["DEVICES"]
PRECISION: int = runtime_config["PRECISION"]

if __name__ == "__main__":
    import gc

    import torch

    torch.set_float32_matmul_precision(runtime_config["FLOAT32_MATMUL_PRECISION"])

    # 1. Instantiate model with fixed initialisation
    model_id = f"{MODEL_SIZE}_{SEED}_{LR:.1e}-{MAX_EPOCHS}"
    model: VJEPA = vjepa_model_builders[MODEL_SIZE](
        video_config=video_config,
        seed=SEED,
    )
    print(f"✅ Model loaded: {model_id}")

    # 2. Load dataset
    datamodule: VideoDataModule = create_video_datamodule(video_config=video_config)
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

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,  # '32-true', 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16'
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=VAL_CHECK_INTERVAL,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)

    # 5. Save model
    trainer.save_checkpoint(f"checkpoints/{MODEL_NAME}_{model_id}.ckpt")

    # 6. Test
    trainer.test(model, datamodule=datamodule)

    # 7. Cleanup
    del datamodule
    gc.collect()
