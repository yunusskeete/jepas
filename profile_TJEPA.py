import itertools
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs import get_text_config, get_text_dataset_config

dataset_config = get_text_dataset_config()
from configs import (
    get_text_experiment_config,
    get_text_profiling_config,
    get_text_runtime_config,
)
from jepa_datasets import PreTokenizedTextDataset, TextDataset
from model import TJEPA
from model.text.text_zoo import tjepa_model_builders
from profiling.profiling_utils import GPUUtilTracker, measure_throughput

# EXPERIMENT
experiment_config = get_text_experiment_config()
MODEL_NAME: str = experiment_config["MODEL_NAME"]
MODEL_SIZE: str = experiment_config["MODEL_SIZE"]
LR: float = experiment_config["LR"]
SEED: int = experiment_config["SEED"]

# PROFILING
profiling_config = get_text_profiling_config()
LOG_DIR = profiling_config["LOG_DIR"]
BATCH_SIZE_OPTIONS: List[int] = profiling_config["BATCH_SIZE_OPTIONS"]
NUM_WORKERS_OPTIONS: List[int] = profiling_config["NUM_WORKERS_OPTIONS"]
PIN_MEMORY_OPTIONS: List[bool] = profiling_config["PIN_MEMORY_OPTIONS"]
PERSISTENT_WORKERS_OPTIONS: List[bool] = profiling_config["PERSISTENT_WORKERS_OPTIONS"]
PREFETCH_FACTOR_OPTIONS: List[int] = profiling_config["PREFETCH_FACTOR_OPTIONS"]


if __name__ == "__main__":
    import gc
    import json

    import torch

    text_config = get_text_config()
    runtime_config = get_text_runtime_config()

    torch.set_float32_matmul_precision(runtime_config["FLOAT32_MATMUL_PRECISION"])

    # 1. Instantiate model with fixed initialisation
    model_id = f"{MODEL_SIZE}_{SEED}_{LR:.1e}"
    model: TJEPA = tjepa_model_builders[MODEL_SIZE]()
    print(f"✅ Model loaded: {model_id}")

    # 2. Load dataset
    use_pre_tokenized_dataset: bool = dataset_config["USE_PRE_TOKENIZED_DATASET"]
    dataset_id: str = (
        dataset_config["TOKENIZED_DATASET_NAME"]
        if use_pre_tokenized_dataset
        else dataset_config["UNTOKENIZED_DATASET_NAME"]
    )
    full_dataset: Dataset = (
        load_dataset(dataset_id, split=dataset_config["UNTOKENIZED_DATASET_SPLIT"])
        if not use_pre_tokenized_dataset
        else load_from_disk(dataset_id)
    )

    splits: DatasetDict = full_dataset.train_test_split(
        test_size=dataset_config["TEST_SPLIT"],
        seed=SEED,
    )

    train_fraction: float = dataset_config["DATASET_TRAIN_FRACTION"]
    if train_fraction < 1.0:
        num_train = int(len(splits["train"]) * train_fraction)
        splits["train"] = splits["train"].select(range(num_train))

    dataset_cls = PreTokenizedTextDataset if use_pre_tokenized_dataset else TextDataset
    dataset = dataset_cls(splits["train"])

    # 3. Sweep Parameters
    fastest_config: Optional[Dict[str, int]] = None
    fastest_ips: float = 0.0

    optim_config: Optional[Dict[str, int]] = None
    best_avg_mem_util: int = 0

    log_dir: str = LOG_DIR.rstrip("/") + f"/{MODEL_NAME}"
    writer = SummaryWriter(log_dir)

    for run_id, (
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    ) in enumerate(
        itertools.product(
            BATCH_SIZE_OPTIONS,
            NUM_WORKERS_OPTIONS,
            PIN_MEMORY_OPTIONS,
            PERSISTENT_WORKERS_OPTIONS,
            PREFETCH_FACTOR_OPTIONS,
        )
    ):
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                shuffle=dataset_config["SHUFFLE_DATASET"],
            )

            gpu_util_tracker = GPUUtilTracker()

            # Track GPU utilisation
            with GPUUtilTracker() as tracker:
                ips = measure_throughput(
                    model=model,
                    dataloader=dataloader,
                    batch_size=batch_size,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    precision=runtime_config["PRECISION"],
                    steps=profiling_config["PROFILING_STEPS"],
                )

                util_results = tracker.result()

            max_gpu_util = util_results["max_gpu_util"]
            max_mem_util = util_results["max_mem_util"]
            avg_gpu_util = util_results["avg_gpu_util"]
            avg_mem_util = util_results["avg_mem_util"]

            print(
                f"IPS: {ips:.2f}, GPU: max={max_gpu_util}% (avg={avg_gpu_util}%) Memory: max={max_mem_util}% (avg={avg_mem_util}%) | batch_size={batch_size} workers={num_workers} pin_memory={pin_memory} persistent={persistent_workers} prefetch={prefetch_factor}"
            )

            config: Dict[str, int] = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": int(pin_memory),
                "persistent_workers": int(persistent_workers),
                "prefetch_factor": prefetch_factor,
                "ips": ips,
                "max_gpu_util": max_gpu_util,
                "max_mem_util": max_mem_util,
                "avg_gpu_util": avg_gpu_util,
                "avg_mem_util": avg_mem_util,
            }

            # Log the hyperparameters
            writer.add_hparams(
                config,
                {"hparam/iterations_per_second": ips},
                run_name=f"run_{run_id}",
            )

            if ips > fastest_ips:
                fastest_ips = ips
                fastest_config = config

            if avg_mem_util > best_avg_mem_util:
                best_avg_mem_util = avg_mem_util
                optim_config = config

        except Exception as e:
            print(f"Failed config: {e}")

    writer.close()

    # 5. Print fastest config
    print("\n⚡️ Fastest Configuration:")
    print(f"IPS: {fastest_config['ips']}")
    print(f"{fastest_config=}")

    # 6. Print optim config
    print("\n🏆 Optim Configuration:")
    print(f"IPS: {optim_config['ips']}")
    print(f"{optim_config=}")

    # 7. Clean up
    gc.collect()
    torch.cuda.empty_cache()

    print("✅ Profiling complete")

    # 8. Save results
    with open(log_dir + "/fastest_config.json", "w", encoding="utf-8") as f:
        json.dump(fastest_config, f)

    with open(log_dir + "/optim_config.json", "w", encoding="utf-8") as f:
        json.dump(optim_config, f)

    print("✅ Results saved")
