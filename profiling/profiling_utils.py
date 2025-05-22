import threading
import time
from typing import Dict, List, Union

import pynvml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


class GPUUtilTracker:
    def __init__(self, device_index: int = 0, interval: float = 0.1):
        self.device_index = device_index
        self.interval = interval

        # Max tracking
        self.max_gpu_util = 0
        self.max_mem_util = 0

        # Average tracking
        self.total_gpu_util = 0
        self.total_mem_util = 0
        self.samples = 0

        # Thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._track)
        pynvml.nvmlInit()

    def __enter__(self) -> "GPUUtilTracker":
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        self._thread.join()
        pynvml.nvmlShutdown()

    def _track(self) -> None:
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu = util.gpu
                mem = util.memory

                self.max_gpu_util = max(self.max_gpu_util, gpu)
                self.max_mem_util = max(self.max_mem_util, mem)

                self.total_gpu_util += gpu
                self.total_mem_util += mem
                self.samples += 1

            except pynvml.NVMLError:
                pass

            time.sleep(self.interval)

    def result(self) -> tuple[int, int]:
        avg_gpu_util = self.total_gpu_util / self.samples if self.samples > 0 else 0
        avg_mem_util = self.total_mem_util / self.samples if self.samples > 0 else 0

        return {
            "max_gpu_util": self.max_gpu_util,
            "max_mem_util": self.max_mem_util,
            "avg_gpu_util": int(avg_gpu_util),
            "avg_mem_util": int(avg_mem_util),
        }


def step(
    model: nn.Module,
    batch: Union[torch.Tensor, Dict[str, torch.Tensor], List[str]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: int,
) -> None:
    if isinstance(batch, torch.Tensor):
        x = batch.to(device)
        inputs = {
            "x": x,
            "target_aspect_ratio": 0.75,
            "target_scale": 0.15,
            "context_aspect_ratio": 1.0,
            "context_scale": 0.85,
        }
    elif isinstance(batch, dict) and "input_ids" in batch:
        token_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        inputs = {"input_ids": token_ids, "attention_mask": attention_mask}
    else:
        token_ids, attention_mask = model.tokenize_batch(batch)
        token_ids, attention_mask = token_ids.to(device), attention_mask.to(device)
        inputs = {"input_ids": token_ids, "attention_mask": attention_mask}

    optimizer.zero_grad()
    with autocast(enabled=bool(precision)):
        outputs = model.forward(**inputs)

    if isinstance(outputs, tuple):
        y_student, y_teacher = outputs
        loss = model.criterion(y_student, y_teacher)
    else:
        loss = outputs["loss"]

    loss.backward()
    optimizer.step()


def measure_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    batch_size: int,
    device: torch.device,
    precision: int = 16,
    steps: int = 50,
) -> float:
    """
    Measures iterations per second for a given DataLoader config.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_time: float = time.time()

    for idx, batch in enumerate(dataloader):
        step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            precision=precision,
        )

        if idx >= steps:
            break

    elapsed: float = time.time() - start_time
    # Compute iterations per second
    ips: float = batch_size * steps / elapsed

    return ips
