from functools import lru_cache
from typing import Any, Dict

from .load_config import load_config


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    return load_config()


####################################################################################################

# IMAGE

####################################################################################################


@lru_cache(maxsize=1)
def get_image_config() -> Dict[str, Any]:
    return get_config()["image"]


@lru_cache(maxsize=1)
def get_image_model_config() -> Dict[str, Any]:
    return get_image_config()["model"]


@lru_cache(maxsize=1)
def get_image_dataset_config() -> Dict[str, Any]:
    return get_image_config()["dataset"]


@lru_cache(maxsize=1)
def get_image_profiling_config() -> Dict[str, Any]:
    return get_image_config()["profiling"]


@lru_cache(maxsize=1)
def get_image_experiment_config() -> Dict[str, Any]:
    return get_image_config()["experiment"]


@lru_cache(maxsize=1)
def get_image_runtime_config() -> Dict[str, Any]:
    return get_image_config()["runtime"]


@lru_cache(maxsize=1)
def get_image_tracking_config() -> Dict[str, Any]:
    return get_image_config()["tracking"]


@lru_cache(maxsize=1)
def get_image_dataset_transforms_config() -> Dict[str, Any]:
    return get_image_dataset_config()["transforms"]


####################################################################################################

# VIDEO

####################################################################################################


@lru_cache(maxsize=1)
def get_video_config() -> Dict[str, Any]:
    return get_config()["video"]


@lru_cache(maxsize=1)
def get_video_model_config() -> Dict[str, Any]:
    return get_video_config()["model"]


@lru_cache(maxsize=1)
def get_video_dataset_config() -> Dict[str, Any]:
    return get_video_config()["dataset"]


@lru_cache(maxsize=1)
def get_video_profiling_config() -> Dict[str, Any]:
    return get_video_config()["profiling"]


@lru_cache(maxsize=1)
def get_video_experiment_config() -> Dict[str, Any]:
    return get_video_config()["experiment"]


@lru_cache(maxsize=1)
def get_video_runtime_config() -> Dict[str, Any]:
    return get_video_config()["runtime"]


@lru_cache(maxsize=1)
def get_video_tracking_config() -> Dict[str, Any]:
    return get_video_config()["tracking"]


@lru_cache(maxsize=1)
def get_video_dataset_transforms_config() -> Dict[str, Any]:
    return get_video_dataset_config()["transforms"]


####################################################################################################

# TEXT

####################################################################################################


@lru_cache(maxsize=1)
def get_text_config() -> Dict[str, Any]:
    return get_config()["text"]


@lru_cache(maxsize=1)
def get_text_model_config() -> Dict[str, Any]:
    return get_text_config()["model"]


@lru_cache(maxsize=1)
def get_text_dataset_config() -> Dict[str, Any]:
    return get_text_config()["dataset"]


@lru_cache(maxsize=1)
def get_text_profiling_config() -> Dict[str, Any]:
    return get_text_config()["profiling"]


@lru_cache(maxsize=1)
def get_text_experiment_config() -> Dict[str, Any]:
    return get_text_config()["experiment"]


@lru_cache(maxsize=1)
def get_text_runtime_config() -> Dict[str, Any]:
    return get_text_config()["runtime"]


@lru_cache(maxsize=1)
def get_text_tracking_config() -> Dict[str, Any]:
    return get_text_config()["tracking"]


@lru_cache(maxsize=1)
def get_text_dataset_tokenization_config() -> Dict[str, Any]:
    return get_text_dataset_config()["tokenization"]
