import json
from typing import Any, Dict

with open("../config.json", encoding="utf-8") as f:
    config: Dict[str, Any] = json.load(f)

####################################################################################################

# TEXT

####################################################################################################

text_config: Dict[str, Any] = config["text"]
text_model_config: Dict[str, Any] = text_config["model"]
text_dataset_config: Dict[str, Any] = text_config["dataset"]
text_profiling_config: Dict[str, Any] = text_config["profiling"]
text_experiment_config: Dict[str, Any] = text_config["experiment"]
text_runtime_config: Dict[str, Any] = text_config["runtime"]
text_tracking_config: Dict[str, Any] = text_config["tracking"]
text_dataset_tokenization_config: Dict[str, Any] = text_dataset_config["tokenization"]

####################################################################################################

# IMAGE

####################################################################################################

image_config: Dict[str, Any] = config["image"]
image_model_config: Dict[str, Any] = image_config["model"]
image_dataset_config: Dict[str, Any] = image_config["dataset"]
image_profiling_config: Dict[str, Any] = image_config["profiling"]
image_experiment_config: Dict[str, Any] = image_config["experiment"]
image_runtime_config: Dict[str, Any] = image_config["runtime"]
image_tracking_config: Dict[str, Any] = image_config["tracking"]
image_dataset_transforms_config: Dict[str, Any] = image_dataset_config["transforms"]

####################################################################################################

# VIDEO

####################################################################################################

video_config: Dict[str, Any] = config["video"]
video_model_config: Dict[str, Any] = video_config["model"]
video_dataset_config: Dict[str, Any] = video_config["dataset"]
video_profiling_config: Dict[str, Any] = video_config["profiling"]
video_experiment_config: Dict[str, Any] = video_config["experiment"]
video_runtime_config: Dict[str, Any] = video_config["runtime"]
video_tracking_config: Dict[str, Any] = video_config["tracking"]
video_dataset_transforms_config: Dict[str, Any] = video_dataset_config["transforms"]
