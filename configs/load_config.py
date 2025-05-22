import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "./config.json") -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    return config
