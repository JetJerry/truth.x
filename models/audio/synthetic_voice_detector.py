from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from utils.logger import logger

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "config.yaml",
)


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SyntheticVoiceDetector:
    """Stub detector for AI-generated / cloned speech.

    Will be extended with a real model in a future step.
    """

    def __init__(self) -> None:
        cfg = _load_config()
        self.model_name: str = cfg["audio"]["model_name"]
        logger.info("SyntheticVoiceDetector initialised (model: %s) [stub]", self.model_name)

    def predict(self, audio_path: str) -> Dict[str, Any]:
        logger.warning("SyntheticVoiceDetector.predict() is a stub — not yet implemented")
        return {"fake_probability": None, "error": "Not implemented"}
