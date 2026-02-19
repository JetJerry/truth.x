from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from utils.logger import logger

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "config.yaml",
)

# Based on prithivMLmods/Deep-Fake-Detector-v2-Model model card/testing:
# usually Label 0: Fake, Label 1: Real OR vice versa.
# We will inspect id2label but also allow manual override if needed.
# For many deepfake models: "FAKE" is often label 0 or 1.
FAKE_KEYWORDS = {"fake", "deepfake", "manipulated", "synthetic", "generated", "0"} # Added '0' as fallback if label is just a number
REAL_KEYWORDS = {"real", "authentic", "original", "genuine", "1", "realism", "clean"} # Added '1' and 'realism' as fallback



def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class VideoDeepfakeDetector:
    """Classifies individual frames as fake or real using a HuggingFace
    image-classification model and returns aggregated scores."""

    def __init__(self) -> None:
        cfg = _load_config()
        video_cfg = cfg["video"]
        self.model_name: str = video_cfg["model_name"]
        self.batch_size: int = video_cfg.get("batch_size", 8)
        self.do_face_detection: bool = video_cfg.get("face_detection", True)
        self.device = torch.device(
            cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )


        logger.info("Loading video model '%s' on %s", self.model_name, self.device)
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        logger.info("Video model id2label: %s", id2label)

        self.fake_label, self.real_label = self._resolve_labels(id2label)
        
        # Explicit override for know models if needed
        if "prithivMLmods" in self.model_name:
             # PrithivML model: Label 0 -> Fake, Label 1 -> Real (Usually, need to verify)
             # Let's rely on _resolve_labels logic which looks at string values if possible.
             # If labels are "LABEL_0", "LABEL_1", we might need logic.
             pass

        logger.info("Video model loaded — fake_label='%s', real_label='%s'",
                     self.fake_label, self.real_label)


    @staticmethod
    def _resolve_labels(id2label: dict) -> tuple[str, str]:
        """Find which label name corresponds to fake and real (case-insensitive)."""
        fake_label = None
        real_label = None

        for _, lbl in id2label.items():
            lower = str(lbl).lower().strip()
            if lower in FAKE_KEYWORDS:
                fake_label = str(lbl)
            elif lower in REAL_KEYWORDS:
                real_label = str(lbl)

        if fake_label and real_label:
            return fake_label, real_label

        labels = [str(v) for v in id2label.values()]
        logger.warning("Could not auto-map labels %s; using first=%s as real, last=%s as fake",
                       id2label, labels[0], labels[-1])
        return labels[-1], labels[0]

    @torch.no_grad()
    def _predict_batch(self, frames: List[Image.Image]) -> List[Dict[str, float]]:
        inputs = self.extractor(images=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        id2label = self.model.config.id2label
        results = []
        for prob_row in probs:
            results.append({id2label[i]: round(p, 4) for i, p in enumerate(prob_row)})
        return results

    def predict(self, frames: List[Image.Image]) -> Dict[str, Any]:
        """Run inference on a list of PIL Images.

        Returns a dict with:
            - per_frame  : list of per-frame label probability dicts
            - average    : averaged probabilities across all frames
            - label      : "fake" or "real"
            - confidence : probability of the predicted label
        """
        if not frames:
            logger.warning("No frames provided for video prediction")
            return {"per_frame": [], "average": {}, "label": "unknown", "confidence": 0.0}

        # Face Detection Step
        if self.do_face_detection:
            from utils.preprocessing import detect_and_crop_faces
            logger.info("Running face detection on %d frames...", len(frames))
            frames = detect_and_crop_faces(frames)
            logger.info("Finished face detection/cropping.")

        logger.info("Running video inference on %d frames (batch_size=%d)", len(frames), self.batch_size)


        per_frame: List[Dict[str, float]] = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            logger.debug("Processing frame batch %d–%d", start, start + len(batch) - 1)
            per_frame.extend(self._predict_batch(batch))

        label_keys = per_frame[0].keys()
        average = {k: round(sum(f[k] for f in per_frame) / len(per_frame), 4) for k in label_keys}

        fake_prob = average.get(self.fake_label, 0.0)
        real_prob = average.get(self.real_label, 0.0)

        if fake_prob >= real_prob:
            canonical_label = "fake"
            confidence = fake_prob
        else:
            canonical_label = "real"
            confidence = real_prob

        result: Dict[str, Any] = {
            "per_frame": per_frame,
            "average": average,
            "label": canonical_label,
            "confidence": confidence,
        }

        logger.info("Video result: label=%s, confidence=%.4f (fake=%.4f, real=%.4f)",
                     canonical_label, confidence, fake_prob, real_prob)
        return result
