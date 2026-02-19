from __future__ import annotations

import os
from typing import Any, Dict

import torch
import librosa
import numpy as np
import yaml
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from utils.logger import logger

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "config.yaml",
)


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SyntheticVoiceDetector:
    """Detects AI-generated / synthetic speech using a Wav2Vec2-based model.
    """

    def __init__(self) -> None:
        cfg = _load_config()
        self.model_name: str = cfg["audio"].get("model_name", "MelodyMachine/Deepfake-audio-detection-V2")
        self.device = torch.device(
            cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        
        logger.info("Loading audio model '%s' on %s", self.model_name, self.device)
        
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Audio model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load audio model: %s", e)
            self.model = None

    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Analyze an audio file and return fake/real probability.
        
        Returns dict with:
            - label: "fake" or "real"
            - confidence: float (0.0 - 1.0)
            - fake_probability: float
            - real_probability: float
        """
        if self.model is None:
            return {"error": "Model not loaded"}
            
        if not os.path.exists(audio_path):
            logger.error("Audio file not found: %s", audio_path)
            return {"error": "File not found"}

        try:
            # Load audio using librosa (16kHz is standard for Wav2Vec2)
            speech, sr = librosa.load(audio_path, sr=16000)
            
            # Helper to chunk audio if it's too long (Wav2Vec2 can be memory intensive)
            # For now, let's take the first 10-20 seconds or just process assuming it fits in VRAM
            # Better approach: Crop to max 30s.
            MAX_DURATION_S = 30
            if len(speech) > MAX_DURATION_S * sr:
                speech = speech[:MAX_DURATION_S * sr]
            
            inputs = self.feature_extractor(
                speech, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()[0]
            
            # Model specific label mapping needs verification.
            # Usually: 0 -> Fake/Spoof, 1 -> Real/Bona fide OR vice versa.
            # MelodyMachine/Deepfake-audio-detection-V2:
            # config.id2label might be {0: 'fake', 1: 'real'} or similar.
            id2label = self.model.config.id2label
            
            # Let's inspect id2label dynamically
            fake_prob = 0.0
            real_prob = 0.0
            
            if id2label:
                for idx, label in id2label.items():
                    label_str = str(label).lower()
                    if "fake" in label_str or "spoof" in label_str:
                        fake_prob = probs[int(idx)]
                    elif "real" in label_str or "bonafide" in label_str:
                        real_prob = probs[int(idx)]
            
            # Fallback if labels are opaque (e.g. LABEL_0)
            # Testing required to confirm. For many anti-spoofing models, 0=spoof, 1=bonafide.
            # But let's assume if we couldn't parse, we take 0 as fake (safe assumption to verify).
            if fake_prob == 0.0 and real_prob == 0.0:
                 fake_prob = probs[0]
                 real_prob = probs[1] if len(probs) > 1 else 1.0 - fake_prob

            if fake_prob > real_prob:
                label = "fake"
                confidence = fake_prob
            else:
                label = "real"
                confidence = real_prob

            logger.info("Audio result: label=%s, conf=%.4f (fake=%.4f, real=%.4f)", 
                        label, confidence, fake_prob, real_prob)

            return {
                "label": label,
                "confidence": round(confidence, 4),
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4)
            }

        except Exception as e:
            logger.error("Audio detection error: %s", e)
            return {"error": str(e)}
