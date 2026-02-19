from typing import Any, Dict, List, Optional

from utils.logger import logger

FAKE_LABELS = {"fake", "deepfake", "ai-generated", "ai", "machine"}
REAL_LABELS = {"real", "human-written", "human"}


def _to_fake_probability(result: Dict[str, Any]) -> Optional[float]:
    """Convert a detector result to a probability-of-fake score (0.0 = real, 1.0 = fake).

    If the detector predicted "real"/"human-written", the fake probability
    is (1 - confidence).  If it predicted "fake"/"ai-generated", the fake
    probability equals its confidence directly.
    """
    label = str(result.get("label", "")).lower()
    confidence = result.get("confidence")
    if confidence is None:
        return None

    if label in FAKE_LABELS:
        return float(confidence)
    elif label in REAL_LABELS:
        return 1.0 - float(confidence)

    return None


def aggregate_results(
    video_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None,
    text_result: Optional[Dict[str, Any]] = None,
    articles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Combine individual detector outputs into a single report.

    Each *_result dict is expected to carry at least:
        - "label"      : str   (e.g. "fake" / "real" / "ai-generated" / "human-written")
        - "confidence" : float (0.0 – 1.0, probability of the predicted label)

    The combined score is a weighted average of each detector's
    *fake probability* (not raw confidence), so a "real" prediction
    correctly pushes the score toward 0.
    """
    weights = {"video": 0.45, "audio": 0.30, "text": 0.25}

    report: Dict[str, Any] = {
        "video": video_result,
        "audio": audio_result,
        "text": text_result,
        "related_articles": articles or [],
        "combined_fake_probability": None,
        "combined_confidence": None,
        "overall_label": None,
    }

    weighted_sum = 0.0
    total_weight = 0.0

    for key, result in [("video", video_result), ("audio", audio_result), ("text", text_result)]:
        if result is None:
            continue

        fake_prob = _to_fake_probability(result)
        if fake_prob is None:
            logger.warning("Cannot derive fake probability from %s result (label=%s), skipping",
                           key, result.get("label"))
            continue

        logger.debug("%s detector: label=%s, confidence=%s → fake_probability=%.4f",
                     key, result.get("label"), result.get("confidence"), fake_prob)

        w = weights[key]
        weighted_sum += fake_prob * w
        total_weight += w

    if total_weight > 0:
        combined_fake = round(weighted_sum / total_weight, 4)
        report["combined_fake_probability"] = combined_fake
        report["combined_confidence"] = round(abs(combined_fake - 0.5) * 2, 4)
        report["overall_label"] = "fake" if combined_fake >= 0.5 else "real"

    logger.info("Aggregated: overall_label=%s, fake_probability=%.4f",
                report["overall_label"], report.get("combined_fake_probability", 0))
    return report
