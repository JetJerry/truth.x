from typing import Any, Dict, List, Optional

from utils.logger import logger


def aggregate_results(
    video_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None,
    text_result: Optional[Dict[str, Any]] = None,
    articles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Combine individual detector outputs into a single report.

    Each *_result dict is expected to carry at least:
        - "label"      : str   (e.g. "fake" / "real")
        - "confidence" : float (0.0 – 1.0)

    The combined confidence is the weighted average of whichever
    detectors actually returned a result.
    """
    weights = {"video": 0.45, "audio": 0.30, "text": 0.25}

    report: Dict[str, Any] = {
        "video": video_result,
        "audio": audio_result,
        "text": text_result,
        "related_articles": articles or [],
        "combined_confidence": None,
        "overall_label": None,
    }

    weighted_sum = 0.0
    total_weight = 0.0

    for key, result in [("video", video_result), ("audio", audio_result), ("text", text_result)]:
        if result is None:
            continue
        confidence = result.get("confidence")
        if confidence is None:
            logger.warning("Missing confidence in %s result, skipping from aggregate", key)
            continue
        w = weights[key]
        weighted_sum += confidence * w
        total_weight += w

    if total_weight > 0:
        combined = round(weighted_sum / total_weight, 4)
        report["combined_confidence"] = combined
        report["overall_label"] = "fake" if combined >= 0.5 else "real"

    logger.info("Aggregated results: %s", report)
    return report
