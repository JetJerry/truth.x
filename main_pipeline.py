"""truth.x — CLI pipeline for deepfake and misinformation detection.

Usage examples:
    python main_pipeline.py --video path/to/video.mp4
    python main_pipeline.py --query "Some suspicious claim to fact-check"
    python main_pipeline.py --video path/to/video.mp4 --query "Claim in the video"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import yaml
from dotenv import load_dotenv

from utils.logger import logger

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="truth.x — Deepfake & Misinformation Detection Pipeline",
    )
    parser.add_argument("--video", type=str, default=None, help="Path to a video file for deepfake analysis")
    parser.add_argument("--query", type=str, default=None, help="Text query for AI-text detection and article search")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.video is None and args.query is None:
        logger.error("No input provided. Use --video and/or --query.")
        sys.exit(1)

    cfg = load_config()
    logger.info("Configuration loaded (device=%s)", cfg.get("device", "N/A"))

    # ------------------------------------------------------------------
    # Lazy-initialise only the models we actually need
    # ------------------------------------------------------------------
    video_detector = None
    audio_detector = None
    text_detector = None
    faiss_search = None

    if args.video:
        logger.info("Initialising video deepfake detector …")
        from models.video.deepfake_detector import VideoDeepfakeDetector
        video_detector = VideoDeepfakeDetector()

        logger.info("Initialising audio synthetic-voice detector …")
        from models.audio.synthetic_voice_detector import SyntheticVoiceDetector
        audio_detector = SyntheticVoiceDetector()

    if args.query:
        logger.info("Initialising AI-text detector …")
        from models.text.ai_text_detector import TextAIDetector
        text_detector = TextAIDetector()

        logger.info("Initialising FAISS article search …")
        from services.faiss_service import FAISSSearch
        faiss_search = FAISSSearch()

    # ------------------------------------------------------------------
    # Run detectors
    # ------------------------------------------------------------------
    video_result = None
    audio_result = None
    text_result = None
    articles = None

    # --- Video ---
    if args.video and video_detector is not None:
        try:
            from utils.preprocessing import extract_frames, extract_audio

            frame_rate = cfg["video"].get("frame_sample_rate", 1)
            logger.info("Extracting frames from '%s' at %s fps", args.video, frame_rate)
            frames = extract_frames(args.video, target_fps=frame_rate)

            t0 = time.perf_counter()
            video_result = video_detector.predict(frames)
            logger.info("Video inference completed in %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("Video detection failed")

        # --- Audio (stub) ---
        if audio_detector is not None:
            try:
                temp_dir = cfg.get("temp_dir", "data/processed")
                audio_path = extract_audio(args.video, temp_dir)

                t0 = time.perf_counter()
                audio_result = audio_detector.predict(audio_path)
                logger.info("Audio inference completed in %.2fs", time.perf_counter() - t0)
            except Exception:
                logger.exception("Audio detection failed")

    # --- Text ---
    if args.query and text_detector is not None:
        try:
            t0 = time.perf_counter()
            text_result = text_detector.predict(args.query)
            logger.info("Text inference completed in %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("Text detection failed")

    # --- FAISS article search ---
    if args.query and faiss_search is not None:
        try:
            t0 = time.perf_counter()
            articles = faiss_search.search(args.query)
            logger.info("Article search completed in %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("Article search failed")

    # ------------------------------------------------------------------
    # Aggregate and output
    # ------------------------------------------------------------------
    from utils.postprocessing import aggregate_results

    report = aggregate_results(
        video_result=video_result,
        audio_result=audio_result,
        text_result=text_result,
        articles=articles,
    )

    output = json.dumps(report, indent=2, default=str)
    print("\n" + "=" * 60)
    print("  truth.x — Detection Report")
    print("=" * 60)
    print(output)
    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
