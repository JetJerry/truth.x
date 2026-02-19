"""truth.x  -  Deepfake & Misinformation Detection Pipeline

Usage:
    python main_pipeline.py --video path/to/video.mp4
    python main_pipeline.py --query "Some suspicious claim"
    python main_pipeline.py --text-file path/to/document.pdf
    python main_pipeline.py --video clip.mp4 --query "Is this real?"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import yaml
from dotenv import load_dotenv

from utils.logger import logger

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="truth.x  -  Detection Pipeline")
    p.add_argument("--video", type=str, default=None, help="Video file for deepfake analysis")
    p.add_argument("--query", type=str, default=None, help="Text for AI-text detection + article search")
    p.add_argument("--text-file", type=str, default=None, help="Document (.pdf .docx .txt) for AI-text detection")
    return p.parse_args()


# ------------------------------------------------------------------
# Clean terminal output
# ------------------------------------------------------------------

def _print_report(report: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  truth.x  -  Detection Report")
    print(sep)

    v = report.get("video")
    if v and v.get("label") != "unknown":
        avg = v.get("average", {})
        avg_str = ", ".join(f"{k}: {val:.2%}" for k, val in avg.items())
        print(f"\n  [Video]  {v['label'].upper()}  (confidence: {v['confidence']:.2%})")
        print(f"           Avg scores  : {avg_str}")
        print(f"           Frames used : {len(v.get('per_frame', []))}")

    a = report.get("audio")
    if a and a.get("error") is None:
        print(f"\n  [Audio]  {a.get('label', 'N/A').upper()}  (confidence: {a.get('confidence', 0):.2%})")
    elif a and a.get("error"):
        print(f"\n  [Audio]  Stub  -  {a['error']}")

    t = report.get("text")
    if t and t.get("label") != "unknown":
        print(f"\n  [Text]   {t['label'].upper()}  (confidence: {t['confidence']:.2%})")
        print(f"           AI probability    : {t.get('ai_probability', 0):.2%}")
        print(f"           Human probability : {t.get('human_probability', 0):.2%}")

    arts = report.get("related_articles", [])
    if arts:
        print(f"\n  [Related Articles]  ({len(arts)} found)")
        for i, art in enumerate(arts, 1):
            score = art.get("similarity_score", 0)
            title = art.get("title", "Untitled")
            print(f"    {i}. [{score:.2%}] {title}")

    overall = report.get("overall_label")
    fake_prob = report.get("combined_fake_probability")
    if overall:
        print(f"\n{'-' * 60}")
        print(f"  OVERALL          : {overall.upper()}")
        if fake_prob is not None:
            print(f"  Fake probability : {fake_prob:.2%}")
    print(f"{sep}\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = parse_args()

    text_file = getattr(args, "text_file", None)
    if text_file:
        from utils.document_reader import read_document
        logger.info("Reading document: %s", text_file)
        file_text = read_document(text_file)
        if not file_text.strip():
            print("ERROR: Document is empty.")
            sys.exit(1)
        logger.info("Extracted %d characters from document", len(file_text))
        args.query = file_text

    if args.video is None and args.query is None:
        print("ERROR: Provide --video, --query, or --text-file.")
        sys.exit(1)

    cfg = load_config()
    logger.info("Configuration loaded (device=%s)", cfg.get("device", "N/A"))

    print("Starting pipeline...", end="", flush=True)

    # We will load models Just-In-Time to save memory/swap
    video_result = None
    audio_result = None
    text_result = None
    articles = None

    import gc

    # ------------------------------------------------------------------
    # Video & Audio Analysis
    # ------------------------------------------------------------------
    if args.video:
        print(f"\n[1/3] Video Analysis: {args.video}")
        
        # 1. Video Visual Analysis
        try:
            print("  Loading Video Model...", end="", flush=True)
            from models.video.deepfake_detector import VideoDeepfakeDetector
            video_detector = VideoDeepfakeDetector()
            print(" done.")

            from utils.preprocessing import extract_frames
            frame_rate = cfg["video"].get("frame_sample_rate", 1)
            frames = extract_frames(args.video, target_fps=frame_rate)

            t0 = time.perf_counter()
            video_result = video_detector.predict(frames)
            logger.info("Video inference completed in %.2fs", time.perf_counter() - t0)
            
            # Explicit cleanup
            print("  Unloading Video Model...")
            del video_detector
            del frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except ImportError as e:
            logger.error("Failed to import preprocessing utils or model: %s", e)
        except Exception as e:
            logger.exception("Video detection failed")
            print(f"Video detection failed: {e}")

        # 2. Audio Analysis
        print("  Loading Audio Model...", end="", flush=True)
        try:
            from models.audio.synthetic_voice_detector import SyntheticVoiceDetector
            audio_detector = SyntheticVoiceDetector()
            print(" done.")
            
            print("  Starting audio analysis...")
            from utils.preprocessing import extract_audio
            temp_dir = cfg.get("temp_dir", "data/processed")
            audio_path = extract_audio(args.video, temp_dir)

            t0 = time.perf_counter()
            audio_result = audio_detector.predict(audio_path)
            logger.info("Audio inference completed in %.2fs", time.perf_counter() - t0)
            print("  Audio analysis complete.")
            
            # Cleanup audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
            # Explicit cleanup
            print("  Unloading Audio Model...")
            del audio_detector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            # If audio model fails to load or predict
            logger.exception("Audio detection failed")
            print(f"Audio detection failed: {e}")


    # ------------------------------------------------------------------
    # Text Analysis
    # ------------------------------------------------------------------
    if args.query:
        print(f"\n[2/3] Text Analysis: \"{args.query[:50]}...\"")
        
        # 3. Text Detection
        try:
            print("  Loading Text Model...", end="", flush=True)
            from models.text.ai_text_detector import TextAIDetector
            text_detector = TextAIDetector()
            print(" done.")

            t0 = time.perf_counter()
            text_result = text_detector.predict(args.query)
            logger.info("Text inference completed in %.2fs", time.perf_counter() - t0)
            
            # Explicit cleanup
            print("  Unloading Text Model...")
            del text_detector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.exception("Text detection failed")
            print(f"Text detection failed: {e}")

        # 4. Related Articles (FAISS)
        try:
            print("\n[3/3] Article Search...")
            from services.faiss_service import FAISSSearch
            faiss_search = FAISSSearch()
            
            t0 = time.perf_counter()
            articles = faiss_search.search(args.query)
            logger.info("Article search completed in %.2fs", time.perf_counter() - t0)
            
            # Explicit cleanup
            del faiss_search
            gc.collect()

        except Exception as e:
            logger.exception("Article search failed")
            print(f"Article search failed: {e}")


    # ------------------------------------------------------------------
    # Aggregate and output
    # ------------------------------------------------------------------
    from utils.postprocessing import aggregate_results

    print("\nAggregating results...")
    report = aggregate_results(
        video_result=video_result,
        audio_result=audio_result,
        text_result=text_result,
        articles=articles,
    )

    _print_report(report)
    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
