"""truth.x — FastAPI server wrapping the detection pipeline.

Start with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from utils.logger import logger
from utils.postprocessing import aggregate_results

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")

_models: Dict[str, Any] = {}


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load all models once at startup so they are reused across requests."""
    load_dotenv()
    cfg = _load_config()
    logger.info("API starting — loading models (device=%s)", cfg.get("device", "N/A"))

    from models.video.deepfake_detector import VideoDeepfakeDetector
    from models.audio.synthetic_voice_detector import SyntheticVoiceDetector
    from models.text.ai_text_detector import TextAIDetector
    from services.faiss_service import FAISSSearch

    _models["config"] = cfg
    _models["video"] = VideoDeepfakeDetector()
    _models["audio"] = SyntheticVoiceDetector()
    _models["text"] = TextAIDetector()
    _models["faiss"] = FAISSSearch()

    logger.info("All models loaded — API is ready")
    yield
    _models.clear()
    logger.info("API shutdown — models released")


app = FastAPI(
    title="truth.x",
    description="Deepfake & Misinformation Detection API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/analyze")
async def analyze(
    video: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Analyze a video file for deepfakes and/or a text query for AI-generated
    content and related fact-check articles."""

    if video is None and query is None:
        raise HTTPException(status_code=400, detail="Provide at least a video file or a text query.")

    cfg = _models["config"]
    video_result = None
    audio_result = None
    text_result = None
    articles = None

    # --- Video & Audio ---
    if video is not None:
        suffix = os.path.splitext(video.filename or ".mp4")[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=cfg.get("temp_dir", "data/processed"))
        try:
            tmp.write(await video.read())
            tmp.close()
            video_path = tmp.name

            from utils.preprocessing import extract_frames, extract_audio

            frame_rate = cfg["video"].get("frame_sample_rate", 1)
            logger.info("Extracting frames from upload '%s' at %s fps", video.filename, frame_rate)
            frames = extract_frames(video_path, target_fps=frame_rate)

            t0 = time.perf_counter()
            video_result = _models["video"].predict(frames)
            logger.info("Video inference completed in %.2fs", time.perf_counter() - t0)

            try:
                temp_dir = cfg.get("temp_dir", "data/processed")
                audio_path = extract_audio(video_path, temp_dir)

                t0 = time.perf_counter()
                audio_result = _models["audio"].predict(audio_path)
                logger.info("Audio inference completed in %.2fs", time.perf_counter() - t0)
            except Exception:
                logger.exception("Audio detection failed")

        except Exception as exc:
            logger.exception("Video detection failed")
            raise HTTPException(status_code=500, detail=f"Video analysis error: {exc}") from exc
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    # --- Text & FAISS ---
    if query is not None:
        try:
            t0 = time.perf_counter()
            text_result = _models["text"].predict(query)
            logger.info("Text inference completed in %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("Text detection failed")

        try:
            t0 = time.perf_counter()
            articles = _models["faiss"].search(query)
            logger.info("Article search completed in %.2fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("Article search failed")

    report = aggregate_results(
        video_result=video_result,
        audio_result=audio_result,
        text_result=text_result,
        articles=articles,
    )
    logger.info("Request completed")
    return report


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
