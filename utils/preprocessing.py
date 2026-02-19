import os
import subprocess
from typing import List

import cv2
from PIL import Image

from utils.logger import logger


def extract_frames(video_path: str, target_fps: float = 1.0) -> List[Image.Image]:
    """Sample frames from a video at *target_fps* frames per second.

    Returns a list of PIL RGB Images.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine FPS for: {video_path}")

    frame_interval = max(1, int(round(native_fps / target_fps)))
    logger.info(
        "Extracting frames from %s (fps=%.2f, total=%d, interval=%d)",
        video_path, native_fps, total_frames, frame_interval,
    )

    frames: List[Image.Image] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        frame_idx += 1

    cap.release()
    logger.info("Extracted %d frames from %s", len(frames), video_path)
    return frames


def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio track from a video file to 16 kHz mono WAV using ffmpeg.

    Returns the path to the output WAV file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]

    logger.info("Extracting audio: %s → %s", video_path, output_path)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it is on your PATH."
        )
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg failed (code %d): %s", exc.returncode, exc.stderr)
        raise RuntimeError(f"Audio extraction failed: {exc.stderr.strip()}")

    logger.info("Audio saved to %s", output_path)
    return output_path
