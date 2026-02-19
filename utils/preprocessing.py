import os
import subprocess
from typing import List

import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from utils.logger import logger


def extract_frames(video_path: str, target_fps: float = 1.0) -> List[Image.Image]:
    """Sample frames from a video at *target_fps* frames per second.

    Returns a list of PIL RGB Images.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Resolve local or system ffmpeg path just in case we need it, though cv2 handles reading.
    # But extract_audio below definitely needs it.


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
    print(f"  Extracting frames from {os.path.basename(video_path)}...", end="", flush=True)

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
    cap.release()
    logger.info("Extracted %d frames from %s", len(frames), video_path)
    print(f" done ({len(frames)} frames).")
    return frames


def detect_and_crop_faces(frames: List[Image.Image]) -> List[Image.Image]:
    """Detect faces in frames using MTCNN and return cropped face images.

    If no face is detected in a frame, the entire frame is used (fallback).
    """
    if not frames:
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Initializing MTCNN for face detection on %s", device)
    print(f"  Running Face Detection on {len(frames)} frames...", end="", flush=True)
    
    try:
        mtcnn = MTCNN(keep_all=True, device=device, post_process=False).eval()
        faces_found = 0
        cropped_images = []

        # Process in batches to avoid OOM if many frames
        batch_size = 32
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            
            # MTCNN can take a list of PIL images
            # boxes_list: list of (N_faces, 4) arrays or None
            try:
                boxes_list, _ = mtcnn.detect(batch)
            except Exception as e:
                logger.warning("MTCNN batch detection failed: %s. Using original frames.", e)
                cropped_images.extend(batch)
                continue

            for frame, boxes in zip(batch, boxes_list):
                if boxes is not None and len(boxes) > 0:
                    # Crop the largest face (or all? Implementation Plan said focus on faces)
                    # Let's crop the largest face to be safe (most prominent subject)
                    # Box format: [x1, y1, x2, y2]
                    
                    # Find largest box area
                    best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                    x1, y1, x2, y2 = map(int, best_box)
                    
                    # Add margin? (Optional, maybe 10%)
                    w, h = frame.size
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        face_crop = frame.crop((x1, y1, x2, y2))
                        cropped_images.append(face_crop)
                        faces_found += 1
                    else:
                        cropped_images.append(frame)
                else:
                    # No face detected, keep original frame context
                    cropped_images.append(frame)

        logger.info("Face detection complete: found faces in %d/%d frames", faces_found, len(frames))
        print(f" done ({faces_found} faces found).")
        return cropped_images

    except Exception as e:
        logger.error("Face detection initialization or runtime failed: %s", e)
        return frames  # Fallback to full frames


        return frames  # Fallback to full frames


def _find_ffmpeg() -> str:
    """Find ffmpeg executable in local directory or system path."""
    # check local project bin
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_bin = os.path.join(project_root, "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.isfile(local_bin):
        return local_bin
    
    local_root_exe = os.path.join(project_root, "ffmpeg", "ffmpeg.exe")
    if os.path.isfile(local_root_exe):
        return local_root_exe

    # Fallback to system path
    return "ffmpeg"


def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio track from a video file to 16 kHz mono WAV using ffmpeg.

    Returns the path to the output WAV file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.wav")

    ffmpeg_exe = _find_ffmpeg()
    
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]

    logger.info("Extracting audio with %s: %s → %s", ffmpeg_exe, video_path, output_path)
    try:
        # Use shell=False to avoid Windows vs *nix issues, but on Windows check=True is enough usually
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg not found at '{ffmpeg_exe}'. Install ffmpeg and ensure it is on your PATH or in ffmpeg/bin"
        )

    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg failed (code %d): %s", exc.returncode, exc.stderr)
        raise RuntimeError(f"Audio extraction failed: {exc.stderr.strip()}")

    logger.info("Audio saved to %s", output_path)
    return output_path
