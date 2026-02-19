<p align="center">
  <strong>truth.x</strong>
</p>
<p align="center">
  <em>Modular, GPU-accelerated AI pipeline for deepfake detection, AI-generated text classification, and fact-check article retrieval.</em>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA%2FCPU-ee4c2c?logo=pytorch&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [High-Level Pipeline Flow](#high-level-pipeline-flow)
  - [Module Dependency Graph](#module-dependency-graph)
- [Project Structure](#project-structure)
- [Core Modules — Detailed Logic](#core-modules--detailed-logic)
  - [1. Video Deepfake Detector](#1-video-deepfake-detector)
  - [2. Synthetic Voice Detector (Stub)](#2-synthetic-voice-detector-stub)
  - [3. AI-Generated Text Detector](#3-ai-generated-text-detector)
  - [4. FAISS Semantic Search Service](#4-faiss-semantic-search-service)
  - [5. Preprocessing Utilities](#5-preprocessing-utilities)
  - [6. Postprocessing — Weighted Aggregation](#6-postprocessing--weighted-aggregation)
  - [7. Document Reader](#7-document-reader)
  - [8. Logger](#8-logger)
- [Integrations & External Dependencies](#integrations--external-dependencies)
- [Configuration Reference](#configuration-reference)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [CLI Pipeline](#cli-pipeline)
  - [FastAPI Server](#fastapi-server)
  - [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Diagnostics](#diagnostics)
- [Future Improvements](#future-improvements)

---

## Overview

**truth.x** is a multi-modal AI detection system that analyzes video, audio, and text content to identify deepfakes, AI-generated text, and related misinformation. It aggregates results from multiple detectors into a single confidence-scored verdict.

The system provides two interfaces:

| Interface | Entry Point | Description |
|---|---|---|
| **CLI** | `main_pipeline.py` | Command-line tool for direct analysis |
| **REST API** | `api.py` | FastAPI server for integration with web applications |

### Key Capabilities

- **Video deepfake detection** — frame-level classification using a HuggingFace image-classification model with batched GPU inference
- **Synthetic voice detection** — audio extraction + placeholder for a future voice authenticity model
- **AI text detection** — RoBERTa-based sequence classification distinguishing human-written from AI-generated text
- **Fact-check retrieval** — semantic similarity search over a local article corpus using Sentence-Transformers embeddings and numpy cosine similarity
- **Document ingestion** — extract text from `.pdf`, `.docx`, and `.txt` files for AI-text detection
- **Weighted confidence aggregation** — normalised fake-probability scores combined via configurable detector weights

---

## Architecture

### High-Level Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│         --video <file>  │  --query <text>  │  --text-file <doc>  │
└────────────┬────────────┴────────┬─────────┴────────┬────────────┘
             │                     │                  │
             ▼                     │                  ▼
  ┌─────────────────────┐         │      ┌─────────────────────┐
  │   PREPROCESSING     │         │      │   DOCUMENT READER   │
  │  ┌───────────────┐  │         │      │  PyPDF2 / docx / io │
  │  │ extract_frames│  │         │      └──────────┬──────────┘
  │  │   (OpenCV)    │  │         │                 │
  │  ├───────────────┤  │         │         extracted text
  │  │ extract_audio │  │         │                 │
  │  │   (ffmpeg)    │  │         ▼                 ▼
  │  └───────────────┘  │    ┌────────────────────────┐
  └──────┬─────┬────────┘    │       TEXT QUERY        │
         │     │             └───────┬─────────┬───────┘
   frames│     │audio                │         │
         ▼     ▼                     ▼         ▼
  ┌────────┐ ┌────────┐     ┌────────────┐ ┌──────────┐
  │ VIDEO  │ │ AUDIO  │     │   TEXT AI   │ │  FAISS   │
  │DETECTOR│ │DETECTOR│     │  DETECTOR  │ │ SEARCH   │
  │(HF IMG)│ │ (stub) │     │ (RoBERTa)  │ │(SentTF)  │
  └───┬────┘ └───┬────┘     └─────┬──────┘ └────┬─────┘
      │          │                │              │
      ▼          ▼                ▼              ▼
  ┌───────────────────────────────────────────────────┐
  │              POSTPROCESSING                        │
  │       Weighted Fake-Probability Aggregation        │
  │   Video: 0.45  │  Audio: 0.30  │  Text: 0.25     │
  └────────────────────────┬──────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   DETECTION REPORT  │
                │  overall_label      │
                │  combined_fake_prob │
                │  per-detector data  │
                │  related_articles   │
                └─────────────────────┘
```

### Module Dependency Graph

```
main_pipeline.py / api.py
  ├── config/config.yaml          (YAML configuration)
  ├── .env                        (environment variables)
  ├── models/
  │   ├── video/deepfake_detector.py
  │   │     └── transformers (AutoFeatureExtractor, AutoModelForImageClassification)
  │   ├── audio/synthetic_voice_detector.py
  │   │     └── (stub — no external model loaded)
  │   └── text/ai_text_detector.py
  │         └── transformers (AutoTokenizer, AutoModelForSequenceClassification)
  ├── services/
  │   └── faiss_service.py
  │         ├── sentence_transformers (SentenceTransformer)
  │         └── numpy (cosine similarity via matrix multiplication)
  └── utils/
      ├── preprocessing.py        → OpenCV (cv2) + ffmpeg subprocess
      ├── postprocessing.py       → weighted aggregation logic
      ├── document_reader.py      → PyPDF2, python-docx
      └── logger.py               → Python stdlib logging
```

---

## Project Structure

```
truth.x/
├── api.py                     # FastAPI server — REST interface
├── main_pipeline.py           # CLI entry point — argparse-based
├── diagnose.py                # Model debugging — prints raw label mappings
├── test1.py                   # Quick FAISS integration smoke test
├── requirements.txt           # Pinned Python dependencies
├── .env                       # Environment variables (git-ignored)
├── .gitignore
│
├── config/
│   └── config.yaml            # All model names, parameters, paths
│
├── data/
│   ├── articles.json          # Fact-check article corpus (JSON array)
│   ├── embeddings.npy         # Pre-computed article embeddings (auto-generated)
│   ├── faiss_metadata.pkl     # Cached article text hash for rebuild detection
│   ├── samples/               # Place test video files here (git-ignored)
│   └── processed/             # Temporary extracted audio/docs (git-ignored)
│
├── models/
│   ├── __init__.py
│   ├── video/
│   │   ├── __init__.py
│   │   └── deepfake_detector.py    # Frame-level deepfake classification
│   ├── audio/
│   │   ├── __init__.py
│   │   └── synthetic_voice_detector.py  # Stub — not implemented yet
│   └── text/
│       ├── __init__.py
│       └── ai_text_detector.py     # AI vs human text classification
│
├── services/
│   ├── __init__.py
│   └── faiss_service.py       # Semantic article search (embeddings + cosine sim)
│
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Dual-output logger (console + file)
│   ├── preprocessing.py       # Frame extraction (OpenCV) + audio extraction (ffmpeg)
│   ├── postprocessing.py      # Weighted result aggregation
│   └── document_reader.py     # PDF / DOCX / TXT text extraction
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # Pytest suite — direct + API modes
│
├── logs/                      # Auto-created at runtime (git-ignored)
│   └── pipeline.log
│
├── hf_cache/                  # HuggingFace model cache (git-ignored)
└── ffmpeg/                    # Local ffmpeg binaries (git-ignored)
```

---

## Core Modules — Detailed Logic

### 1. Video Deepfake Detector

**File:** `models/video/deepfake_detector.py`
**Class:** `VideoDeepfakeDetector`

#### How It Works

1. **Initialization:**
   - Reads `config.yaml` for the model name (`dima806/deepfake_vs_real_image_detection`), batch size, and device preference.
   - Loads a HuggingFace **`AutoFeatureExtractor`** and **`AutoModelForImageClassification`** from the Hub.
   - Moves the model to GPU (`cuda`) if available, otherwise falls back to CPU.
   - Auto-resolves the model's `id2label` dictionary to identify which output index corresponds to "fake" vs "real" using keyword matching against sets like `{"fake", "deepfake", "manipulated", "synthetic", "generated"}` and `{"real", "authentic", "original", "genuine"}`.

2. **Inference — `predict(frames)`:**
   - Accepts a list of `PIL.Image` objects (extracted from video).
   - Processes frames in configurable batches (`batch_size`, default `8`) for memory efficiency.
   - For each batch: runs the feature extractor to produce tensors, feeds them through the model, and applies `softmax` to get per-class probabilities.
   - Returns per-frame probability dictionaries.

3. **Aggregation:**
   - Averages probabilities across all frames for each label.
   - Compares `avg(fake_prob)` vs `avg(real_prob)` to determine the final `label` and `confidence`.

4. **Output Format:**
   ```json
   {
     "per_frame": [{"Fake": 0.87, "Real": 0.13}, ...],
     "average": {"Fake": 0.82, "Real": 0.18},
     "label": "fake",
     "confidence": 0.82
   }
   ```

---

### 2. Synthetic Voice Detector (Stub)

**File:** `models/audio/synthetic_voice_detector.py`
**Class:** `SyntheticVoiceDetector`

This module is a **placeholder** for future synthetic speech detection. Currently:

- Reads the `audio.model_name` from config (`facebook/wav2vec2-base-960h`) but does **not** load the model.
- The `predict(audio_path)` method returns `{"fake_probability": None, "error": "Not implemented"}`.
- Audio is still extracted from videos via ffmpeg (in `preprocessing.py`), so the pipeline infrastructure is in place for when a real model is added.

---

### 3. AI-Generated Text Detector

**File:** `models/text/ai_text_detector.py`
**Class:** `TextAIDetector`

#### How It Works

1. **Initialization:**
   - Loads a HuggingFace **`AutoTokenizer`** and **`AutoModelForSequenceClassification`** from the Hub.
   - Default model: `roberta-base-openai-detector` (fine-tuned RoBERTa for detecting GPT-generated text).
   - Auto-resolves label indices by matching `id2label` against keyword sets (`{"fake", "ai", "ai-generated", "machine", "generated"}` / `{"real", "human", "human-written"}`).
   - Falls back to `idx 0 = Real, idx N-1 = Fake` if keyword matching fails.

2. **Inference — `predict(text)`:**
   - Tokenizes the input with truncation at `max_length` (default `512` tokens).
   - Runs a single forward pass through the model.
   - Applies `softmax` to extract `ai_probability` and `human_probability`.
   - Returns the dominant label and its confidence.

3. **Output Format:**
   ```json
   {
     "label": "ai-generated",
     "confidence": 0.94,
     "ai_probability": 0.94,
     "human_probability": 0.06
   }
   ```

---

### 4. FAISS Semantic Search Service

**File:** `services/faiss_service.py`
**Class:** `FAISSSearch`

> **Note:** Despite the module name, this implementation uses **pure numpy cosine similarity** (matrix dot product on L2-normalised vectors) instead of the FAISS C++ library. This design choice avoids broken `swig_ptr` bindings in some FAISS builds on Windows while maintaining identical mathematical results to `IndexFlatIP`.

#### How It Works

1. **Initialization:**
   - Loads the **Sentence-Transformers** model (`all-MiniLM-L6-v2`) for sentence/paragraph encoding.
   - Reads `data/articles.json` — a JSON array of fact-check article objects with `title`, `content`, `summary`, and `description` fields.
   - For each article, concatenates all text fields into a single string.

2. **Index Building:**
   - Encodes all article texts into dense embeddings using [`SentenceTransformer.encode()`](https://www.sbert.net/).
   - L2-normalises each embedding vector for cosine similarity via inner product.
   - Caches embeddings to `data/embeddings.npy` and article text hashes to `data/faiss_metadata.pkl`.
   - On subsequent runs, compares cached article texts with current articles and **only rebuilds** if the corpus has changed.

3. **Search — `search(query, k=3)`:**
   - Encodes the query with the same model.
   - Computes cosine similarity: `scores = embeddings @ query_vec.T`.
   - Returns the top-`k` articles sorted by descending similarity, each annotated with a `similarity_score`.

4. **Output Format:**
   ```json
   [
     {
       "title": "Fact-Check: Miracle Spice Cancer Cure",
       "content": "...",
       "similarity_score": 0.7823
     },
     ...
   ]
   ```

---

### 5. Preprocessing Utilities

**File:** `utils/preprocessing.py`

#### `extract_frames(video_path, target_fps=1.0) → List[PIL.Image]`

1. Opens the video with **OpenCV** (`cv2.VideoCapture`).
2. Reads the native FPS and total frame count.
3. Calculates a `frame_interval = round(native_fps / target_fps)` to sample frames at the desired rate.
4. Iterates through every frame, keeping only those at the interval.
5. Converts each selected frame from BGR (OpenCV default) to RGB and wraps it as a `PIL.Image`.

#### `extract_audio(video_path, output_dir) → str`

1. Shells out to **ffmpeg** via `subprocess.run()`.
2. Extracts the audio track and re-encodes it as **16 kHz mono PCM WAV** (`-acodec pcm_s16le -ar 16000 -ac 1`).
3. Saves to `data/processed/<filename>.wav`.
4. Raises `RuntimeError` if ffmpeg is not installed or the extraction fails.

---

### 6. Postprocessing — Weighted Aggregation

**File:** `utils/postprocessing.py`
**Function:** `aggregate_results(...) → dict`

This module combines individual detector outputs into a unified report with a single overall verdict.

#### Aggregation Algorithm

1. **Normalise to fake-probability:** Each detector's result is converted to a `fake_probability` score on a `[0.0, 1.0]` scale:
   - If the detector predicted a **fake label** (e.g., `"fake"`, `"ai-generated"`):
     `fake_prob = confidence`
   - If the detector predicted a **real label** (e.g., `"real"`, `"human-written"`):
     `fake_prob = 1.0 - confidence`

2. **Weighted average** using predefined weights:

   | Detector | Weight |
   |----------|--------|
   | Video    | 0.45   |
   | Audio    | 0.30   |
   | Text     | 0.25   |

   Only detectors that produced a valid result contribute to the average. Weights are renormalised by dividing `weighted_sum` by `total_weight` (the sum of only contributing detector weights).

3. **Final verdict:**
   - `combined_fake_probability = weighted_sum / total_weight`
   - `overall_label = "fake"` if `combined_fake_probability ≥ 0.5`, else `"real"`
   - `combined_confidence = |combined_fake_probability - 0.5| × 2` (maps 0.5→0.0 certainty, 0.0/1.0→1.0 certainty)

4. **Output:**
   ```json
   {
     "video": { ... },
     "audio": { ... },
     "text": { ... },
     "related_articles": [ ... ],
     "combined_fake_probability": 0.7835,
     "combined_confidence": 0.567,
     "overall_label": "fake"
   }
   ```

---

### 7. Document Reader

**File:** `utils/document_reader.py`

Extracts plain text from uploaded or local documents for AI-text detection.

| Format  | Library        | Method |
|---------|---------------|--------|
| `.txt`  | built-in `io` | Direct file read with UTF-8 encoding |
| `.pdf`  | `PyPDF2`      | `PdfReader` → iterate pages → `page.extract_text()` |
| `.docx` | `python-docx` | `Document` → iterate paragraphs → join non-empty text |

The `detect_format(filename)` helper returns the extension if supported, or `None` — used by the API to validate uploads before processing.

---

### 8. Logger

**File:** `utils/logger.py`

Sets up a `"truth.x"` named logger with two handlers:

| Handler | Level | Destination |
|---------|-------|-------------|
| Console (`StreamHandler`) | `WARNING` | Terminal (stderr) |
| File (`FileHandler`) | `DEBUG` | `logs/pipeline.log` |

Format: `2025-01-11 14:23:05 | INFO     | truth.x | Message text`

The `logs/` directory is auto-created on import.

---

## Integrations & External Dependencies

| Integration | Package | Purpose | Used In |
|---|---|---|---|
| **HuggingFace Transformers** | `transformers` | Model loading, tokenization, and inference for image classification and text classification | `deepfake_detector.py`, `ai_text_detector.py` |
| **Sentence-Transformers** | `sentence-transformers` | Dense sentence/paragraph embeddings for semantic search | `faiss_service.py` |
| **PyTorch** | `torch` | GPU/CPU tensor computation, model execution, softmax | `deepfake_detector.py`, `ai_text_detector.py` |
| **OpenCV** | `opencv-python` | Video file reading, frame-by-frame extraction, BGR→RGB conversion | `preprocessing.py` |
| **Pillow** | `Pillow` | PIL Image objects for model input compatibility | `preprocessing.py` |
| **ffmpeg** | system binary | Audio track extraction from video to WAV | `preprocessing.py` (subprocess) |
| **FastAPI** | `fastapi` | REST API framework with async request handling | `api.py` |
| **Uvicorn** | `uvicorn` | ASGI server for running FastAPI | `api.py` |
| **NumPy** | `numpy` | L2 normalisation, matrix multiplication for cosine similarity, embedding storage | `faiss_service.py` |
| **PyPDF2** | `PyPDF2` | PDF text extraction | `document_reader.py` |
| **python-docx** | `python-docx` | DOCX text extraction | `document_reader.py` |
| **PyYAML** | `pyyaml` | YAML configuration parsing | all modules |
| **python-dotenv** | `python-dotenv` | `.env` file loading for environment variables | `main_pipeline.py`, `api.py` |
| **python-multipart** | `python-multipart` | Multipart form data parsing for file uploads | `api.py` |
| **Google Fact Check API** | REST (future) | Live fact-check search — API key stored in `.env` | planned integration |

---

## Configuration Reference

All parameters are centralised in `config/config.yaml`:

### Global Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `device` | string | `"cuda"` | Compute device — `"cuda"` (GPU) or `"cpu"`. Falls back to CPU if CUDA is unavailable. |
| `seed` | int | `42` | Random seed (for reproducibility in future extensions) |

### Video Processing

| Key | Type | Default | Description |
|---|---|---|---|
| `video.model_name` | string | `"dima806/deepfake_vs_real_image_detection"` | HuggingFace model ID for frame-level deepfake classification |
| `video.frame_sample_rate` | float | `1` | Frames to extract per second of video |
| `video.batch_size` | int | `8` | Number of frames per inference batch |
| `video.face_detection` | bool | `false` | Reserved for future face-cropping before classification |

### Audio Processing

| Key | Type | Default | Description |
|---|---|---|---|
| `audio.model_name` | string | `"facebook/wav2vec2-base-960h"` | HuggingFace model ID (stub — not loaded at runtime) |
| `audio.sample_rate` | int | `16000` | Target audio sample rate in Hz for ffmpeg extraction |
| `audio.use_asr` | bool | `false` | Reserved for future ASR (automatic speech recognition) flags |

### Text Processing

| Key | Type | Default | Description |
|---|---|---|---|
| `text.model_name` | string | `"roberta-base-openai-detector"` | HuggingFace model ID for AI-text classification |
| `text.max_length` | int | `512` | Maximum token length for tokeniser truncation |

### FAISS / Retrieval

| Key | Type | Default | Description |
|---|---|---|---|
| `retrieval.articles_path` | string | `"data/articles.json"` | Path to the JSON fact-check article corpus |
| `retrieval.index_path` | string | `"data/faiss/index.faiss"` | Path to FAISS index file (actual embeddings stored as `.npy`) |
| `retrieval.embedder_model` | string | `"sentence-transformers/all-MiniLM-L6-v2"` | Sentence-Transformers model for article/query encoding |
| `retrieval.top_k` | int | `5` | Number of top results to return from similarity search |

### Paths

| Key | Type | Default | Description |
|---|---|---|---|
| `temp_dir` | string | `"data/processed"` | Directory for temporary files (extracted audio, uploaded documents) |

### Environment Variables (`.env`)

| Variable | Description |
|---|---|
| `ARTICLES_FILE` | Override for article corpus path |
| `FAISS_INDEX` | Override for FAISS index path |
| `TEMP_DIR` | Override for temp directory |
| `FACT_CHECK_API_KEY` | Google Fact Check Tools API key (for future live search) |
| `FACT_CHECK_API_URL` | Google Fact Check Tools endpoint |
| `HF_HOME` | HuggingFace model cache directory (default: `./hf_cache`) |

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and available on `PATH`
- **CUDA-capable GPU** (optional, recommended for faster inference)

### 1. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install PyTorch

Install PyTorch **separately** to get the correct CUDA build:

```bash
# With CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

ffmpeg is required for audio extraction from video files.

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to your system `PATH`.
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

### 5. Configure

1. Copy or edit `config/config.yaml` to adjust model names, batch sizes, and paths.
2. Create a `.env` file (see the [Environment Variables](#environment-variables-env) section) with your API keys and path overrides.

---

## Usage

### CLI Pipeline

The CLI entry point is `main_pipeline.py`, which accepts three input modes:

```bash
# Analyze a video for deepfakes
python main_pipeline.py --video data/samples/test.mp4

# Check text for AI generation + search related fact-check articles
python main_pipeline.py --query "A common spice has been found to cure cancer"

# Analyze a document file (PDF, DOCX, or TXT)
python main_pipeline.py --text-file path/to/document.pdf

# Combine video analysis with a text query
python main_pipeline.py --video data/samples/test.mp4 --query "Is this speech authentic?"
```

**Output:** The pipeline prints a formatted detection report to the terminal and logs full details to `logs/pipeline.log`.

#### CLI Execution Flow

1. Load `.env` environment variables.
2. If `--text-file` is provided, read the document and set its text as the `--query`.
3. Load `config/config.yaml`.
4. **Lazy-load** only the models needed for the provided inputs (video → `VideoDeepfakeDetector` + `SyntheticVoiceDetector`; query → `TextAIDetector` + `FAISSSearch`).
5. Run each detector, timing each inference step.
6. Aggregate all results into a unified report via `postprocessing.aggregate_results()`.
7. Print the formatted report.

---

### FastAPI Server

Start the server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### Startup Behaviour

All four models are loaded **once** at startup via an `asynccontextmanager` lifespan handler and reused across all requests. This avoids repeated model loading overhead. On shutdown, all models are cleaned up.

---

### API Endpoints

#### `POST /analyze`

Accepts multipart form data with any combination of the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | File upload | No | Video file (`.mp4`, `.avi`, `.mov`, etc.) |
| `query` | Form string | No | Text to analyse for AI generation |
| `document` | File upload | No | Document (`.pdf`, `.docx`, `.txt`) — text is extracted and used as `query` |

At least **one** of `video`, `query`, or `document` must be provided.

**Response:** JSON report identical to the CLI output format.

**Examples:**

```bash
# Video only
curl -X POST http://localhost:8000/analyze \
  -F "video=@data/samples/test.mp4"

# Text query only
curl -X POST http://localhost:8000/analyze \
  -F "query=A miracle spice cures cancer"

# Document upload
curl -X POST http://localhost:8000/analyze \
  -F "document=@report.pdf"

# Video + text query
curl -X POST http://localhost:8000/analyze \
  -F "video=@data/samples/test.mp4" \
  -F "query=Is this speech authentic?"
```

#### `GET /health`

Returns `{"status": "ok"}` — useful for load balancer health checks.

---

## Testing

The test suite (`tests/test_pipeline.py`) supports two execution modes:

### Direct Mode (default — no server required)

```bash
pytest tests/test_pipeline.py -v
```

Runs the following tests by importing pipeline modules directly:

| Test | What It Validates |
|---|---|
| `test_text_detection` | `TextAIDetector.predict()` returns correct keys and valid label/confidence values |
| `test_faiss_search` | `FAISSSearch.search()` returns articles with similarity scores |
| `test_video_detection` | Frame extraction + `VideoDeepfakeDetector.predict()` (skipped if no sample video) |
| `test_aggregation` | `aggregate_results()` correctly combines stubbed detector outputs |

### API Mode (requires a running server)

```bash
# First, start the server
uvicorn api:app --host 0.0.0.0 --port 8000

# Then, run API tests
pytest tests/test_pipeline.py --api-url http://localhost:8000 -v
```

Runs HTTP tests against `POST /analyze` and `GET /health`.

---

## Diagnostics

**File:** `diagnose.py`

A standalone script that loads both the text and video models and prints their raw `id2label` / `label2id` mappings along with sample inference outputs. Useful for debugging label-resolution issues when swapping models.

```bash
python diagnose.py
```

---

## Future Improvements

- **Audio detector** — Replace the stub with a real synthetic-speech classifier (e.g., fine-tuned wav2vec2 or Whisper-based detector).
- **Face-level analysis** — Add face detection/cropping before frame classification for higher accuracy on multi-face videos.
- **Streaming video support** — Accept video URLs and stream frames instead of requiring local file uploads.
- **Live fact-check integration** — Connect the Google Fact Check Tools API (key already configured in `.env`) alongside the local FAISS index.
- **Batch processing** — Accept multiple files/queries in a single CLI invocation or API request.
- **Authentication & rate limiting** — Secure the API with token-based auth and request throttling for production use.
- **Model versioning** — Track and swap model weights without code changes via config or a model registry.
- **Docker packaging** — Containerise the full stack with CUDA support for reproducible, single-command deployment.
- **Async inference** — Run video, audio, and text detection concurrently for lower latency.
- **Web dashboard** — Build a frontend for visual report exploration and batch upload management.

