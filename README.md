# truth.x

A modular, GPU-enabled AI system for detecting deepfake videos, AI-generated speech, AI-generated text, and retrieving related fact-check articles. Designed as a model-only pipeline with a CLI controller and an optional FastAPI server.

## Features

- **Video deepfake detection** — samples frames from a video and classifies each as fake or real using a HuggingFace image-classification model, returning per-frame and aggregate scores.
- **Synthetic voice detection** — stub module ready for future implementation; extracts audio via ffmpeg for analysis.
- **AI-generated text detection** — classifies a text passage as human-written or AI-generated using a RoBERTa-based detector.
- **Fact-check article retrieval** — encodes a query with Sentence-Transformers and searches a FAISS index of articles, returning the most relevant matches with similarity scores.
- **Weighted confidence aggregation** — combines detector outputs into a single report with an overall label and confidence score.

## Project Structure

```
truth.x/
├── api.py                  # FastAPI server
├── main_pipeline.py        # CLI entry point
├── requirements.txt        # Pinned dependencies
├── .env                    # Environment variables (not committed)
├── config/
│   └── config.yaml         # Model names, parameters, paths
├── data/
│   ├── articles.json       # Sample fact-check articles
│   ├── samples/            # Place test videos here
│   └── processed/          # Extracted audio, temp files
├── models/
│   ├── video/
│   │   └── deepfake_detector.py
│   ├── audio/
│   │   └── synthetic_voice_detector.py
│   └── text/
│       └── ai_text_detector.py
├── services/
│   └── faiss_service.py    # FAISS index + search
├── utils/
│   ├── logger.py           # Console + file logging
│   ├── preprocessing.py    # Frame extraction, audio extraction
│   └── postprocessing.py   # Result aggregation
└── logs/
    └── pipeline.log        # Auto-created at runtime
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install PyTorch with CUDA

Install PyTorch separately to get the correct CUDA build for your system:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you do not have a CUDA-capable GPU, use the CPU build instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes both `faiss-cpu` and `faiss-gpu`. Install only the one that matches your hardware. Comment out or remove the other.

### 4. Install ffmpeg

ffmpeg is required for audio extraction from video files.

- **Windows:** download from https://ffmpeg.org/download.html and add to PATH.
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

## Configuration

All tuneable parameters live in `config/config.yaml`:

| Section | Key | Description |
|---|---|---|
| **Global** | `device` | `"cuda"` or `"cpu"` |
| **Video** | `model_name` | HuggingFace model for frame classification |
| | `frame_sample_rate` | Frames to extract per second of video |
| | `batch_size` | Frames per inference batch |
| **Audio** | `model_name` | HuggingFace model (stub, for future use) |
| | `sample_rate` | Target audio sample rate in Hz |
| **Text** | `model_name` | HuggingFace sequence classifier |
| | `max_length` | Max token length for truncation |
| **Retrieval** | `embedder_model` | Sentence-Transformers model for FAISS |
| | `articles_path` | Path to the JSON article corpus |
| | `top_k` | Number of articles to return |

Environment-specific overrides and API keys go in `.env`.

## Usage

### CLI Pipeline

Analyze a video for deepfakes:

```bash
python main_pipeline.py --video data/samples/test.mp4
```

Check text for AI generation and search related articles:

```bash
python main_pipeline.py --query "A common spice has been found to cure cancer"
```

Combine both:

```bash
python main_pipeline.py --video data/samples/test.mp4 --query "Is this speech authentic?"
```

The pipeline prints a JSON report to stdout and logs details to `logs/pipeline.log`.

### FastAPI Server

Start the server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

All models are loaded once at startup and reused across requests.

**Endpoints:**

- `POST /analyze` — upload a video file and/or provide a text query.
- `GET /health` — returns `{"status": "ok"}`.

Example with curl:

```bash
# Video only
curl -X POST http://localhost:8000/analyze \
  -F "video=@data/samples/test.mp4"

# Text query only
curl -X POST http://localhost:8000/analyze \
  -F "query=A miracle spice cures cancer"

# Both
curl -X POST http://localhost:8000/analyze \
  -F "video=@data/samples/test.mp4" \
  -F "query=Is this speech authentic?"
```

## Future Improvements

- **Audio detector** — replace the stub with a real synthetic-speech classifier (e.g. fine-tuned wav2vec2 or Whisper-based detector).
- **Face-level analysis** — add face detection/cropping before frame classification for higher accuracy.
- **Streaming video support** — accept video URLs and stream frames instead of requiring local files.
- **Expanded article corpus** — integrate live fact-check APIs (Google Fact Check Tools, ClaimBuster) alongside the local FAISS index.
- **Batch processing** — accept multiple files/queries in a single request.
- **Authentication and rate limiting** — secure the API for production use.
- **Model versioning** — track and swap model weights without code changes.
- **Docker packaging** — containerise the full stack with CUDA support for reproducible deployment.
