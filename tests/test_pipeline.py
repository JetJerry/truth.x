"""Tests for the truth.x detection pipeline.

Can run in two modes:

1. **Direct** (default) — imports and calls the pipeline modules directly.
   No running server required.

       pytest tests/test_pipeline.py

2. **API** — sends HTTP requests to a running FastAPI server.
   Start the server first, then:

       pytest tests/test_pipeline.py --api-url http://localhost:8000
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

SAMPLE_VIDEO = os.path.join(ROOT_DIR, "data", "samples", "test.mp4")
SAMPLE_QUERY = "A common household spice can cure late-stage cancer within weeks"

EXPECTED_REPORT_KEYS = {"video", "audio", "text", "related_articles", "combined_confidence", "overall_label"}
EXPECTED_VIDEO_KEYS = {"per_frame", "average", "label", "confidence"}
EXPECTED_TEXT_KEYS = {"label", "confidence", "ai_probability", "human_probability"}


def pytest_addoption(parser):
    parser.addoption("--api-url", default=None, help="Base URL of a running truth.x API server")


@pytest.fixture
def api_url(request):
    return request.config.getoption("--api-url")


# --------------------------------------------------------------------------
# Direct pipeline tests (no server needed)
# --------------------------------------------------------------------------

class TestDirectPipeline:
    """Call pipeline modules directly."""

    def test_text_detection(self):
        from models.text.ai_text_detector import TextAIDetector

        detector = TextAIDetector()
        result = detector.predict(SAMPLE_QUERY)

        print("\n[Text Detection Result]")
        print(json.dumps(result, indent=2))

        for key in EXPECTED_TEXT_KEYS:
            assert key in result, f"Missing key: {key}"
        assert result["label"] in {"ai-generated", "human-written", "unknown"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_faiss_search(self):
        from services.faiss_service import FAISSSearch

        searcher = FAISSSearch()
        results = searcher.search(SAMPLE_QUERY, k=3)

        print("\n[FAISS Search Results]")
        print(json.dumps(results, indent=2, default=str))

        assert isinstance(results, list)
        if results:
            assert "similarity_score" in results[0]

    @pytest.mark.skipif(not os.path.isfile(SAMPLE_VIDEO), reason="No sample video at data/samples/test.mp4")
    def test_video_detection(self):
        from utils.preprocessing import extract_frames
        from models.video.deepfake_detector import VideoDeepfakeDetector

        frames = extract_frames(SAMPLE_VIDEO, target_fps=1.0)
        assert len(frames) > 0, "No frames extracted"

        detector = VideoDeepfakeDetector()
        result = detector.predict(frames)

        print("\n[Video Detection Result]")
        print(json.dumps({k: v for k, v in result.items() if k != "per_frame"}, indent=2))

        for key in EXPECTED_VIDEO_KEYS:
            assert key in result, f"Missing key: {key}"
        assert result["label"] in {"fake", "real", "unknown"}

    def test_aggregation(self):
        from utils.postprocessing import aggregate_results

        video_stub = {"label": "fake", "confidence": 0.85}
        text_stub = {"label": "ai-generated", "confidence": 0.72}

        report = aggregate_results(video_result=video_stub, text_result=text_stub)

        print("\n[Aggregated Report]")
        print(json.dumps(report, indent=2, default=str))

        for key in EXPECTED_REPORT_KEYS:
            assert key in report, f"Missing key: {key}"
        assert report["combined_confidence"] is not None
        assert report["overall_label"] in {"fake", "real"}


# --------------------------------------------------------------------------
# API tests (requires a running server)
# --------------------------------------------------------------------------

class TestAPI:
    """Send requests to the FastAPI server."""

    def _skip_if_no_server(self, api_url):
        if api_url is None:
            pytest.skip("No --api-url provided; skipping API tests")

    def test_health(self, api_url):
        self._skip_if_no_server(api_url)
        import requests

        resp = requests.get(f"{api_url}/health", timeout=10)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_analyze_query(self, api_url):
        self._skip_if_no_server(api_url)
        import requests

        resp = requests.post(
            f"{api_url}/analyze",
            data={"query": SAMPLE_QUERY},
            timeout=60,
        )

        print("\n[API /analyze Response — query]")
        print(json.dumps(resp.json(), indent=2, default=str))

        assert resp.status_code == 200
        body = resp.json()
        for key in EXPECTED_REPORT_KEYS:
            assert key in body, f"Missing key: {key}"

    @pytest.mark.skipif(not os.path.isfile(SAMPLE_VIDEO), reason="No sample video at data/samples/test.mp4")
    def test_analyze_video(self, api_url):
        self._skip_if_no_server(api_url)
        import requests

        with open(SAMPLE_VIDEO, "rb") as f:
            resp = requests.post(
                f"{api_url}/analyze",
                files={"video": ("test.mp4", f, "video/mp4")},
                timeout=120,
            )

        print("\n[API /analyze Response — video]")
        print(json.dumps(resp.json(), indent=2, default=str))

        assert resp.status_code == 200
        body = resp.json()
        for key in EXPECTED_REPORT_KEYS:
            assert key in body, f"Missing key: {key}"

    def test_analyze_no_input(self, api_url):
        self._skip_if_no_server(api_url)
        import requests

        resp = requests.post(f"{api_url}/analyze", timeout=10)
        assert resp.status_code == 400
