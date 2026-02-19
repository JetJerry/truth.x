
import sys
import os
import torch
import numpy as np
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.video.deepfake_detector import VideoDeepfakeDetector
from models.audio.synthetic_voice_detector import SyntheticVoiceDetector
from models.text.ai_text_detector import TextAIDetector
from utils.logger import logger

def test_video_model():
    print("\n=== Testing Video Model ===")
    try:
        detector = VideoDeepfakeDetector()
        print("Model loaded successfully.")
        
        # Create dummy frames (black images)
        dummy_frames = [Image.new('RGB', (224, 224), color='black') for _ in range(2)]
        
        # Test predict
        result = detector.predict(dummy_frames)
        print(f"Prediction result: {result}")
        
    except Exception as e:
        print(f"Video model test FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_audio_model():
    print("\n=== Testing Audio Model ===")
    try:
        detector = SyntheticVoiceDetector()
        print("Model loaded successfully.")
        
        # We can't easily pass dummy audio to predict() because it expects a file path.
        # But we can test if model loaded.
        if detector.model is not None:
             print("Audio model initialized OK.")
        else:
             print("Audio model failed to initialize.")
             
    except Exception as e:
        print(f"Audio model test FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_text_model():
    print("\n=== Testing Text Model ===")
    try:
        detector = TextAIDetector()
        print("Model loaded successfully.")
        
        result = detector.predict("This is a simple test sentence.")
        print(f"Prediction result: {result}")
        
    except Exception as e:
        print(f"Text model test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_model()
    test_audio_model()
    test_text_model()
