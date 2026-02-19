"""Diagnostic: print raw model outputs to identify label mapping issues."""
import os, sys, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

print("=" * 60)
print("TEXT MODEL DIAGNOSIS")
print("=" * 60)
text_model_name = cfg["text"]["model_name"]
print(f"Model: {text_model_name}")

tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)
text_model.eval()

id2label = text_model.config.id2label
label2id = text_model.config.label2id
print(f"id2label: {id2label}")
print(f"label2id: {label2id}")
print(f"Num labels: {text_model.config.num_labels}")

test_texts = {
    "clearly_human": "The detailed agreement between India and the US is undergoing technical and legal processes in both governments and is expected to be signed once these processes are completed, Parliament was informed on Friday.",
    "clearly_ai_fake": "According to totally unreliable science, the reason phones fall screen-down is because gravity specifically targets expensive objects to build character in humans, while Wi-Fi signals disappear in one corner of the room because the laws of physics require routers to create a mystery zone for suspense.",
    "ai_generated_sample": "In the rapidly evolving landscape of artificial intelligence, the convergence of machine learning algorithms and neural network architectures has precipitated a paradigm shift in computational methodologies, enabling unprecedented capabilities in natural language processing and computer vision applications.",
}

for name, text in test_texts.items():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = text_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    print(f"\n--- {name} ---")
    print(f"  Text: {text[:80]}...")
    for idx, prob in enumerate(probs):
        label = id2label[idx]
        print(f"  [{idx}] {label}: {prob:.4f} ({prob:.2%})")

print("\n" + "=" * 60)
print("VIDEO MODEL DIAGNOSIS")
print("=" * 60)
video_model_name = cfg["video"]["model_name"]
print(f"Model: {video_model_name}")

extractor = AutoFeatureExtractor.from_pretrained(video_model_name)
video_model = AutoModelForImageClassification.from_pretrained(video_model_name)
video_model.eval()

vid_id2label = video_model.config.id2label
vid_label2id = video_model.config.label2id
print(f"id2label: {vid_id2label}")
print(f"label2id: {vid_label2id}")
print(f"Num labels: {video_model.config.num_labels}")
