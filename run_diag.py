"""Post-fix diagnostic: verify all model predictions are correct."""
import os, sys, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

output_lines = []
def p(s=""):
    output_lines.append(s)
    print(s)

# ================================================================
# TEXT MODEL
# ================================================================
p("=" * 60)
p(f"TEXT MODEL: {cfg['text']['model_name']}")
p("=" * 60)

from models.text.ai_text_detector import TextAIDetector

detector = TextAIDetector()

tests = {
    "clearly_human": "The detailed agreement between India and the US is undergoing technical and legal processes in both governments and is expected to be signed once these processes are completed, Parliament was informed on Friday.",
    "ai_fake_claim": "According to totally unreliable science, the reason phones fall screen-down is because gravity specifically targets expensive objects to build character in humans.",
    "ai_generated_text": "In the rapidly evolving landscape of artificial intelligence, the convergence of machine learning algorithms and neural network architectures has precipitated a paradigm shift in computational methodologies, enabling unprecedented capabilities in natural language processing and computer vision applications.",
    "chatgpt_style": "Certainly! Here are some key points to consider. First, it is important to understand the nuances. Second, we should acknowledge the complexity. In conclusion, there are multiple perspectives to evaluate.",
}

correct = 0
total = len(tests)
expected = {
    "clearly_human": "human-written",
    "ai_fake_claim": "human-written",
    "ai_generated_text": "ai-generated",
    "chatgpt_style": "ai-generated",
}

for name, text in tests.items():
    result = detector.predict(text)
    exp = expected[name]
    match = "PASS" if result["label"] == exp else "FAIL"
    if result["label"] == exp:
        correct += 1
    p(f"\n--- {name} ---")
    p(f"  Expected: {exp}")
    p(f"  Got:      {result['label']} (confidence={result['confidence']:.4f})")
    p(f"  AI prob:  {result['ai_probability']:.4f}")
    p(f"  [{match}]")

p(f"\nText model accuracy: {correct}/{total}")

# ================================================================
# FAISS CONFIG
# ================================================================
p("\n" + "=" * 60)
p("FAISS CONFIG CHECK")
p("=" * 60)

from services.faiss_service import FAISSSearch
import inspect

fs = FAISSSearch()
p(f"  articles_path = {fs.articles_path}")
p(f"  index_path    = {fs.index_path}")
p(f"  model_name    = {fs.model_name}")
p(f"  top_k         = {fs.top_k}")
p(f"  Config match:   top_k={fs.top_k} == config={cfg['retrieval']['top_k']} ? {'PASS' if fs.top_k == cfg['retrieval']['top_k'] else 'FAIL'}")

results = fs.search("deepfake detection")
p(f"  Search returned {len(results)} results (expected {fs.top_k})")

# ================================================================
# POSTPROCESSING
# ================================================================
p("\n" + "=" * 60)
p("POSTPROCESSING CHECK")
p("=" * 60)
from utils.postprocessing import aggregate_results

report = aggregate_results(
    video_result={"label": "real", "confidence": 0.9},
    text_result={"label": "ai-generated", "confidence": 0.9},
)
p(f"Video=real(0.90), Text=ai-gen(0.90):")
p(f"  overall={report['overall_label']}, fake_prob={report['combined_fake_probability']}")

p("\n" + "=" * 60)
p("ALL CHECKS COMPLETE")
p("=" * 60)

with open("diag_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
