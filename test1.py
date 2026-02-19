# test_faiss_fix.py
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("Testing FAISS fix...")

try:
    from services.faiss_service import FAISSSearch
    print("✓ Import successful")
    
    # Initialize
    print("\nInitializing FAISSSearch...")
    faiss_search = FAISSSearch()
    print("✓ Initialization successful")
    
    # Test search
    print("\nTesting search...")
    results = faiss_search.search("deepfake detection", k=2)
    print(f"✓ Search returned {len(results)} results")
    
    if results:
        print("\nSample results:")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r.get('title', 'No title')} (score: {r.get('similarity_score', 0):.3f})")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()