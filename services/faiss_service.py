# services/faiss_service.py

import os
import json
import pickle
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("truth.x")

# Project root = parent of services/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows in-place."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x /= norms
    return x


class FAISSSearch:
    """Semantic similarity search using Sentence-Transformers embeddings.

    Uses numpy-based cosine similarity (inner product on L2-normalised vectors)
    which is equivalent to a FAISS IndexFlatIP but avoids the broken swig_ptr
    bindings in certain FAISS builds on Windows.

    All settings are read from config/config.yaml under the 'retrieval' section.
    """

    def __init__(self) -> None:
        cfg = _load_config()
        retrieval_cfg = cfg.get("retrieval", {})

        # Resolve all paths relative to project root so CWD doesn't matter
        self.articles_path: str = os.path.join(
            _PROJECT_ROOT, retrieval_cfg.get("articles_path", "data/articles.json")
        )
        # np.save() auto-appends .npy, so always use .npy extension
        raw_index = retrieval_cfg.get("index_path", "data/faiss/embeddings.npy")
        if not raw_index.endswith(".npy"):
            raw_index += ".npy"
        self.index_path: str = os.path.join(_PROJECT_ROOT, raw_index)
        # Store metadata alongside the index
        self.metadata_path: str = os.path.join(
            os.path.dirname(self.index_path), "faiss_metadata.pkl"
        )
        self.model_name: str = retrieval_cfg.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.top_k: int = retrieval_cfg.get("top_k", 5)

        logger.info(
            "FAISSSearch config: articles_path=%s, index_path=%s, model=%s, top_k=%d",
            self.articles_path, self.index_path, self.model_name, self.top_k,
        )

        logger.info(f"Loading sentence-transformer model '{self.model_name}'")
        self.model = SentenceTransformer(self.model_name)

        self.articles: List[Dict[str, Any]] = self._load_articles()
        self.embeddings: Optional[np.ndarray] = self._load_or_build_index()

    def _load_articles(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.articles_path):
            logger.warning(f"Articles file not found: {self.articles_path}")
            return []

        try:
            with open(self.articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            logger.info(f"Loaded {len(articles)} articles from '{self.articles_path}'")
            return articles
        except Exception as e:
            logger.error(f"Error loading articles: {e}")
            return []

    def _get_article_texts(self, articles: List[Dict[str, Any]]) -> List[str]:
        texts = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')
            summary = article.get('summary', '')
            description = article.get('description', '')

            text_parts = [title, summary, description, content]
            text = ' '.join([part for part in text_parts if part]).strip()

            if text:
                texts.append(text)
            else:
                texts.append(f"Article {article.get('id', 'unknown')}")

        return texts

    def _build_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        if not texts:
            logger.warning("No texts to encode")
            return None

        logger.info(f"Encoding {len(texts)} articles")

        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32, copy=True)
        _normalize_l2(embeddings)

        logger.info(f"Embeddings ready: shape={embeddings.shape}, dtype={embeddings.dtype}")

        try:
            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            np.save(self.index_path, embeddings)

            with open(self.metadata_path, 'wb') as f:
                pickle.dump(texts, f)

            logger.info(f"Saved embeddings to '{self.index_path}'")
        except Exception as e:
            logger.warning(f"Failed to save embeddings: {e}")

        return embeddings

    def _load_or_build_index(self) -> Optional[np.ndarray]:
        texts = self._get_article_texts(self.articles)

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                logger.info(f"Loading embeddings from '{self.index_path}'")
                embeddings = np.load(self.index_path)

                with open(self.metadata_path, 'rb') as f:
                    saved_texts = pickle.load(f)

                if saved_texts == texts:
                    logger.info(f"Embeddings loaded ({embeddings.shape[0]} vectors)")
                    return embeddings
                else:
                    logger.info("Articles changed, rebuilding embeddings …")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}, rebuilding …")

        return self._build_embeddings(texts)

    def search(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        """Return the k most similar articles to the query.

        If k is not provided, uses the top_k value from config.yaml.
        """
        if k is None:
            k = self.top_k

        if self.embeddings is None or len(self.articles) == 0:
            logger.warning("No embeddings available; cannot search")
            return []

        try:
            query_vec = self.model.encode([query])
            query_vec = np.array(query_vec, dtype=np.float32, copy=True)
            _normalize_l2(query_vec)

            scores = (self.embeddings @ query_vec.T).squeeze()

            k = min(k, len(self.articles))
            top_indices = np.argsort(scores)[::-1][:k]

            results = []
            for idx in top_indices:
                article = self.articles[idx].copy()
                article['similarity_score'] = round(float(scores[idx]), 4)
                results.append(article)

            logger.info(f"Search returned {len(results)} results (best={results[0]['similarity_score']:.4f})")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []
