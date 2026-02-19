# services/faiss_service.py

import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("truth.x")


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
    """

    def __init__(self,
                 articles_path: str = "data/articles.json",
                 index_path: str = "data/embeddings.npy",
                 metadata_path: str = "data/faiss_metadata.pkl",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

        self.articles_path = articles_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        logger.info(f"Loading sentence-transformer model '{model_name}'")
        self.model = SentenceTransformer(model_name)

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

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return the k most similar articles to the query."""
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
