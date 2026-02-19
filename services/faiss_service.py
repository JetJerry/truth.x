# services/faiss_service.py

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("truth.x")

class FAISSSearch:
    def __init__(self, 
                 articles_path: str = "data/articles.json",
                 index_path: str = "data/faiss_index.bin",
                 metadata_path: str = "data/faiss_metadata.pkl",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.articles_path = articles_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        
        # Load the sentence transformer model
        logger.info(f"Loading sentence-transformer model '{model_name}'")
        self.model = SentenceTransformer(model_name)
        
        # Load or build the FAISS index
        self.index, self.articles = self._load_or_build_index()
        
    def _load_articles(self) -> List[Dict[str, Any]]:
        """Load articles from JSON file"""
        if not os.path.exists(self.articles_path):
            logger.warning(f"Articles file not found: {self.articles_path}")
            # Return empty list if file doesn't exist
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
        """Extract text from articles for embedding"""
        texts = []
        for article in articles:
            # Combine title and content for better search
            title = article.get('title', '')
            content = article.get('content', '')
            # Also include any other text fields that might be useful
            summary = article.get('summary', '')
            description = article.get('description', '')
            
            # Combine all text fields
            text_parts = [title, summary, description, content]
            text = ' '.join([part for part in text_parts if part]).strip()
            
            if text:  # Only add if there's text
                texts.append(text)
            else:
                # If no text fields, use a placeholder
                texts.append(f"Article {article.get('id', 'unknown')}")
        
        return texts
    
    def _build_index(self, texts: List[str]) -> Optional[faiss.Index]:
        """Build FAISS index from article texts"""
        if not texts:
            logger.warning("No texts to index, creating empty index")
            # Create empty index with correct dimension
            return faiss.IndexFlatIP(384)  # all-MiniLM-L6-v2 dimension is 384
        
        logger.info(f"Encoding {len(texts)} articles for FAISS index")
        
        try:
            # Encode texts to embeddings
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # CRITICAL FIX: Ensure it's a numpy array with float32 dtype
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            embeddings = embeddings.astype('float32')
            
            # Verify shape
            logger.info(f"Embeddings shape: {embeddings.shape}")
            
            # Normalize for cosine similarity (Inner Product)
            # This expects a numpy array
            faiss.normalize_L2(embeddings)
            
            # Create index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            logger.info(f"Built FAISS index with {index.ntotal} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            import traceback
            traceback.print_exc()
            # Return empty index as fallback
            return faiss.IndexFlatIP(384)
    
    def _load_or_build_index(self):
        """Load existing index or build new one"""
        articles = self._load_articles()
        texts = self._get_article_texts(articles)
        
        # Try to load existing index
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                logger.info(f"Loading FAISS index from '{self.index_path}'")
                index = faiss.read_index(self.index_path)
                
                with open(self.metadata_path, 'rb') as f:
                    saved_texts = pickle.load(f)
                
                # Check if index matches current articles
                if saved_texts == texts:
                    logger.info("FAISS index loaded successfully")
                    return index, articles
                else:
                    logger.info("Articles changed, rebuilding index...")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, rebuilding...")
        
        # Build new index
        index = self._build_index(texts)
        
        # Save index and metadata if we have articles
        if texts and index and index.ntotal > 0:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                
                # Save index
                faiss.write_index(index, self.index_path)
                
                # Save metadata
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(texts, f)
                
                logger.info(f"Saved FAISS index to '{self.index_path}'")
            except Exception as e:
                logger.warning(f"Failed to save index: {e}")
        else:
            logger.warning("No index to save")
        
        return index, articles
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar articles"""
        if not hasattr(self, 'index') or self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Ensure numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            query_embedding = query_embedding.astype('float32')
            
            # Normalize query
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(k, self.index.ntotal)
            if k == 0:
                return []
                
            scores, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.articles):
                    article = self.articles[idx].copy()
                    article['similarity_score'] = float(scores[0][i])
                    results.append(article)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []