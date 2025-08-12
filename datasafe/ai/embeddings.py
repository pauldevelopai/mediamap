"""
Text embeddings for similarity and near-duplicate detection
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity

from ..config import HF_EMBED_MODEL, DEDUP_SIM

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manage text embeddings for similarity analysis"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            self.model = SentenceTransformer(HF_EMBED_MODEL)
            logger.info(f"Initialized embeddings model: {HF_EMBED_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            self.model = None
    
    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array of embeddings or None if encoding fails
        """
        if not self.model:
            logger.warning("Embedding model not available")
            return None
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Encoded {len(texts)} texts into embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            return None
    
    def encode_single(self, text: str) -> Optional[np.ndarray]:
        """
        Encode a single text into embedding
        
        Args:
            text: Text to encode
            
        Returns:
            Numpy array embedding or None if encoding fails
        """
        result = self.encode([text])
        return result[0] if result is not None else None
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.encode([text1, text2])
        if embeddings is None:
            return 0.0
        
        try:
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(sim)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def find_similar(self, query_text: str, candidate_texts: List[str], 
                    threshold: float = None) -> List[Tuple[int, str, float]]:
        """
        Find similar texts from a list of candidates
        
        Args:
            query_text: Text to find similarities for
            candidate_texts: List of candidate texts
            threshold: Similarity threshold (defaults to DEDUP_SIM)
            
        Returns:
            List of tuples (index, text, similarity_score) above threshold
        """
        if threshold is None:
            threshold = DEDUP_SIM
        
        if not candidate_texts:
            return []
        
        all_texts = [query_text] + candidate_texts
        embeddings = self.encode(all_texts)
        
        if embeddings is None:
            return []
        
        try:
            query_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            results = []
            for i, (text, sim) in enumerate(zip(candidate_texts, similarities)):
                if sim >= threshold:
                    results.append((i, text, float(sim)))
            
            # Sort by similarity score, descending
            results.sort(key=lambda x: x[2], reverse=True)
            
            logger.info(f"Found {len(results)} similar texts above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Similar text search failed: {e}")
            return []
    
    def is_duplicate(self, text1: str, text2: str, threshold: float = None) -> bool:
        """
        Check if two texts are near-duplicates
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (defaults to DEDUP_SIM)
            
        Returns:
            True if texts are considered duplicates
        """
        if threshold is None:
            threshold = DEDUP_SIM
        
        sim = self.similarity(text1, text2)
        return sim >= threshold
    
    def deduplicate(self, texts: List[str], threshold: float = None) -> List[int]:
        """
        Find indices of unique texts, removing near-duplicates
        
        Args:
            texts: List of texts to deduplicate
            threshold: Similarity threshold (defaults to DEDUP_SIM)
            
        Returns:
            List of indices for unique texts
        """
        if threshold is None:
            threshold = DEDUP_SIM
        
        if len(texts) <= 1:
            return list(range(len(texts)))
        
        embeddings = self.encode(texts)
        if embeddings is None:
            return list(range(len(texts)))
        
        try:
            unique_indices = []
            
            for i, embedding in enumerate(embeddings):
                is_unique = True
                
                # Check against all previously selected unique texts
                for j in unique_indices:
                    sim = cosine_similarity([embedding], [embeddings[j]])[0][0]
                    if sim >= threshold:
                        is_unique = False
                        break
                
                if is_unique:
                    unique_indices.append(i)
            
            logger.info(f"Deduplicated {len(texts)} texts to {len(unique_indices)} unique items")
            return unique_indices
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return list(range(len(texts)))

# Global embedding manager instance
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get or create global embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Convenience function to calculate text similarity
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    manager = get_embedding_manager()
    return manager.similarity(text1, text2)

def find_duplicates(texts: List[str], threshold: float = None) -> List[int]:
    """
    Convenience function to find unique text indices
    
    Args:
        texts: List of texts to deduplicate
        threshold: Similarity threshold
        
    Returns:
        List of indices for unique texts
    """
    manager = get_embedding_manager()
    return manager.deduplicate(texts, threshold)
