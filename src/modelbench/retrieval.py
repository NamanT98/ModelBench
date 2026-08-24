"""Demonstration retrieval for Few-Shot Text-to-SQL."""

from __future__ import annotations

import time
import logging
from typing import Protocol, runtime_checkable

from modelbench.types import TextToSQLSample
from modelbench.schema import _nltk_tokenize_question

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


@runtime_checkable
class ExampleRetriever(Protocol):
    """Protocol for example retrieval strategies."""

    def retrieve(self, question: str, k: int) -> list[TextToSQLSample]:
        """Retrieve the top-k most relevant demonstrations for a question."""
        ...


class JaccardSimilarityRetriever:
    """Retrieves examples using NLTK-tokenized Jaccard similarity.
    
    This is a deterministic, lightweight retrieval method that avoids
    heavy ML dependencies like embeddings.
    """

    def __init__(self, training_data: list[TextToSQLSample], cache_path: str = ".jaccard_cache.pkl") -> None:
        self.training_data = training_data
        self._index: list[tuple[TextToSQLSample, set[str]]] = []
        
        import pickle
        from pathlib import Path
        cache_file = Path(cache_path)
        if cache_file.exists():
            logger.info("Loading pre-computed Jaccard index from %s", cache_file)
            try:
                with cache_file.open("rb") as f:
                    self._index = pickle.load(f)
                if len(self._index) == len(self.training_data):
                    return
                logger.info("Cache length mismatch. Rebuilding index...")
                self._index = []
            except Exception as e:
                logger.warning("Failed to load Jaccard cache: %s", e)
                
        logger.info("Initializing Jaccard similarity index with %d samples...", len(training_data))
        start = time.perf_counter()
        
        # Pre-tokenize all training questions
        for sample in self.training_data:
            tokens = _nltk_tokenize_question(sample.question)
            self._index.append((sample, tokens))
            
        elapsed = time.perf_counter() - start
        logger.info("Jaccard index initialized in %.2fs", elapsed)
        
        try:
            with cache_file.open("wb") as f:
                pickle.dump(self._index, f)
            logger.info("Saved Jaccard index cache to %s", cache_file)
        except Exception as e:
            logger.warning("Failed to save Jaccard cache: %s", e)

    def retrieve(self, question: str, k: int) -> list[TextToSQLSample]:
        if k <= 0 or not self._index:
            return []
            
        target_tokens = _nltk_tokenize_question(question)
        if not target_tokens:
            # Fallback if question tokenizes to nothing (e.g., just punctuation)
            return [sample for sample, _ in self._index[:k]]
            
        scores: list[tuple[float, str, TextToSQLSample]] = []
        
        for sample, sample_tokens in self._index:
            if not sample_tokens:
                continue
                
            intersection = len(target_tokens & sample_tokens)
            union = len(target_tokens | sample_tokens)
            score = intersection / union if union > 0 else 0.0
            
            # Tie-breaker using sample.question to ensure deterministic ordering
            scores.append((score, sample.question, sample))
            
        # Sort descending by score, then ascending by question string
        scores.sort(key=lambda x: (-x[0], x[1]))
        
        return [sample for _, _, sample in scores[:k]]


class NumpyVectorIndex:
    """A simple in-memory vector index using NumPy.
    
    Suitable for small to medium corpora (~7,000 examples).
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embeddings = None
        self.sample_ids: list[str] = []

    def add(self, embeddings, sample_ids: list[str]):
        import numpy as np
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")
        if len(embeddings) != len(sample_ids):
            raise ValueError("Embeddings and sample_ids must have the same length")
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.sample_ids.extend(sample_ids)

    def search(self, query_embedding, k: int) -> tuple[list[str], list[float]]:
        import numpy as np
        if self.embeddings is None:
            return [], []
            
        # Since embeddings are L2 normalized, dot product is mathematically equivalent to cosine similarity.
        scores = np.dot(self.embeddings, query_embedding)
        
        # Tie-breaker using sample_ids to ensure deterministic ordering
        scored_items = [
            (float(scores[i]), self.sample_ids[i])
            for i in range(len(scores))
        ]
        
        # Sort descending by score, then ascending by sample_id
        scored_items.sort(key=lambda x: (-x[0], x[1]))
        
        top_k = scored_items[:k]
        return [item[1] for item in top_k], [item[0] for item in top_k]

    def save(self, cache_file_prefix: str, metadata: dict):
        import numpy as np
        import json
        np.save(f"{cache_file_prefix}.npy", self.embeddings)
        with open(f"{cache_file_prefix}.json", "w") as f:
            json.dump({
                "metadata": metadata,
                "dimension": self.dimension,
                "sample_ids": self.sample_ids
            }, f, indent=2)

    @classmethod
    def load(cls, cache_file_prefix: str) -> tuple['NumpyVectorIndex', dict]:
        import numpy as np
        import json
        with open(f"{cache_file_prefix}.json", "r") as f:
            data = json.load(f)
            
        index = cls(dimension=data["dimension"])
        index.embeddings = np.load(f"{cache_file_prefix}.npy")
        index.sample_ids = data["sample_ids"]
        
        return index, data["metadata"]


class EmbeddingRetriever:
    """Retrieves examples using semantic embeddings and cosine similarity.
    
    This is an M6 strategy that evaluates whether semantic similarity 
    produces better Text-to-SQL demonstrations than lexical matching.
    """
    
    def __init__(self, training_data: list[TextToSQLSample], embedding_model_id: str, cache_prefix: str = ".embedding_cache_v1") -> None:
        if not embedding_model_id:
            raise ValueError("embedding_model_id is required for EmbeddingRetriever")
            
        self.training_data = training_data
        self.embedding_model_id = embedding_model_id
        self.cache_prefix = cache_prefix
        
        self._sample_map = {s.question: s for s in training_data}
        
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for EmbeddingRetriever. Run `pip install sentence-transformers numpy`.")
            
        import json
        from pathlib import Path
        
        cache_metadata_path = Path(f"{cache_prefix}.json")
        cache_vectors_path = Path(f"{cache_prefix}.npy")
        
        # Strict metadata for cache invalidation
        expected_metadata = {
            "embedding_model_id": self.embedding_model_id,
            "corpus_size": len(self.training_data),
            "retrieval_metric": "cosine_similarity_via_dot_product",
            "normalize_embeddings": True
        }
        
        self.index = None
        self.model = None
        
        if cache_metadata_path.exists() and cache_vectors_path.exists():
            try:
                index, metadata = NumpyVectorIndex.load(cache_prefix)
                if metadata == expected_metadata:
                    expected_sample_ids = [s.question for s in self.training_data]
                    if index.sample_ids == expected_sample_ids:
                        logger.info("Loading pre-computed embedding index from %s", cache_prefix)
                        self.index = index
                    else:
                        logger.info("Cache sample IDs mismatch. Rebuilding embedding index...")
                else:
                    logger.info("Cache metadata mismatch. Rebuilding embedding index...")
            except Exception as e:
                logger.warning("Failed to load embedding cache: %s", e)
                
        if self.index is None:
            logger.info("Initializing embedding index with %s for %d samples...", self.embedding_model_id, len(training_data))
            start = time.perf_counter()
            
            self.model = SentenceTransformer(self.embedding_model_id)
            
            sample_ids = [s.question for s in self.training_data]
            
            # Embeddings are L2 normalized so dot product behaves as cosine similarity
            logger.info("Encoding %d questions...", len(sample_ids))
            embeddings = self.model.encode(sample_ids, normalize_embeddings=True, show_progress_bar=False)
            
            self.index = NumpyVectorIndex(dimension=embeddings.shape[1])
            self.index.add(embeddings, sample_ids)
            
            elapsed = time.perf_counter() - start
            logger.info("Embedding index initialized in %.2fs", elapsed)
            
            try:
                self.index.save(self.cache_prefix, expected_metadata)
                logger.info("Saved embedding index cache to %s.*", self.cache_prefix)
            except Exception as e:
                logger.warning("Failed to save embedding cache: %s", e)
                
    def retrieve(self, question: str, k: int) -> list[TextToSQLSample]:
        if k <= 0 or self.index is None:
            return []
            
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required")
            self.model = SentenceTransformer(self.embedding_model_id)
            
        start = time.perf_counter()
        
        # Must normalize query for cosine similarity via dot product
        query_embedding = self.model.encode(question, normalize_embeddings=True)
        
        sample_ids, scores = self.index.search(query_embedding, k)
        
        # Mapping ids back to TextToSQLSample
        retrieved_samples = [self._sample_map[sid] for sid in sample_ids]
        
        # Store diagnostics on the instance for runner to pick up
        self.last_retrieval_scores = scores
        self.last_retrieval_latency = time.perf_counter() - start
        
        return retrieved_samples


def create_retriever(
    strategy: str, 
    training_data: list[TextToSQLSample],
    **kwargs
) -> ExampleRetriever:
    """Create a retriever instance from a strategy name."""
    if strategy == "jaccard_nltk":
        return JaccardSimilarityRetriever(training_data)
    elif strategy == "embedding":
        embedding_model = kwargs.get("embedding_model")
        if not embedding_model:
            raise ValueError("embedding_model must be provided for the 'embedding' strategy")
        return EmbeddingRetriever(training_data, embedding_model_id=embedding_model)
    raise ValueError(f"Unsupported retrieval strategy: {strategy!r}")
