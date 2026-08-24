"""Demonstration retrieval for Few-Shot Text-to-SQL."""

from __future__ import annotations

import time
import logging
from typing import Protocol, runtime_checkable

from modelbench.types import TextToSQLSample
from modelbench.schema import _nltk_tokenize_question

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


def create_retriever(
    strategy: str, 
    training_data: list[TextToSQLSample]
) -> ExampleRetriever:
    """Create a retriever instance from a strategy name."""
    if strategy == "jaccard_nltk":
        return JaccardSimilarityRetriever(training_data)
    raise ValueError(f"Unsupported retrieval strategy: {strategy!r}")
