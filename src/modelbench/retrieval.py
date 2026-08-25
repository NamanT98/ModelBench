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

    def retrieve_with_scores(self, question: str, k: int | None = None) -> list[tuple[TextToSQLSample, float]]:
        if not self._index:
            return []
            
        target_tokens = _nltk_tokenize_question(question)
        if not target_tokens:
            # Fallback if question tokenizes to nothing (e.g., just punctuation)
            limit = k if k is not None else len(self._index)
            return [(sample, 0.0) for sample, _ in self._index[:limit]]
            
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
        
        if k is not None:
            scores = scores[:k]
            
        return [(sample, score) for score, _, sample in scores]

    def retrieve(self, question: str, k: int) -> list[TextToSQLSample]:
        if k <= 0:
            return []
        scored = self.retrieve_with_scores(question, k)
        return [sample for sample, _ in scored]


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
                
            # Move model to CPU after building index to free VRAM for the generator
            self.model.cpu()
                
    def retrieve_with_scores(self, question: str, k: int | None = None) -> list[tuple[TextToSQLSample, float]]:
        if self.index is None:
            return []
            
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required")
            # Load directly to CPU for inference to preserve VRAM for the generator
            self.model = SentenceTransformer(self.embedding_model_id, device="cpu")
            
        # Must normalize query for cosine similarity via dot product
        query_embedding = self.model.encode(question, normalize_embeddings=True)
        
        limit = k if k is not None else len(self.training_data)
        sample_ids, scores = self.index.search(query_embedding, limit)
        
        return [(self._sample_map[sid], float(score)) for sid, score in zip(sample_ids, scores)]

    def retrieve(self, question: str, k: int) -> list[TextToSQLSample]:
        if k <= 0:
            return []
        scored = self.retrieve_with_scores(question, k)
        return [sample for sample, _ in scored]

from modelbench.types import RetrievalResult

class HybridRetriever:
    """Retrieves examples using a hybrid of lexical and semantic signals."""
    
    def __init__(
        self, 
        lexical_retriever: ExampleRetriever, 
        semantic_retriever: ExampleRetriever,
        strategy: str,
        alpha: float | None = None,
        rrf_constant: int = 60,
        union_n: int = 10
    ):
        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever
        self.strategy = strategy
        self.alpha = alpha
        self.rrf_constant = rrf_constant
        self.union_n = union_n

    def retrieve(self, question: str, k: int) -> RetrievalResult | list[TextToSQLSample]:
        if k <= 0:
            return []
            
        start = time.perf_counter()
        
        if self.strategy == "hybrid_score":
            if self.alpha is None:
                raise ValueError("hybrid_alpha must be set for hybrid_score strategy")
            
            lex_all = self.lexical_retriever.retrieve_with_scores(question, k=None)
            sem_all = self.semantic_retriever.retrieve_with_scores(question, k=None)
            
            lex_scores = [score for _, score in lex_all]
            lex_min, lex_max = (min(lex_scores), max(lex_scores)) if lex_scores else (0, 0)
            lex_dict = {}
            for sample, score in lex_all:
                key = (sample.db_id, sample.question)
                if lex_max == lex_min:
                    lex_dict[key] = 0.5
                else:
                    lex_dict[key] = (score - lex_min) / (lex_max - lex_min)
            
            sem_scores = [score for _, score in sem_all]
            sem_min, sem_max = (min(sem_scores), max(sem_scores)) if sem_scores else (0, 0)
            sem_dict = {}
            for sample, score in sem_all:
                key = (sample.db_id, sample.question)
                if sem_max == sem_min:
                    sem_dict[key] = 0.5
                else:
                    sem_dict[key] = (score - sem_min) / (sem_max - sem_min)
                    
            hybrid_scores = []
            for sample, _ in lex_all:
                key = (sample.db_id, sample.question)
                l_score = lex_dict.get(key, 0.0)
                s_score = sem_dict.get(key, 0.0)
                h_score = self.alpha * l_score + (1 - self.alpha) * s_score
                hybrid_scores.append((h_score, sample.question, sample, l_score, s_score))
                
            hybrid_scores.sort(key=lambda x: (-x[0], x[1]))
            top_k = hybrid_scores[:k]
            
            lex_top_k = [sample for sample, _ in sorted(lex_all, key=lambda x: (-x[1], x[0].question))[:k]]
            sem_top_k = [sample for sample, _ in sorted(sem_all, key=lambda x: (-x[1], x[0].question))[:k]]
            overlap = len(set((s.db_id, s.question) for s in lex_top_k) & set((s.db_id, s.question) for s in sem_top_k))
            
            return RetrievalResult(
                samples=[x[2] for x in top_k],
                diagnostics={
                    "strategy": self.strategy,
                    "alpha": self.alpha,
                    "latency_seconds": time.perf_counter() - start,
                    "lexical_top_k_ids": [(s.db_id, s.question) for s in lex_top_k],
                    "semantic_top_k_ids": [(s.db_id, s.question) for s in sem_top_k],
                    "hybrid_top_k_ids": [(x[2].db_id, x[2].question) for x in top_k],
                    "overlap_lex_sem": overlap,
                    "fused_scores": [x[0] for x in top_k],
                }
            )
            
        elif self.strategy == "hybrid_rrf":
            lex_all = self.lexical_retriever.retrieve_with_scores(question, k=None)
            sem_all = self.semantic_retriever.retrieve_with_scores(question, k=None)
            
            lex_sorted = sorted(lex_all, key=lambda x: (-x[1], x[0].question))
            sem_sorted = sorted(sem_all, key=lambda x: (-x[1], x[0].question))
            
            lex_ranks = { (s.db_id, s.question): i + 1 for i, (s, _) in enumerate(lex_sorted) }
            sem_ranks = { (s.db_id, s.question): i + 1 for i, (s, _) in enumerate(sem_sorted) }
            
            hybrid_scores = []
            for sample, _ in lex_all:
                key = (sample.db_id, sample.question)
                r_lex = lex_ranks.get(key, len(lex_all))
                r_sem = sem_ranks.get(key, len(sem_all))
                rrf_score = 1.0 / (self.rrf_constant + r_lex) + 1.0 / (self.rrf_constant + r_sem)
                hybrid_scores.append((rrf_score, sample.question, sample, r_lex, r_sem))
                
            hybrid_scores.sort(key=lambda x: (-x[0], x[1]))
            top_k = hybrid_scores[:k]
            
            overlap = len(set((s.db_id, s.question) for s, _ in lex_sorted[:k]) & set((s.db_id, s.question) for s, _ in sem_sorted[:k]))
            
            return RetrievalResult(
                samples=[x[2] for x in top_k],
                diagnostics={
                    "strategy": self.strategy,
                    "rrf_constant": self.rrf_constant,
                    "latency_seconds": time.perf_counter() - start,
                    "lexical_top_k_ids": [(s.db_id, s.question) for s, _ in lex_sorted[:k]],
                    "semantic_top_k_ids": [(s.db_id, s.question) for s, _ in sem_sorted[:k]],
                    "hybrid_top_k_ids": [(x[2].db_id, x[2].question) for x in top_k],
                    "overlap_lex_sem": overlap,
                    "fused_scores": [x[0] for x in top_k],
                }
            )
            
        elif self.strategy == "hybrid_union":
            lex_all = self.lexical_retriever.retrieve_with_scores(question, k=None)
            sem_all = self.semantic_retriever.retrieve_with_scores(question, k=None)
            
            lex_sorted = sorted(lex_all, key=lambda x: (-x[1], x[0].question))
            sem_sorted = sorted(sem_all, key=lambda x: (-x[1], x[0].question))
            
            lex_top_n = lex_sorted[:self.union_n]
            sem_top_n = sem_sorted[:self.union_n]
            
            union_keys = set((s.db_id, s.question) for s, _ in lex_top_n) | set((s.db_id, s.question) for s, _ in sem_top_n)
            
            union_pool = {}
            for s, score in lex_all:
                key = (s.db_id, s.question)
                if key in union_keys:
                    union_pool[key] = {"sample": s, "lex": score}
            for s, score in sem_all:
                key = (s.db_id, s.question)
                if key in union_keys:
                    union_pool[key]["sem"] = score
                    
            lex_scores = [v.get("lex", 0.0) for v in union_pool.values()]
            sem_scores = [v.get("sem", 0.0) for v in union_pool.values()]
            
            lex_min, lex_max = (min(lex_scores), max(lex_scores)) if lex_scores else (0, 0)
            sem_min, sem_max = (min(sem_scores), max(sem_scores)) if sem_scores else (0, 0)
            
            hybrid_scores = []
            for key, v in union_pool.items():
                s = v["sample"]
                l_score = v.get("lex", 0.0)
                s_score = v.get("sem", 0.0)
                
                norm_l = 0.5 if lex_max == lex_min else (l_score - lex_min) / (lex_max - lex_min)
                norm_s = 0.5 if sem_max == sem_min else (s_score - sem_min) / (sem_max - sem_min)
                
                h_score = 0.5 * norm_l + 0.5 * norm_s
                hybrid_scores.append((h_score, s.question, s))
                
            hybrid_scores.sort(key=lambda x: (-x[0], x[1]))
            top_k = hybrid_scores[:k]
            
            overlap = len(set((s.db_id, s.question) for s, _ in lex_sorted[:k]) & set((s.db_id, s.question) for s, _ in sem_sorted[:k]))
            
            return RetrievalResult(
                samples=[x[2] for x in top_k],
                diagnostics={
                    "strategy": self.strategy,
                    "union_n": self.union_n,
                    "candidates_considered": len(union_pool),
                    "latency_seconds": time.perf_counter() - start,
                    "lexical_top_k_ids": [(s.db_id, s.question) for s, _ in lex_sorted[:k]],
                    "semantic_top_k_ids": [(s.db_id, s.question) for s, _ in sem_sorted[:k]],
                    "hybrid_top_k_ids": [(x[2].db_id, x[2].question) for x in top_k],
                    "overlap_lex_sem": overlap,
                    "fused_scores": [x[0] for x in top_k],
                }
            )
            
        else:
            raise ValueError(f"Unknown hybrid strategy: {self.strategy}")


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
    elif strategy.startswith("hybrid_"):
        lexical = JaccardSimilarityRetriever(training_data)
        embedding_model = kwargs.get("embedding_model")
        if not embedding_model:
            raise ValueError("embedding_model must be provided for hybrid strategies")
        semantic = EmbeddingRetriever(training_data, embedding_model_id=embedding_model)
        
        return HybridRetriever(
            lexical_retriever=lexical,
            semantic_retriever=semantic,
            strategy=strategy,
            alpha=kwargs.get("hybrid_alpha"),
            rrf_constant=kwargs.get("hybrid_rrf_constant", 60),
            union_n=kwargs.get("hybrid_union_n", 10)
        )
    raise ValueError(f"Unsupported retrieval strategy: {strategy!r}")
