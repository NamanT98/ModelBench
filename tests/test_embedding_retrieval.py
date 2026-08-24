import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import numpy as np

from modelbench.types import TextToSQLSample
from modelbench.retrieval import EmbeddingRetriever, NumpyVectorIndex, create_retriever

@pytest.fixture
def mock_training_data():
    return [
        TextToSQLSample(
            question="What is the name of the youngest head?",
            db_id="department_management",
            db_path="datasets/spider/database/department_management/department_management.sqlite",
            gold_sql="SELECT name FROM head ORDER BY age ASC LIMIT 1"
        ),
        TextToSQLSample(
            question="How many heads are older than 56?",
            db_id="department_management",
            db_path="datasets/spider/database/department_management/department_management.sqlite",
            gold_sql="SELECT count(*) FROM head WHERE age > 56"
        ),
        TextToSQLSample(
            question="List the names and born states of all heads.",
            db_id="department_management",
            db_path="datasets/spider/database/department_management/department_management.sqlite",
            gold_sql="SELECT name ,  born_state FROM head"
        )
    ]

@pytest.fixture
def mock_dev_data():
    return [
        TextToSQLSample(
            question="Find the total budget of departments managed by heads older than 50.",
            db_id="department_management",
            db_path="datasets/spider/database/department_management/department_management.sqlite",
            gold_sql="SELECT sum(budget) FROM department JOIN head ON ... WHERE age > 50"
        )
    ]

def test_embedding_retriever_initialization(mock_training_data, tmp_path):
    # Test initialization with mocking the sentence transformer to avoid actual downloads
    with patch("modelbench.retrieval.SentenceTransformer") as MockTransformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        MockTransformer.return_value = mock_model
        
        cache_prefix = str(tmp_path / "test_cache")
        retriever = EmbeddingRetriever(
            mock_training_data, 
            embedding_model_id="test/model", 
            cache_prefix=cache_prefix
        )
        
        assert retriever.index is not None
        assert retriever.index.dimension == 3
        assert len(retriever.index.sample_ids) == 3
        
        # Verify cache files were created
        assert Path(f"{cache_prefix}.json").exists()
        assert Path(f"{cache_prefix}.npy").exists()
        
        # Verify metadata
        with open(f"{cache_prefix}.json") as f:
            metadata_file = json.load(f)
            assert metadata_file["metadata"]["embedding_model_id"] == "test/model"
            assert metadata_file["metadata"]["corpus_size"] == 3

def test_cache_loading(mock_training_data, tmp_path):
    with patch("modelbench.retrieval.SentenceTransformer") as MockTransformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0], [1.0], [1.0]])
        MockTransformer.return_value = mock_model
        
        cache_prefix = str(tmp_path / "test_cache")
        
        # First init creates cache
        retriever1 = EmbeddingRetriever(mock_training_data, embedding_model_id="test/model", cache_prefix=cache_prefix)
        assert MockTransformer.call_count == 1
        
        # Second init loads cache
        retriever2 = EmbeddingRetriever(mock_training_data, embedding_model_id="test/model", cache_prefix=cache_prefix)
        # Should not have called encode or SentenceTransformer constructor again during init
        assert MockTransformer.call_count == 1
        assert retriever2.index is not None
        assert retriever2.index.dimension == 1

def test_cache_invalidation_model_id(mock_training_data, tmp_path):
    with patch("modelbench.retrieval.SentenceTransformer") as MockTransformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0], [1.0], [1.0]])
        MockTransformer.return_value = mock_model
        
        cache_prefix = str(tmp_path / "test_cache")
        
        # Init with model A
        EmbeddingRetriever(mock_training_data, embedding_model_id="modelA", cache_prefix=cache_prefix)
        assert MockTransformer.call_count == 1
        
        # Init with model B -> should rebuild
        EmbeddingRetriever(mock_training_data, embedding_model_id="modelB", cache_prefix=cache_prefix)
        assert MockTransformer.call_count == 2

def test_retrieval_determinism(mock_training_data, tmp_path):
    with patch("modelbench.retrieval.SentenceTransformer") as MockTransformer:
        mock_model = MagicMock()
        # Make the training embeddings exactly the same for sample 1 and 2, and different for 0
        mock_model.encode.side_effect = [
            # Training embeddings
            np.array([
                [1.0, 0.0],
                [0.0, 1.0],  # Sample 1
                [0.0, 1.0]   # Sample 2 (Identical embedding to 1)
            ]),
            # Query embedding
            np.array([0.0, 1.0])
        ]
        MockTransformer.return_value = mock_model
        
        cache_prefix = str(tmp_path / "test_cache")
        retriever = EmbeddingRetriever(mock_training_data, embedding_model_id="test", cache_prefix=cache_prefix)
        
        # When querying [0.0, 1.0], sample 1 and 2 will tie with score 1.0.
        # Deterministic sorting should use sample.question string to tie-break.
        # Q1: "How many heads are older than 56?" (H...)
        # Q2: "List the names and born states of all heads." (L...)
        # Expected order (descending score, ascending text): Q1 then Q2
        results = retriever.retrieve("query", k=2)
        assert len(results) == 2
        assert results[0].question == "How many heads are older than 56?"
        assert results[1].question == "List the names and born states of all heads."

def test_k_0_and_empty(mock_training_data, tmp_path):
    with patch("modelbench.retrieval.SentenceTransformer") as MockTransformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0], [1.0], [1.0]])
        MockTransformer.return_value = mock_model
        
        cache_prefix = str(tmp_path / "test_cache")
        retriever = EmbeddingRetriever(mock_training_data, embedding_model_id="test", cache_prefix=cache_prefix)
        
        assert len(retriever.retrieve("query", k=0)) == 0
        assert len(retriever.retrieve("query", k=-1)) == 0

def test_leakage_prevention(mock_training_data, mock_dev_data, tmp_path):
    """Ensure dev data can NEVER enter the training index or cache metadata."""
    with patch("modelbench.retrieval.SentenceTransformer") as MockTransformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0], [1.0], [1.0]])
        MockTransformer.return_value = mock_model
        
        cache_prefix = str(tmp_path / "test_cache")
        retriever = EmbeddingRetriever(mock_training_data, embedding_model_id="test", cache_prefix=cache_prefix)
        
        # Verify index only contains training samples
        assert len(retriever.index.sample_ids) == 3
        for dev_sample in mock_dev_data:
            assert dev_sample.question not in retriever.index.sample_ids
            
        # Verify metadata explicitly
        with open(f"{cache_prefix}.json") as f:
            metadata_file = json.load(f)
            cached_ids = metadata_file["sample_ids"]
            for dev_sample in mock_dev_data:
                assert dev_sample.question not in cached_ids

def test_create_retriever(mock_training_data):
    with pytest.raises(ValueError, match="embedding_model must be provided"):
        create_retriever("embedding", mock_training_data)
        
    with patch("modelbench.retrieval.SentenceTransformer") as MockTransformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0], [1.0], [1.0]])
        MockTransformer.return_value = mock_model
        
        retriever = create_retriever(
            "embedding", 
            mock_training_data, 
            embedding_model="test/model"
        )
        assert isinstance(retriever, EmbeddingRetriever)
