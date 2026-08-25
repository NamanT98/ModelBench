import pytest
from unittest.mock import MagicMock
from modelbench.types import TextToSQLSample
from modelbench.retrieval import JaccardSimilarityRetriever, EmbeddingRetriever, HybridRetriever

@pytest.fixture
def mock_training_data():
    return [
        TextToSQLSample("query1", "db1", "/path1", "SELECT 1"),
        TextToSQLSample("query2", "db1", "/path1", "SELECT 2"),
        TextToSQLSample("query3", "db2", "/path2", "SELECT 3"),
        TextToSQLSample("query4", "db2", "/path2", "SELECT 4"),
    ]

@pytest.fixture
def mock_lexical():
    retriever = MagicMock()
    def mock_retrieve_lex(question, k=None):
        scores = [(TextToSQLSample("query1", "db1", "/path1", "SELECT 1"), 0.8),
                  (TextToSQLSample("query2", "db1", "/path1", "SELECT 2"), 0.5),
                  (TextToSQLSample("query3", "db2", "/path2", "SELECT 3"), 0.2),
                  (TextToSQLSample("query4", "db2", "/path2", "SELECT 4"), 0.0)]
        if k is not None:
            return scores[:k]
        return scores
    retriever.retrieve_with_scores.side_effect = mock_retrieve_lex
    return retriever

@pytest.fixture
def mock_semantic():
    retriever = MagicMock()
    def mock_retrieve_sem(question, k=None):
        scores = [(TextToSQLSample("query4", "db2", "/path2", "SELECT 4"), 0.9),
                  (TextToSQLSample("query3", "db2", "/path2", "SELECT 3"), 0.7),
                  (TextToSQLSample("query2", "db1", "/path1", "SELECT 2"), 0.4),
                  (TextToSQLSample("query1", "db1", "/path1", "SELECT 1"), 0.1)]
        if k is not None:
            return scores[:k]
        return scores
    retriever.retrieve_with_scores.side_effect = mock_retrieve_sem
    return retriever

def test_hybrid_score_normalization(mock_lexical, mock_semantic):
    retriever = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_score", alpha=0.5)
    result = retriever.retrieve("test query", k=3)
    
    assert len(result.samples) == 3
    assert result.samples[0].question == "query1"
    assert result.samples[1].question == "query2"
    assert result.samples[2].question == "query4"

def test_hybrid_score_alpha_bias(mock_lexical, mock_semantic):
    retriever = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_score", alpha=1.0)
    result = retriever.retrieve("test query", k=3)
    
    assert result.samples[0].question == "query1"
    assert result.samples[1].question == "query2"
    assert result.samples[2].question == "query3"
    
    retriever = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_score", alpha=0.0)
    result = retriever.retrieve("test query", k=3)
    
    assert result.samples[0].question == "query4"
    assert result.samples[1].question == "query3"
    assert result.samples[2].question == "query2"

def test_hybrid_rrf(mock_lexical, mock_semantic):
    retriever = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_rrf", rrf_constant=60)
    result = retriever.retrieve("test query", k=4)
    
    assert result.samples[0].question == "query1"
    assert result.samples[1].question == "query4"
    assert result.samples[2].question == "query2"
    assert result.samples[3].question == "query3"

def test_hybrid_union(mock_lexical, mock_semantic):
    retriever = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_union", union_n=2)
    result = retriever.retrieve("test query", k=3)
    
    assert result.samples[0].question == "query1"
    assert result.samples[1].question == "query2"
    assert result.samples[2].question == "query4"
    
def test_hybrid_union_partial(mock_lexical, mock_semantic):
    retriever = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_union", union_n=1)
    result = retriever.retrieve("test query", k=2)
    
    assert len(result.samples) == 2
    assert result.samples[0].question == "query1"
    assert result.samples[1].question == "query4"
    
    assert result.diagnostics["candidates_considered"] == 2
    assert result.diagnostics["union_n"] == 1
