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

def test_hybrid_rrf_candidate_n(mock_lexical, mock_semantic):
    retriever = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_rrf", rrf_constant=60, candidate_n=2)
    result = retriever.retrieve("test query", k=3)
    
    assert len(result.samples) == 3
    # Union of top-2 lexical and top-2 semantic
    # lex: query1, query2. sem: query4, query3
    # union: query1, query2, query3, query4
    # Expected scores:
    # query1: lex rank 1, sem absent (contrib 0). Score: 1/61
    # query2: lex rank 2, sem absent (contrib 0). Score: 1/62
    # query3: lex absent, sem rank 2 (contrib 0). Score: 1/62
    # query4: lex absent, sem rank 1 (contrib 0). Score: 1/61
    # query1 and query4 tie for 1st (Score 1/61). Tie-breaker by question str: query1 > query4
    # query2 and query3 tie for 3rd (Score 1/62). Tie-breaker: query2 > query3
    assert result.samples[0].question == "query1"
    assert result.samples[1].question == "query4"
    assert result.samples[2].question == "query2"

def test_hybrid_rrf_missing_rank():
    # Test explicitly that missing rank gives 0.0 contribution
    lex = MagicMock()
    lex.retrieve_with_scores.return_value = [(TextToSQLSample("q1", "db", "p", "s"), 1.0)]
    sem = MagicMock()
    sem.retrieve_with_scores.return_value = [(TextToSQLSample("q2", "db", "p", "s"), 1.0)]
    
    retriever = HybridRetriever(lex, sem, strategy="hybrid_rrf", rrf_constant=1, candidate_n=1)
    res = retriever.retrieve("query", k=2)
    
    # q1 score = 1/(1+1) + 0 = 0.5
    # q2 score = 0 + 1/(1+1) = 0.5
    # tie break: q1 > q2
    assert res.samples[0].question == "q1"
    assert res.samples[1].question == "q2"
    assert res.diagnostics["fused_scores"][0] == 0.5
    assert res.diagnostics["fused_scores"][1] == 0.5

def test_hybrid_rrf_corpus_size(mock_lexical, mock_semantic):
    # candidate_n >= corpus_size should behave exactly like candidate_n = None
    ret_full = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_rrf", rrf_constant=60, candidate_n=None)
    res_full = ret_full.retrieve("query", k=4)
    
    ret_bounded = HybridRetriever(mock_lexical, mock_semantic, strategy="hybrid_rrf", rrf_constant=60, candidate_n=100)
    res_bounded = ret_bounded.retrieve("query", k=4)
    
    for s1, s2 in zip(res_full.samples, res_bounded.samples):
        assert s1.question == s2.question
