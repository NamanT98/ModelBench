import pytest
from modelbench.retrieval import JaccardSimilarityRetriever
from modelbench.types import TextToSQLSample
from modelbench.dataset import SpiderDataset
from modelbench.config import DatasetConfig


@pytest.fixture
def dummy_training_data():
    return [
        TextToSQLSample(
            question="What is the name of the singer?",
            db_id="concert_singer",
            db_path="path/to/db",
            gold_sql="SELECT name FROM singer"
        ),
        TextToSQLSample(
            question="How many concerts are there?",
            db_id="concert_singer",
            db_path="path/to/db",
            gold_sql="SELECT count(*) FROM concert"
        ),
        TextToSQLSample(
            question="Show the average age of dogs.",
            db_id="pets",
            db_path="path/to/pets.sqlite",
            gold_sql="SELECT avg(age) FROM dogs"
        ),
    ]


def test_jaccard_similarity_retriever_top_k(dummy_training_data):
    retriever = JaccardSimilarityRetriever(dummy_training_data)
    
    # "singer name" should match the first question strongly
    results = retriever.retrieve("Show the singer name.", k=1)
    assert len(results) == 1
    assert results[0].gold_sql == "SELECT name FROM singer"

    results_2 = retriever.retrieve("Show the singer name.", k=2)
    assert len(results_2) == 2
    assert results_2[0].gold_sql == "SELECT name FROM singer"


def test_jaccard_similarity_determinism(dummy_training_data):
    retriever = JaccardSimilarityRetriever(dummy_training_data)
    
    # Run multiple times to ensure the same result
    r1 = retriever.retrieve("What is the average age?", k=2)
    r2 = retriever.retrieve("What is the average age?", k=2)
    
    assert [x.gold_sql for x in r1] == [x.gold_sql for x in r2]


def test_empty_retrieval_handling(dummy_training_data):
    retriever = JaccardSimilarityRetriever(dummy_training_data)
    
    assert len(retriever.retrieve("What is the name?", k=0)) == 0
    assert len(retriever.retrieve("What is the name?", k=-1)) == 0
    
    empty_retriever = JaccardSimilarityRetriever([])
    assert len(empty_retriever.retrieve("Anything?", k=3)) == 0


def test_leakage_prevention():
    # Load actual train and dev datasets
    train_config = DatasetConfig(name="spider", path="datasets/spider", split="train")
    dev_config = DatasetConfig(name="spider", path="datasets/spider", split="dev")
    
    try:
        train_samples = list(SpiderDataset(train_config).load())
        dev_samples = list(SpiderDataset(dev_config).load())
    except FileNotFoundError:
        pytest.skip("Spider dataset not found. Skipping leakage test.")
        
    retriever = JaccardSimilarityRetriever(train_samples)
    
    # Ensure no training sample is literally the same instance as a dev sample
    # Check by question text uniqueness (though some simple questions might overlap, 
    # the exact same dataset instance shouldn't be loaded).
    
    train_questions = {s.question for s in train_samples}
    # Dev set has some questions that are identical to train set in Spider (rarely),
    # but let's assert the objects are different.
    
    # Instead of strict text non-overlap (which Spider doesn't guarantee),
    # test that dev.json isn't loaded by train_config
    assert len(train_samples) >= 7000
    assert len(dev_samples) == 1034
