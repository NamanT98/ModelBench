import pytest
from modelbench.schema import _nltk_tokenize_question, _nltk_tokenize_name, _nltk_name_matches_tokens

def test_nltk_tokenize_name_snake_case():
    tokens = _nltk_tokenize_name("Customer_ID")
    assert "customer" in tokens
    assert "id" in tokens

def test_nltk_tokenize_name_camel_case():
    tokens = _nltk_tokenize_name("customerID")
    assert "customer" in tokens
    assert "id" in tokens

def test_nltk_tokenize_name_hyphen():
    tokens = _nltk_tokenize_name("customer-id")
    assert "customer" in tokens
    assert "id" in tokens

def test_nltk_lemmatization():
    tokens = _nltk_tokenize_question("How many countries have singers?")
    assert "country" in tokens
    assert "singer" in tokens
    # stop words should be removed
    assert "how" not in tokens
    assert "have" not in tokens

def test_nltk_stop_words_preserves_meaning():
    # checking negative words "not" is removed per our decision
    tokens = _nltk_tokenize_question("users who do not have friends")
    assert "user" in tokens
    assert "friend" in tokens
    assert "not" not in tokens
    assert "do" not in tokens

def test_nltk_name_matches_tokens():
    question_tokens = _nltk_tokenize_question("Show the names of all the singers")
    # "singer" is in question_tokens
    assert _nltk_name_matches_tokens("singers", question_tokens)
    assert _nltk_name_matches_tokens("Singer_ID", question_tokens)
    assert not _nltk_name_matches_tokens("countries", question_tokens)
