import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import string
from src.data.builder import SLUDataBuilder

@pytest.fixture
def mock_df():
    """Provides a minimal DataFrame for testing."""
    return pd.DataFrame({
        'words': [['Play', 'Music', '123']],
        'slots': [['O', 'B-service', 'O']]
    })

@pytest.fixture
def builder(mock_df):
    """Initializes the builder with mock data."""
    return SLUDataBuilder(train_split=mock_df, val_split=mock_df)

def test_get_word_shape(builder):
    # Test lowercase collapse
    assert builder._get_word_shape("music") == "x"
    # Test title case collapse
    assert builder._get_word_shape("Music") == "Xx"
    # Test digits and punctuation
    assert builder._get_word_shape("123-ABC") == "dpX"
    # Test mixed
    assert builder._get_word_shape("Apple!") == "Xxp"

def test_token2features_structure(builder):
    sentence = ["Play", "some", "jazz"]
    features = builder._token2features(sentence, 0)
    
    # Check if essential keys exist
    assert 'word.lower' in features
    assert 'word.shape' in features
    assert features['word.lower'] == "play"
    
    # Check Boundary condition (BOS - Beginning of Sentence)
    # At index 0, prev1 and prev2 should be BOS
    assert features['BOS/EOS_prev1'] is True
    assert features['BOS/EOS_prev2'] is True

def test_build_crf_dataset(builder, mock_df):
    X, y, tokens = builder.build_crf_dataset(mock_df)
    
    # Verify dimensions
    assert len(X) == 1  # One sentence
    assert len(X[0]) == 3  # Three words in that sentence
    assert len(y[0]) == 3  # Three slot labels
    assert tokens[0] == ["Play", "Music", "123"]
    
    # Verify a specific feature value
    assert X[0][2]['word.isdigit'] is True  # "123" is a digit