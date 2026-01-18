import pytest
import pandas as pd
from src.data.splitter import SLUDataSplitter

@pytest.fixture
def group_test_df():
    """
    Creates a dataframe where some utterances are identical.
    'Play music' appears twice. 
    In a group-aware split, these two rows should NOT be separated.
    """
    return pd.DataFrame({
        'words': [
            ['Play', 'music'], ['Play', 'music'], 
            ['Set', 'alarm'], ['Check', 'weather'],
            ['Stop', 'music'], ['Lights', 'on']
        ],
        'intent': ['play_music', 'play_music', 'set_alarm', 'get_weather', 'stop', 'lights_on']
    })

def test_split_proportions(group_test_df):
    # Testing a 50/50 split on 6 samples
    splitter = SLUDataSplitter(group_test_df, test_size=0.5, random_state=42)
    train_df, val_df = splitter.split_data()
    
    # Check that the data is actually split
    assert len(train_df) + len(val_df) == len(group_test_df)
    assert len(train_df) > 0
    assert len(val_df) > 0

def test_group_integrity(group_test_df):
    """
    Ensures that identical word sequences are kept in the same split.
    """
    splitter = SLUDataSplitter(group_test_df, test_size=0.3, random_state=42)
    train_df, val_df = splitter.split_data()
    
    # Get sequences as strings for easy comparison
    train_seqs = set(train_df['words'].apply(lambda x: ' '.join(x)))
    val_seqs = set(val_df['words'].apply(lambda x: ' '.join(x)))
    
    # THE CORE ASSERTION: The intersection of sequences must be empty.
    # If 'Play music' is in train, it MUST NOT be in validation.
    assert train_seqs.intersection(val_seqs) == set(), "Overlapping sequences found in train and val!"

def test_random_state_reproducibility(group_test_df):
    """Ensures that the same random_state produces the same split."""
    s1 = SLUDataSplitter(group_test_df, test_size=0.5, random_state=10)
    train1, _ = s1.split_data()
    
    s2 = SLUDataSplitter(group_test_df, test_size=0.5, random_state=10)
    train2, _ = s2.split_data()
    
    pd.testing.assert_frame_equal(train1, train2)

def test_word_sequence_column_creation(group_test_df):
    """Ensures the helper column 'word_sequence' is created if missing."""
    splitter = SLUDataSplitter(group_test_df)
    assert 'word_sequence' not in group_test_df.columns
    
    splitter.split_data()
    assert 'word_sequence' in splitter.df.columns