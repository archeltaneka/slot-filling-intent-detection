import logging

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SLUDataSplitter:
    def __init__(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        self.df = df
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        # Split into train and test sets
        if 'word_sequence' not in self.df.columns: # Ensure grouping key exists
            self.df['word_sequence'] = self.df['words'].apply(lambda x: ' '.join(x))

        X = self.df.index.values
        y = self.df['intent'].values
        groups = self.df['word_sequence'].values

        # Group-aware split (keeps identical utterances within the same fold)
        logging.info(f"Splitting {self.test_size*100:.1f}% of data into validation set and {100-self.test_size*100:.1f}% of data into training set...")
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, val_idx = next(gss.split(X, y, groups=groups))

        train_data_group = self.df.loc[train_idx].reset_index(drop=True)
        val_data_group = self.df.loc[val_idx].reset_index(drop=True)

        logging.info(f"Original dataset size: {len(self.df):,}")
        logging.info(f"Training set size: {len(train_data_group):,} ({len(train_data_group)/len(self.df)*100:.1f}%)")
        logging.info(f"Validation set size: {len(val_data_group):,} ({len(val_data_group)/len(self.df)*100:.1f}%)")

        return train_data_group, val_data_group