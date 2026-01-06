import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SLUFeatureEngineer:
    def __init__(self, train_split: pd.DataFrame, val_split: pd.DataFrame):
        self.train_split = train_split
        self.val_split = val_split

    def _create_tfidf_features(self):
        # Prepare texts and labels
        X_train_text = [' '.join(words) for words in self.train_split['words']]
        y_train_intent = self.train_split['intent'].values
        X_val_text = [' '.join(words) for words in self.val_split['words']]
        y_val_intent = self.val_split['intent'].values

        # Vectorizer (word + char n-grams)
        intent_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1,2),
            min_df=2,
            max_features=20000,
        )

        logging.info("Creating TF-IDF features for train and validation sets...")
        X_train_tfidf = intent_vectorizer.fit_transform(X_train_text)
        X_val_tfidf = intent_vectorizer.transform(X_val_text)

        return X_train_tfidf, X_val_tfidf

    def _encode_intent(self):
        intent_encoder = LabelEncoder()
        intent_encoder.fit(self.train_split['intent'].values)

        # Slot label encoder (preserve BIO strings as-is for CRF)
        logging.info("Encoding intent labels...")
        all_slot_labels = sorted({slot for slots in self.train_split['slots'] for slot in slots})
        slot_label_to_id = {label: idx for idx, label in enumerate(all_slot_labels)}
        id_to_slot_label = {idx: label for label, idx in slot_label_to_id.items()}

        return intent_encoder, slot_label_to_id, id_to_slot_label    
    
    def engineer_features(self):
        X_train_tfidf, X_val_tfidf = self._create_tfidf_features()
        intent_encoder, slot_label_to_id, id_to_slot_label = self._encode_intent()
        
        return X_train_tfidf, X_val_tfidf, intent_encoder, slot_label_to_id, id_to_slot_label