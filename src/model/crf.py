import logging

import pandas as pd
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

from src.evaluation import SLUEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class CRFModel:
    def __init__(self, slot_label_to_id):
        from sklearn_crfsuite import CRF
        self.model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.slot_label_to_id = slot_label_to_id

    def train(self, X_train, y_train):
        logging.info("Training CRF model...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def _convert_to_ids(self, slot_sequences):
        """Internal helper to convert label strings to IDs."""
        return [[self.slot_label_to_id.get(s, 0) for s in seq] for seq in slot_sequences]

    def evaluate(self, evaluator, X_val, y_val_true_labels):
        logging.info("Evaluating CRF model...")
        
        # Generate predictions
        y_pred_labels = self.predict(X_val)
        
        # Prepare data for the evaluator
        y_true_ids = self._convert_to_ids(y_val_true_labels)
        y_pred_ids = self._convert_to_ids(y_pred_labels)
        val_lengths = [len(seq) for seq in y_val_true_labels]
        
        # Handle Dummy Intents (since CRF is slot-only)
        dummy_intents = [0] * len(y_true_ids)

        # Run Evaluation
        raw_results = evaluator.evaluate_model(
            y_true_intents=dummy_intents,
            y_pred_intents=dummy_intents,
            y_true_slots=y_true_ids,
            y_pred_slots=y_pred_ids,
            lengths=val_lengths,
            verbose=False
        )
        results = {
            'model_name': 'CRF (Slot Filling)',
            'slot_f1': raw_results.get('slot_f1', 0),
            'entity_f1': raw_results.get('entity_f1', 0),
            'predictions': y_pred_ids
        }
        
        logging.info(f"CRF Results | Slot F1: {results['slot_f1']:.4f} | Entity F1: {results['entity_f1']:.4f}")
        
        return results