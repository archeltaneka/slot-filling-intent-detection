import logging

import joblib
import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.ensemble import RandomForestClassifier


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class BaselineModel:
    def __init__(self, slot_label_to_id, intent_encoder):
        from sklearn_crfsuite import CRF
        self.crf_model = CRF(
            c1=np.float64(0.10019258321841086), 
            c2=np.float64(0.005543179485876227)
        )
        self.rf_model = RandomForestClassifier(**{
            'n_estimators': 200,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 0.5,
            'max_depth': None,
            'class_weight': None,
            'bootstrap': False
        })
        self.slot_label_to_id = slot_label_to_id
        self.intent_encoder = intent_encoder

    def _train_intent_crf(self, X_train, y_train):
        logging.info("Training Intent CRF model...")
        self.rf_model.fit(X_train, y_train)

    def _train_slot_rf(self, X_train, y_train):
        logging.info("Training Slot RF model...")
        self.crf_model.fit(X_train, y_train)

    def _predict_intent_crf(self, X):
        return self.rf_model.predict(X)

    def _predict_slot_rf(self, X):
        return self.crf_model.predict(X)

    def _convert_slot_to_ids(self, slot_sequences):
        """Internal helper to convert label strings to IDs."""
        return [[self.slot_label_to_id.get(s, 0) for s in seq] for seq in slot_sequences]

    def _convert_intent_to_ids(self, intent_label):
        return self.intent_encoder.transform(intent_label)

    def train(self, X_train_intent, X_train_slot, y_train_intent, y_train_slot):
        logging.info("Training Intent RF model...")
        self.rf_model.fit(X_train_intent, y_train_intent)
        logging.info("Training Slot CRF model...")
        self.crf_model.fit(X_train_slot, y_train_slot)

    def predict(self, X_intent, X_slot):
        y_intent_pred = self._predict_intent_crf(X_intent)
        y_slot_pred = self._predict_slot_rf(X_slot)
        return y_intent_pred, y_slot_pred

    def evaluate(self, evaluator, X_val_intent, y_val_true_intent, X_val_slot, y_val_true_slot):
        logging.info("Evaluating CRF model...")
        
        # Generate predictions
        y_intent_pred, y_slot_pred = self.predict(X_val_intent, X_val_slot)
        
        # Prepare data for the evaluator
        y_slot_true_ids = self._convert_slot_to_ids(y_val_true_slot)
        y_slot_pred_ids = self._convert_slot_to_ids(y_slot_pred)
        y_val_true_intent_ids = self._convert_intent_to_ids(y_val_true_intent)
        val_lengths = [len(seq) for seq in y_val_true_slot]

        # Run Evaluation
        raw_results = evaluator.evaluate_model(
            y_true_intents=y_val_true_intent_ids,
            y_pred_intents=y_intent_pred,
            y_true_slots=y_slot_true_ids,
            y_pred_slots=y_slot_pred_ids,
            lengths=val_lengths,
            verbose=False
        )
        results = {
            'model_name': 'Baseline (CRF Slot Filling + RF Intent)',
            'intent_accuracy': raw_results.get('intent_accuracy', 0),
            'slot_f1': raw_results.get('slot_f1', 0),
            'entity_f1': raw_results.get('entity_f1', 0)
        }
        
        logging.info(f"Baseline Results | Intent Accuracy: {results['intent_accuracy']:.4f} | Slot F1: {results['slot_f1']:.4f} | Entity F1: {results['entity_f1']:.4f}")
        
        return results

    def save(self, filepath):
        """Saves the entire bundle to a single file."""
        joblib.dump(self, filepath)
        print(f"Model saved successfully to {filepath}")

    @staticmethod
    def load(filepath):
        """Loads the entire bundle from a file."""
        return joblib.load(filepath)