#!/usr/bin/env python3
"""
Evaluation Template for Slot Filling and Intent Detection

This template provides standard evaluation metrics for both tasks.
Students can use these functions to evaluate their models consistently.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class SLUEvaluator:
    """
    Evaluator for Spoken Language Understanding tasks:
    - Intent Detection (classification)
    - Slot Filling (sequence labeling)
    """
    
    def __init__(self, slot_vocab: Dict[str, int], intent_vocab: Dict[str, int]):
        """
        Initialize evaluator with vocabularies.
        
        Args:
            slot_vocab: Mapping from slot labels to IDs
            intent_vocab: Mapping from intent labels to IDs
        """
        self.slot_vocab = slot_vocab
        self.intent_vocab = intent_vocab
        
        # Reverse mappings
        self.id_to_slot = {v: k for k, v in slot_vocab.items()}
        self.id_to_intent = {v: k for k, v in intent_vocab.items()}
    
    def intent_accuracy(self, y_true: List[int], y_pred: List[int]) -> float:
        """
        Calculate intent detection accuracy.
        
        Args:
            y_true: True intent IDs
            y_pred: Predicted intent IDs
            
        Returns:
            float: Accuracy score
        """
        # Calculate the proportion of correctly predicted intents
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true) if len(y_true) > 0 else 0.0
    
    def slot_metrics(self, y_true: List[List[int]], y_pred: List[List[int]], 
                    lengths: List[int]) -> Dict[str, float]:
        """
        Calculate slot filling metrics (precision, recall, F1).
        
        Args:
            y_true: True slot label sequences (padded)
            y_pred: Predicted slot label sequences (padded)  
            lengths: Actual sequence lengths (before padding)
            
        Returns:
            Dict with precision, recall, f1 scores
        """
        # Remove padding based on actual lengths
        # Calculate precision, recall, F1 for slot filling
        # Handle the BIO tagging scheme properly
        
        all_true_slots = []
        all_pred_slots = []
        
        # Remove padding and flatten sequences
        for true_seq, pred_seq, length in zip(y_true, y_pred, lengths):
            # Only consider non-padded positions
            true_slots = true_seq[:length]
            pred_slots = pred_seq[:length]
            
            all_true_slots.extend(true_slots)
            all_pred_slots.extend(pred_slots)
        
        # Calculate metrics
        return self._calculate_classification_metrics(all_true_slots, all_pred_slots)
    
    def _calculate_classification_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate precision, recall, F1 for multi-class classification."""
        # Get unique classes (excluding padding)
        classes = set(y_true) | set(y_pred)
        if 0 in classes:  # Remove padding token
            classes.remove(0)
        
        true_positives = defaultdict(int)
        false_positives = defaultdict(int) 
        false_negatives = defaultdict(int)
        
        # For each class, count TP, FP, FN
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == 0 and pred_label == 0:  # Skip padding
                continue
                
            if true_label == pred_label and true_label != 0:
                true_positives[true_label] += 1
            else:
                if pred_label != 0:
                    false_positives[pred_label] += 1
                if true_label != 0:
                    false_negatives[true_label] += 1
        
        # Calculate macro averages
        precisions = []
        recalls = []
        f1s = []
        
        for class_id in classes:
            tp = true_positives[class_id]
            fp = false_positives[class_id]
            fn = false_negatives[class_id]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        return {
            'precision': np.mean(precisions) if precisions else 0.0,
            'recall': np.mean(recalls) if recalls else 0.0,
            'f1': np.mean(f1s) if f1s else 0.0
        }
    
    def entity_level_f1(self, y_true: List[List[int]], y_pred: List[List[int]], 
                       lengths: List[int]) -> float:
        """
        Calculate entity-level F1 score (stricter evaluation).
        An entity is correct only if both boundaries and type are correct.
        
        Args:
            y_true: True slot label sequences
            y_pred: Predicted slot label sequences
            lengths: Actual sequence lengths
            
        Returns:
            float: Entity-level F1 score
        """
        # Extract entities from BIO tags and compare complete entities
        # rather than individual tokens (stricter evaluation)
        
        def extract_entities(labels, length):
            """Extract entities from BIO-tagged sequence."""
            entities = []
            current_entity = None
            
            for i in range(length):
                label = self.id_to_slot.get(labels[i], 'O')
                
                if label.startswith('B-'):
                    # Start of new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {'type': label[2:], 'start': i, 'end': i}
                elif label.startswith('I-') and current_entity:
                    # Continue current entity
                    if label[2:] == current_entity['type']:
                        current_entity['end'] = i
                    else:
                        # Type mismatch, end current entity
                        entities.append(current_entity)
                        current_entity = None
                else:
                    # O label or invalid I- tag
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            # Add last entity if exists
            if current_entity:
                entities.append(current_entity)
                
            return {(e['start'], e['end'], e['type']) for e in entities}
        
        all_true_entities = set()
        all_pred_entities = set()
        
        for true_seq, pred_seq, length in zip(y_true, y_pred, lengths):
            true_entities = extract_entities(true_seq, length)
            pred_entities = extract_entities(pred_seq, length)
            
            all_true_entities.update(true_entities)
            all_pred_entities.update(pred_entities)
        
        # Calculate entity-level metrics
        tp = len(all_true_entities & all_pred_entities)
        fp = len(all_pred_entities - all_true_entities)
        fn = len(all_true_entities - all_pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def detailed_intent_report(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """Generate detailed classification report for intents."""
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Use sklearn's classification_report for detailed per-class metrics
        
        # Get intent names
        target_names = [self.id_to_intent[i] for i in sorted(self.intent_vocab.values())]
        
        report = classification_report(y_true, y_pred, 
                                     target_names=target_names, 
                                     output_dict=True,
                                     zero_division=0)
        
        return report
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                             title: str = "Intent Confusion Matrix"):
        """Plot confusion matrix for intent detection."""
        from sklearn.metrics import confusion_matrix
        
        # Create and plot confusion matrix using seaborn
        
        # Get intent names for labels
        intent_names = [self.id_to_intent[i] for i in sorted(self.intent_vocab.values())]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=intent_names, 
                   yticklabels=intent_names,
                   cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return plt.gcf()
    
    def evaluate_model(self, y_true_intents: List[int], y_pred_intents: List[int],
                      y_true_slots: List[List[int]], y_pred_slots: List[List[int]],
                      lengths: List[int], verbose: bool = True) -> Dict:
        """
        Complete evaluation of both intent detection and slot filling.
        
        Returns:
            Dict: Complete evaluation results
        """
        results = {}
        
        # Intent detection metrics
        intent_acc = self.intent_accuracy(y_true_intents, y_pred_intents)
        results['intent_accuracy'] = intent_acc
        
        # Slot filling metrics
        slot_metrics = self.slot_metrics(y_true_slots, y_pred_slots, lengths)
        results.update({f'slot_{k}': v for k, v in slot_metrics.items()})
        
        # Entity-level F1
        entity_f1 = self.entity_level_f1(y_true_slots, y_pred_slots, lengths)
        results['entity_f1'] = entity_f1
        
        if verbose:
            print("=== Evaluation Results ===")
            print(f"Intent Accuracy: {intent_acc:.4f}")
            print(f"Slot Precision:  {slot_metrics['precision']:.4f}")
            print(f"Slot Recall:     {slot_metrics['recall']:.4f}")
            print(f"Slot F1:         {slot_metrics['f1']:.4f}")
            print(f"Entity F1:       {entity_f1:.4f}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Example usage with dummy data for testing
    
    # Dummy vocabularies for testing
    slot_vocab = {"<PAD>": 0, "O": 1, "B-fromloc.city_name": 2, "I-fromloc.city_name": 3}
    intent_vocab = {"atis_flight": 0, "atis_airfare": 1}
    
    evaluator = SLUEvaluator(slot_vocab, intent_vocab)
    
    # Dummy predictions for testing
    y_true_intents = [0, 1, 0, 1]
    y_pred_intents = [0, 1, 1, 1]  # One wrong prediction
    
    y_true_slots = [[1, 1, 2, 3, 0], [1, 2, 1, 0, 0]]
    y_pred_slots = [[1, 1, 2, 3, 0], [1, 2, 2, 0, 0]]  # One wrong prediction
    lengths = [4, 3]
    
    # Test evaluation
    results = evaluator.evaluate_model(
        y_true_intents, y_pred_intents,
        y_true_slots, y_pred_slots, 
        lengths, verbose=True
    )
    
    print(f"\nComplete results: {results}")