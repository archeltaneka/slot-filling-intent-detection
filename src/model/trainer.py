import torch
import logging
from copy import deepcopy

class JointTrainer:
    def __init__(self, model, optimizer, slot_criterion, intent_criterion, 
                 device, evaluator, alpha_slot=1.0, alpha_intent=1.0):
        self.model = model
        self.optimizer = optimizer
        self.slot_criterion = slot_criterion
        self.intent_criterion = intent_criterion
        self.device = device
        self.evaluator = evaluator
        self.alpha_slot = alpha_slot
        self.alpha_intent = alpha_intent
        
        # Point 1 Fix: Store best results so we can report them at the end
        self.best_val_f1 = -1
        self.best_state = None
        self.best_results = None 

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for words, slots, intents, lengths in loader:
            words, slots = words.to(self.device), slots.to(self.device)
            intents, lengths = intents.to(self.device), lengths.to(self.device)

            self.optimizer.zero_grad()
            slot_logits, intent_logits = self.model(words, lengths)

            B, T, C = slot_logits.shape
            slot_loss = self.slot_criterion(slot_logits.view(B*T, C), slots.view(B*T))
            intent_loss = self.intent_criterion(intent_logits, intents)

            loss = (self.alpha_slot * slot_loss) + (self.alpha_intent * intent_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        metrics = {'true_s': [], 'pred_s': [], 'true_i': [], 'pred_i': [], 'lens': []}
        
        with torch.no_grad():
            for words, slots, intents, lengths in loader:
                words, slots = words.to(self.device), slots.to(self.device)
                intents, lengths = intents.to(self.device), lengths.to(self.device)
                
                slot_logits, intent_logits = self.model(words, lengths)

                metrics['pred_s'].extend(slot_logits.argmax(dim=-1).cpu().tolist())
                metrics['true_s'].extend(slots.cpu().tolist())
                metrics['pred_i'].extend(intent_logits.argmax(dim=-1).cpu().tolist())
                metrics['true_i'].extend(intents.cpu().tolist())
                metrics['lens'].extend(lengths.cpu().tolist())

        # Point 2 Fix: Pass data directly to evaluator. 
        # If your SLUEvaluator is reused, we ensure we pass fresh data lists.
        results = self.evaluator.evaluate_model(
            y_true_intents=metrics['true_i'],
            y_pred_intents=metrics['pred_i'],
            y_true_slots=metrics['true_s'],
            y_pred_slots=metrics['pred_s'],
            lengths=metrics['lens'],
            verbose=False
        )
        return results

    def save_best_model(self, current_results):
        """
        Expects the full results dictionary from evaluate().
        """
        current_f1 = current_results['slot_f1']
        if current_f1 > self.best_val_f1:
            self.best_val_f1 = current_f1
            # Deep copy the state dict to ensure it's a snapshot
            self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            # Store the full results dict of the best epoch
            self.best_results = deepcopy(current_results)
            return True
        return False