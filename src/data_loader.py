#!/usr/bin/env python3
"""
Data Loader Template for Slot Filling and Intent Detection

This template provides a complete framework for loading and preprocessing airline dialogue dataset.
All functions are fully implemented and ready to use.

Data format: word:slot_label word:slot_label ... <=> intent_label
Example: i:O want:O to:O fly:O from:O boston:B-fromloc.city_name to:O denver:B-toloc.city_name <=> atis_flight
"""

from collections import Counter, defaultdict
import logging
import os
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SLUDataLoader:
    """
    Data loader for airline dialogue dataset with slot filling and intent detection tasks.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = data_dir
        
        # Vocabularies and mappings
        self.word_vocab = {}  # word -> id
        self.slot_vocab = {}  # slot_label -> id  
        self.intent_vocab = {} # intent -> id
        
        # Reverse mappings
        self.id_to_word = {}
        self.id_to_slot = {}  
        self.id_to_intent = {}
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.PAD_ID = 0
        self.UNK_ID = 1
        
        # Dataset containers
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        
    def parse_line(self, line: str) -> Tuple[List[str], List[str], str]:
        """
        Parse a single line from the dataset.
        
        Args:
            line (str): Input line in format "word:slot word:slot ... <=> intent"
            
        Returns:
            Tuple[List[str], List[str], str]: (words, slot_labels, intent)
            
        Example:
            Input: "i:O want:O to:O fly:O <=> atis_flight"
            Output: (["i", "want", "to", "fly"], ["O", "O", "O", "O"], "atis_flight")
        """
        # Split by " <=> " first to separate input and intent
        # Then split input by spaces and extract word:slot pairs
        
        if " <=> " not in line:
            raise ValueError(f"Invalid line format: {line}")
            
        input_part, intent = line.strip().split(" <=> ")
        
        words = []
        slot_labels = []
        
        # Parse input_part to extract words and slot labels
        # Split by spaces, then split each token by ":" to get word and slot
        tokens = input_part.split()
        for token in tokens:
            if ':' in token:
                word, slot = token.split(':', 1)  # Split only on first ':'
                words.append(word)
                slot_labels.append(slot)
            else:
                # Handle cases where there might be no slot label (rare edge case)
                words.append(token)
                slot_labels.append("O")
        
        return words, slot_labels, intent
    
    def load_data_file(self, filename: str) -> List[Tuple[List[str], List[str], str]]:
        """
        Load data from a file.
        
        Args:
            filename (str): Name of the file to load
            
        Returns:
            List[Tuple[List[str], List[str], str]]: List of (words, slot_labels, intent) tuples
        """
        filepath = os.path.join(self.data_dir, filename)
        data = []
        
        # Read the file line by line and parse each line
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            words, slots, intent = self.parse_line(line)
                            data.append((words, slots, intent))
                        except ValueError as e:
                            logging.info(f"Error parsing line {line_num} in {filename}: {e}")
                            continue
        except FileNotFoundError:
            logging.info(f"File not found: {filepath}")
            return []
        
        logging.info(f"Loaded {len(data)} samples from {filename}")
        return data
    
    def build_vocabularies(self, train_data: List[Tuple[List[str], List[str], str]]) -> None:
        """
        Build vocabularies from training data.
        
        Args:
            train_data: List of (words, slot_labels, intent) tuples
        """
        # Collect all unique words, slot labels, and intents
        # Create mappings from strings to IDs
        # Include special tokens (PAD, UNK)
        
        # Initialize with special tokens
        self.word_vocab = {self.PAD_TOKEN: self.PAD_ID, self.UNK_TOKEN: self.UNK_ID}
        self.slot_vocab = {self.PAD_TOKEN: self.PAD_ID}
        self.intent_vocab = {}
        
        # Collect all unique tokens
        all_words = set()
        all_slots = set()
        all_intents = set()
        
        for words, slots, intent in train_data:
            all_words.update(words)
            all_slots.update(slots)
            all_intents.add(intent)
        
        # Build word vocabulary
        for i, word in enumerate(sorted(all_words), start=len(self.word_vocab)):
            self.word_vocab[word] = i
            
        # Build slot vocabulary  
        for i, slot in enumerate(sorted(all_slots), start=len(self.slot_vocab)):
            self.slot_vocab[slot] = i
            
        # Build intent vocabulary
        for i, intent in enumerate(sorted(all_intents)):
            self.intent_vocab[intent] = i
            
        # Create reverse mappings
        self.id_to_word = {v: k for k, v in self.word_vocab.items()}
        self.id_to_slot = {v: k for k, v in self.slot_vocab.items()}
        self.id_to_intent = {v: k for k, v in self.intent_vocab.items()}
        
        logging.info(f"Vocabulary sizes - Words: {len(self.word_vocab)}, "
              f"Slots: {len(self.slot_vocab)}, Intents: {len(self.intent_vocab)}")
    
    def words_to_ids(self, words: List[str]) -> List[int]:
        """Convert words to their corresponding IDs."""
        # Convert each word to its ID, use UNK_ID for unknown words
        return [self.word_vocab.get(word, self.UNK_ID) for word in words]
    
    def slots_to_ids(self, slots: List[str]) -> List[int]:
        """Convert slot labels to their corresponding IDs."""
        return [self.slot_vocab.get(slot, 0) for slot in slots]  # 0 should be PAD_ID, but slots should always exist
    
    def intent_to_id(self, intent: str) -> int:
        """Convert intent to its corresponding ID."""
        return self.intent_vocab.get(intent, 0)  # Intent should always exist in training
    
    def pad_sequences(self, sequences: List[List[int]], max_length: Optional[int] = None) -> np.ndarray:
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of sequences to pad
            max_length: Maximum length to pad to (if None, use longest sequence)
            
        Returns:
            np.ndarray: Padded sequences
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        # Pad each sequence to max_length using PAD_ID (0)
        padded = np.zeros((len(sequences), max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]
            
        return padded
    
    def prepare_batch(self, data: List[Tuple[List[str], List[str], str]], 
                     max_length: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Prepare a batch of data for training/evaluation.
        
        Args:
            data: List of (words, slots, intent) tuples
            max_length: Maximum sequence length
            
        Returns:
            Dict containing:
                - 'words': padded word IDs
                - 'slots': padded slot label IDs  
                - 'intents': intent IDs
                - 'lengths': actual sequence lengths (before padding)
        """
        # Convert words, slots, intents to IDs
        # Pad sequences and keep track of original lengths
        
        word_ids = []
        slot_ids = []
        intent_ids = []
        lengths = []
        
        for words, slots, intent in data:
            word_ids.append(self.words_to_ids(words))
            slot_ids.append(self.slots_to_ids(slots))
            intent_ids.append(self.intent_to_id(intent))
            lengths.append(len(words))
        
        # Pad sequences
        padded_words = self.pad_sequences(word_ids, max_length)
        padded_slots = self.pad_sequences(slot_ids, max_length)
        
        return {
            'words': padded_words,
            'slots': padded_slots, 
            'intents': np.array(intent_ids),
            'lengths': np.array(lengths)
        }
    
    def load_all_data(self) -> None:
        """Load training, validation, and test data."""
        logging.info("Loading airline dialogue dataset...")
        
        # Load training data and build vocabularies
        self.train_data = self.load_data_file("train.txt")
        self.build_vocabularies(self.train_data)
        
        # Load validation and test data
        self.valid_data = self.load_data_file("valid.txt")
        
        # For test data, students will use student_test.txt (without labels)
        # But for development, we can load the full test set
        if os.path.exists(os.path.join(self.data_dir, "test.txt")):
            self.test_data = self.load_data_file("test.txt")
        
        data = []
        for line_num, row in enumerate(self.train_data):
            data.append({
                            'line_num': line_num,
                            'words': row[0],
                            'slots': row[1],
                            'intent': row[2],
                            'length': len(row[0])
                        })
        df = pd.DataFrame(data)
        logging.info(f'Loaded {len(df)} rows of data')

        return df


# Example usage and testing
if __name__ == "__main__":
    
    # Initialize data loader
    loader = SLUDataLoader()
    
    # Load all data
    loader.load_all_data()
    
    # logging.info statistics
    loader.get_data_stats()
    
    # Test batch preparation
    if loader.train_data:
        sample_batch = loader.train_data[:5]  # First 5 samples
        batch = loader.prepare_batch(sample_batch)
        
        logging.info(f"\n=== Sample Batch ===")
        logging.info(f"Word IDs shape: {batch['words'].shape}")
        logging.info(f"Slot IDs shape: {batch['slots'].shape}")  
        logging.info(f"Intent IDs shape: {batch['intents'].shape}")
        logging.info(f"Lengths: {batch['lengths']}")
        
        # Show first sample conversion
        words, slots, intent = sample_batch[0]
        logging.info(f"\nFirst sample:")
        logging.info(f"Words: {words}")
        logging.info(f"Slots: {slots}")
        logging.info(f"Intent: {intent}")
        logging.info(f"Word IDs: {batch['words'][0][:len(words)]}")
        logging.info(f"Slot IDs: {batch['slots'][0][:len(slots)]}")
        logging.info(f"Intent ID: {batch['intents'][0]}")