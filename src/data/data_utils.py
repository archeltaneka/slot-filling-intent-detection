import json
import os
import requests
import zipfile
import logging

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


class SLUDataset(Dataset):
    def __init__(self, df, word_to_id, slot_to_id, intent_to_id):
        self.samples = []
        for _, row in df.iterrows():
            tokens = [w.lower() for w in row['words']]
            self.samples.append((tokens, row['slots'], row['intent']))

        self.word_to_id = word_to_id # Use the passed-in vocab
        self.slot_to_id = slot_to_id
        self.intent_to_id = intent_to_id
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens, slots, intent = self.samples[idx]
        word_ids = [self.word_to_id.get(t, self.word_to_id[UNK_TOKEN]) for t in tokens]
        slot_ids = [self.slot_to_id[s] for s in slots]
        intent_id = self.intent_to_id[intent]
        return torch.tensor(word_ids), torch.tensor(slot_ids), torch.tensor(intent_id), len(word_ids)


class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, slot_to_id, intent_to_id, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.slot_to_id = slot_to_id
        self.intent_to_id = intent_to_id
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        words, slots, intent = sample['words'], sample['slots'], sample['intent']
        
        encoding = self.tokenizer(
            ' '.join(words),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Alignment logic: Map BERT subwords to our original slot labels
        slot_labels = []
        word_idx = 0
        for token_id in encoding['input_ids'][0]:
            if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                slot_labels.append(-100) # Standard PyTorch ignore index
            else:
                if word_idx < len(slots):
                    slot_labels.append(self.slot_to_id[slots[word_idx]])
                    # If this isn't a subword (starts with ##), move to next original word
                    token_text = self.tokenizer.convert_ids_to_tokens(token_id.item())
                    if not token_text.startswith('##'):
                        word_idx += 1
                else:
                    slot_labels.append(self.slot_to_id.get('O', 0))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'slot_labels': torch.tensor(slot_labels, dtype=torch.long),
            'intent_label': self.intent_to_id[intent],
            'original_length': len(words)
        }


def build_vocab(df):
    word_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for words in df['words']:
        for w in words:
            wl = w.lower()
            if wl not in word_to_id:
                word_to_id[wl] = len(word_to_id)
    id_to_word = {i: w for w, i in word_to_id.items()}

    return word_to_id, id_to_word

def get_collate_fn(word_pad_idx, slot_pad_idx):
    def slucollate(batch):
        word_seqs, slot_seqs, intent_ids, lengths = zip(*batch)
        max_len = max(lengths)
        
        padded_words = torch.full((len(batch), max_len), word_pad_idx, dtype=torch.long)
        padded_slots = torch.full((len(batch), max_len), slot_pad_idx, dtype=torch.long)
        
        for i, (w, s, l) in enumerate(zip(word_seqs, slot_seqs, lengths)):
            padded_words[i, :l] = w
            padded_slots[i, :l] = s
            
        return padded_words, padded_slots, torch.stack(intent_ids), torch.tensor(lengths, dtype=torch.long)
    return slucollate

def bert_collate_fn(batch):
    return (
        torch.stack([item['input_ids'] for item in batch]),
        torch.stack([item['attention_mask'] for item in batch]),
        torch.stack([item['slot_labels'] for item in batch]),
        torch.tensor([item['intent_label'] for item in batch], dtype=torch.long),
        [item['original_length'] for item in batch]
    )

def save_vocab(vocab, path):
    with open(path, 'w') as f:
        json.dump(vocab, f)