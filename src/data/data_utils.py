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
    def __init__(self, df, word_to_id, slot_label_to_id, intent_to_id):
        self.samples = []
        for _, row in df.iterrows():
            tokens = [w.lower() for w in row['words']]
            self.samples.append((tokens, row['slots'], row['intent']))

        self.word_to_id = word_to_id # Use the passed-in vocab
        self.slot_to_id = slot_label_to_id
        self.intent_to_id = intent_to_id
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens, slots, intent = self.samples[idx]
        word_ids = [self.word_to_id.get(t, self.word_to_id[UNK_TOKEN]) for t in tokens]
        slot_ids = [self.slot_to_id[s] for s in slots]
        intent_id = self.intent_to_id[intent]
        return torch.tensor(word_ids), torch.tensor(slot_ids), torch.tensor(intent_id), len(word_ids)


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

def load_embeddings(embedding_path, word_to_id, embedding_dim=100):
    if not os.path.exists(embedding_path):
        logging.info(f"Pretrained embeddings not found at {embedding_path}. Using random init.")
        embedding_matrix = torch.randn(len(word_to_id), embedding_dim) * 0.05
        embedding_matrix[0] = torch.zeros(embedding_dim)  # PAD -> zeros
        return embedding_matrix
    
    logging.info(f"Loading pretrained embeddings from {embedding_path} ...")
    embedding_matrix = torch.randn(len(word_to_id), embedding_dim) * 0.05
    embedding_matrix[0] = torch.zeros(embedding_dim)
    hits = 0
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < embedding_dim + 1:
                continue
            token = parts[0].lower()
            vec = torch.tensor([float(x) for x in parts[1:1+embedding_dim]])
            if token in word_to_id:
                embedding_matrix[word_to_id[token]] = vec
                hits += 1
    logging.info(f"Embedding coverage: {hits}/{len(word_to_id)} = {hits/len(word_to_id)*100:.1f}%")
    return embedding_matrix

def download_glove(base_dir='files/embedding', dim=100):
    os.makedirs(base_dir, exist_ok=True)
    target_txt = os.path.join(base_dir, f'glove.6B.{dim}d.txt')
    if os.path.exists(target_txt):
        logging.info(f"Found pretrained embeddings: {target_txt}")
        return target_txt
    zip_url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    zip_path = os.path.join(base_dir, 'glove.6B.zip')
    logging.info(f"Downloading GloVe embeddings from {zip_url} ...")
    try:
        with requests.get(zip_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            member = f'glove.6B.{dim}d.txt'
            if member not in zf.namelist():
                raise FileNotFoundError(f"{member} not found in zip archive.")
            zf.extract(member, base_dir)
        logging.info("Extraction complete.")
    except Exception as e:
        logging.info(f"Failed to download GloVe: {e}. Proceeding with random init.")
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass
    return target_txt