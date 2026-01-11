import os
import requests
import zipfile
import logging

import joblib
import torch
from transformers import BertTokenizer, BertModel, BertConfig

from src.model.models import JointBiLSTM, JointBiLSTMAttn, JointBERTModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class ModelLoader:
    def __init__(self, config, device, vocabs):
        self.config = config
        self.device = device
        self.vocabs = vocabs

    def load_baseline(self, path):
        return joblib.load(path)

    def load_bilstm(self, path, attn=False):
        model_class = JointBiLSTMAttn if attn else JointBiLSTM
        model = model_class(
            vocab_size=len(self.vocabs['word_to_id']),
            embed_dim=self.config['embed_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_slots=len(self.vocabs['slot_to_id']),
            num_intents=len(self.vocabs['intent_to_id']),
            dropout=self.config['dropout'],
            pad_idx=self.vocabs['word_to_id'].get('<PAD>', 0)
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        return model.to(self.device).eval()

    def load_bert(self, path):
        model = JointBERTModel(
            bert_model_name=self.config['bert_model_name'],
            num_intents=len(self.vocabs['intent_to_id']),
            num_slots=len(self.vocabs['slot_to_id']),
            dropout=self.config['dropout']
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        return model.to(self.device).eval()

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

def load_bert_tokenizer(model_name):
    try:
        return BertTokenizer.from_pretrained(model_name)
    except Exception as e:
        logging.warning(f"Failed to load {model_name}, falling back to distilbert.")
        return BertTokenizer.from_pretrained('distilbert-base-uncased')