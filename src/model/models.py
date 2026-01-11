import logging

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Joint BiLSTM with two heads
class JointBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_intents, num_slots,
                 embedding_matrix=None, dropout=0.3, pad_idx=0):
        super().__init__()
        logging.info("Initializing a JointBiLSTM model...")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_matrix)
        
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        enc_out_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        
        # Slot head: token-level classification
        self.slot_classifier = nn.Linear(enc_out_dim, num_slots)
        
        # Intent head: utterance-level classification (use mean pooling)
        self.intent_pool = nn.AdaptiveAvgPool1d(1) # Converts token-level into sentence-level
        self.intent_classifier = nn.Linear(enc_out_dim, num_intents)
    
    def forward(self, x, lengths):
        # x: [B, T]
        mask = (x != 0).float()  # PAD assumed 0
        embeds = self.embedding(x)  # [B, T, E]
        embeds = self.dropout(embeds)
        
        # Pack padded for efficient LSTM
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        enc_out_packed, _ = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)  # [B, T, 2H]
        enc_out = self.dropout(enc_out)
        
        # Slot logits
        slot_logits = self.slot_classifier(enc_out)  # [B, T, num_slots]
        
        # Intent logits via masked mean pooling
        # transpose to [B, 2H, T] for pooling
        enc_out_t = enc_out.transpose(1, 2)
        # Avoid division by zero
        lengths_clamped = lengths.clamp(min=1).float().unsqueeze(1)  # [B, 1]
        pooled = (enc_out * mask.unsqueeze(-1)).sum(dim=1) / lengths_clamped  # [B, 2H]
        pooled = self.dropout(pooled)
        intent_logits = self.intent_classifier(pooled)  # [B, num_intents]
        
        return slot_logits, intent_logits


# BiLSTM + Attention joint model
class JointBiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_intents, num_slots,
                 embedding_matrix=None, dropout=0.3, pad_idx=0):
        super().__init__()
        logging.info("Initializing a JointBiLSTM with attention model...")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_matrix)
        
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        enc_out_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        
        # Scaled dot-product style attention for intent pooling
        self.attn_vector = nn.Parameter(torch.randn(enc_out_dim))
        
        # Slot head
        self.slot_classifier = nn.Linear(enc_out_dim, num_slots)
        
        # Intent head
        self.intent_classifier = nn.Linear(enc_out_dim, num_intents)
    
    def forward(self, x, lengths, return_attention=False):
        mask = (x != 0)  # [B, T]
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        enc_out_packed, _ = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)  # [B, T, 2H]
        enc_out = self.dropout(enc_out)
        
        # Slot logits directly from encoder outputs
        slot_logits = self.slot_classifier(enc_out)  # [B, T, num_slots]
        
        # Attention over time for intent
        # Compute scores: s_t = h_t · a  (a is attn_vector) --> dot product attention
        scores = torch.einsum('bth,h->bt', enc_out, self.attn_vector)  # [B, T]
        # Or we can also use the Bahdanau-style attention using tanh activation function
        # score = torch.tanh(self.W(enc_out))
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=1)  # [B, T]
        
        # Weighted sum
        context = torch.einsum('bth,bt->bh', enc_out, attn_weights)  # [B, 2H]
        context = self.dropout(context)
        intent_logits = self.intent_classifier(context)  # [B, num_intents]
        
        if return_attention:
            return slot_logits, intent_logits, attn_weights
        return slot_logits, intent_logits


# Joint BERT Model Class
class JointBERTModel(nn.Module):
    def __init__(self, bert_model_name, num_intents, num_slots, dropout=0.3):
        super().__init__()
        logging.info(f"Initializing JointBERT with {bert_model_name}...")
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_config = self.bert.config
        self.dropout = nn.Dropout(dropout)
        
        self.intent_classifier = nn.Linear(self.bert_config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.bert_config.hidden_size, num_slots)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.zeros_(self.intent_classifier.bias)
        nn.init.xavier_uniform_(self.slot_classifier.weight)
        nn.init.zeros_(self.slot_classifier.bias)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, return_attention=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_attentions=return_attention
        )
        
        sequence_output = self.dropout(outputs.last_hidden_state)
        # Using the [CLS] token representation for intent
        pooled_output = self.dropout(outputs.pooler_output)
        
        slot_logits = self.slot_classifier(sequence_output)
        intent_logits = self.intent_classifier(pooled_output)
        
        if return_attention:
            return slot_logits, intent_logits, outputs.attentions
        return slot_logits, intent_logits