import logging

import torch
import torch.nn as nn

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
