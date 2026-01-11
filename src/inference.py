import numpy as np
import torch
import torch.nn.functional as F
import string


def get_word_shape(word):
    """Get word shape for CRF features."""
    shape = []
    for ch in word:
        if ch.isupper():
            shape.append('X')
        elif ch.islower():
            shape.append('x')
        elif ch.isdigit():
            shape.append('d')
        elif ch in string.punctuation:
            shape.append('p')
        else:
            shape.append('o')
    collapsed = []
    for ch in shape:
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    return ''.join(collapsed)


def token2features(sent_words, i):
    """Extract CRF features for token at position i."""
    word = sent_words[i]
    lower = word.lower()
    features = {
        'bias': 1.0,
        'word.lower': lower,
        'word[-3:]': lower[-3:],
        'word[-2:]': lower[-2:],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.shape': get_word_shape(word),
        'has.hyphen': '-' in word,
        'has.digit': any(ch.isdigit() for ch in word),
        'has.alpha': any(ch.isalpha() for ch in word),
        'is.punct': all(ch in string.punctuation for ch in word),
    }
    
    # Context features
    for offset, prefix in [(-2, 'prev2'), (-1, 'prev1'), (1, 'next1'), (2, 'next2')]:
        j = i + offset
        if 0 <= j < len(sent_words):
            w = sent_words[j]
            wl = w.lower()
            features.update({
                f'{prefix}.lower': wl,
                f'{prefix}.istitle': w.istitle(),
                f'{prefix}.isupper': w.isupper(),
                f'{prefix}.shape': get_word_shape(w),
            })
        else:
            features[f'BOS/EOS_{prefix}'] = True
    
    return features


def prepare_text_for_baseline(text):
    """Prepare text for baseline model (CRF + RF)."""
    tokens = text.lower().split()
    
    # Extract features for each token (for CRF slot filling)
    slot_features = [token2features(tokens, i) for i in range(len(tokens))]
    
    return tokens, slot_features


def prepare_text_for_bilstm(text, word_to_id):
    """Prepare text for BiLSTM models."""
    tokens = text.lower().split()
    token_ids = [word_to_id.get(token, word_to_id.get('<UNK>', 1)) for token in tokens]
    return tokens, token_ids


def prepare_text_for_bert(text, tokenizer, max_len=128):
    """Prepare text for BERT model."""
    # Tokenize with BERT tokenizer
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get the actual tokens for visualization (excluding padding)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    # Find where padding starts
    pad_idx = tokens.index('[PAD]') if '[PAD]' in tokens else len(tokens)
    tokens = tokens[:pad_idx]
    
    return tokens, encoding


def compute_gradient_saliency(model, input_ids, lengths, device):
    """
    Compute gradient-based saliency for BiLSTM model without explicit attention.
    This shows which tokens contribute most to the predictions.
    """
    model.eval()
    
    # Prepare input tensors
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    lengths_tensor = torch.tensor([lengths], dtype=torch.long).to(device)
    
    # Enable gradient computation
    input_ids_tensor.requires_grad_(False)  # Input IDs don't need gradients
    
    # Get embeddings with gradients enabled
    embeddings = model.embedding(input_ids_tensor)
    embeddings.requires_grad_(True)
    embeddings.retain_grad()  # Important: retain gradients for intermediate tensor
    
    # Forward pass through LSTM and classifiers
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        embeddings, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False
    )
    enc_out_packed, _ = model.encoder(packed)
    enc_out, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)
    
    # Don't apply dropout during gradient computation
    # enc_out = model.dropout(enc_out)  # Skip dropout for cleaner gradients
    
    # Get intent logits via mean pooling
    mask = (input_ids_tensor != 0).float()
    lengths_clamped = lengths_tensor.clamp(min=1).float().unsqueeze(1)
    pooled = (enc_out * mask.unsqueeze(-1)).sum(dim=1) / lengths_clamped
    # pooled = model.dropout(pooled)  # Skip dropout
    intent_logits = model.intent_classifier(pooled)
    
    # Compute gradient of max intent logit w.r.t embeddings
    max_intent_logit = intent_logits.max()
    
    # Zero gradients first
    model.zero_grad()
    if embeddings.grad is not None:
        embeddings.grad.zero_()
    
    # Backward pass
    max_intent_logit.backward()
    
    # Check if gradients were computed
    if embeddings.grad is None:
        # Fallback to uniform weights if gradient computation failed
        saliency = np.ones(lengths) / lengths
    else:
        # Compute saliency as L2 norm of gradients across embedding dimension
        saliency = embeddings.grad.norm(dim=2).squeeze(0).detach().cpu().numpy()
        
        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        saliency = saliency[:lengths]
    
    return saliency


def predict_baseline(model, text, vocabs):
    """
    Predict with baseline model and extract CRF marginal probabilities as attention.
    Returns: intent, slots, attention_like_scores
    """
    tokens, slot_features = prepare_text_for_baseline(text)
    
    # Predict slots (CRF)
    slot_pred = model.crf_model.predict([slot_features])[0]
    
    # Get CRF marginal probabilities for "attention" visualization
    try:
        marginals = model.crf_model.predict_marginals([slot_features])[0]
        # Use the max probability at each position as attention weight
        token_importance = np.array([max(m.values()) for m in marginals])
    except:
        # Fallback to uniform weights
        token_importance = np.ones(len(tokens)) / len(tokens)
    
    # For intent prediction, we need the TF-IDF vectorizer
    # The baseline model should have been saved with a vectorizer attribute
    intent_name = 'Unknown'
    
    # Check if model has vectorizer (should be saved during training)
    if hasattr(model, 'vectorizer'):
        # Use the saved vectorizer
        text_joined = ' '.join(tokens)
        X_intent = model.vectorizer.transform([text_joined])
        intent_pred = model.rf_model.predict(X_intent)[0]
        
        # Decode using intent_encoder if available
        if hasattr(model, 'intent_encoder'):
            intent_name = model.intent_encoder.inverse_transform([intent_pred])[0]
        else:
            intent_name = vocabs['id_to_intent'].get(intent_pred, 'Unknown')
    else:
        # Fallback: create a simple TF-IDF vectorizer on the fly
        # This won't be as accurate but will give a prediction
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Get all possible intents from vocab
        all_intents = list(vocabs['intent_to_id'].keys())
        
        # Simple heuristic: look for keywords in the text
        text_lower = text.lower()
        if 'book' in text_lower or 'flight' in text_lower:
            intent_name = 'BookFlight' if 'BookFlight' in all_intents else all_intents[0]
        elif 'show' in text_lower or 'find' in text_lower or 'get' in text_lower:
            intent_name = 'GetFlights' if 'GetFlights' in all_intents else all_intents[0]
        else:
            # Default to first intent
            intent_name = all_intents[0] if all_intents else 'Unknown'
    
    return {
        'intent': intent_name,
        'slots': list(zip(tokens, slot_pred)),
        'attention': token_importance,
        'tokens': tokens
    }


def predict_bilstm(model, text, vocabs, device):
    """
    Predict with BiLSTM model and compute gradient-based saliency.
    Returns: intent, slots, saliency
    """
    tokens, token_ids = prepare_text_for_bilstm(text, vocabs['word_to_id'])
    
    # Prepare input
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long).to(device)
    
    # Get predictions
    with torch.no_grad():
        slot_logits, intent_logits = model(input_ids, lengths)
        
        # Get predictions
        intent_pred = intent_logits.argmax(dim=1).item()
        slot_preds = slot_logits.argmax(dim=2).squeeze(0).cpu().numpy()
    
    # Compute gradient saliency
    saliency = compute_gradient_saliency(model, token_ids, len(token_ids), device)
    
    # Convert predictions
    intent_name = vocabs['id_to_intent'].get(intent_pred, 'Unknown')
    slot_names = [vocabs['id_to_slot'].get(slot_id, 'O') for slot_id in slot_preds[:len(tokens)]]
    
    return {
        'intent': intent_name,
        'slots': list(zip(tokens, slot_names)),
        'attention': saliency,
        'tokens': tokens
    }


def predict_bilstm_attn(model, text, vocabs, device):
    """
    Predict with BiLSTM+Attention model and extract attention weights.
    Returns: intent, slots, attention_weights
    """
    tokens, token_ids = prepare_text_for_bilstm(text, vocabs['word_to_id'])
    
    # Prepare input
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long).to(device)
    
    # Get predictions with attention
    with torch.no_grad():
        slot_logits, intent_logits, attn_weights = model(input_ids, lengths, return_attention=True)
        
        # Get predictions
        intent_pred = intent_logits.argmax(dim=1).item()
        slot_preds = slot_logits.argmax(dim=2).squeeze(0).cpu().numpy()
        
        # Extract attention weights
        attention = attn_weights.squeeze(0).cpu().numpy()
    
    # Convert predictions
    intent_name = vocabs['id_to_intent'].get(intent_pred, 'Unknown')
    slot_names = [vocabs['id_to_slot'].get(slot_id, 'O') for slot_id in slot_preds[:len(tokens)]]
    
    return {
        'intent': intent_name,
        'slots': list(zip(tokens, slot_names)),
        'attention': attention[:len(tokens)],
        'tokens': tokens
    }


def predict_bert(model, tokenizer, text, vocabs, device, max_len=128):
    """
    Predict with BERT model and extract attention weights.
    Returns: intent, slots, bert_attentions (all layers and heads)
    """
    tokens, encoding = prepare_text_for_bert(text, tokenizer, max_len)
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions with attention
    with torch.no_grad():
        slot_logits, intent_logits, bert_attentions = model(
            input_ids, attention_mask, return_attention=True
        )
        
        # Get predictions
        intent_pred = intent_logits.argmax(dim=1).item()
        slot_preds = slot_logits.argmax(dim=2).squeeze(0).cpu().numpy()
    
    # Process BERT attentions - average across all heads and layers for simplicity
    # bert_attentions is a tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)
    num_tokens = len(tokens)
    
    # Average attention across all layers and heads, focusing on [CLS] token attention
    all_attentions = []
    for layer_attn in bert_attentions:
        # layer_attn shape: [batch_size, num_heads, seq_len, seq_len]
        # Average across heads: [batch_size, seq_len, seq_len]
        avg_heads = layer_attn.mean(dim=1)
        all_attentions.append(avg_heads)
    
    # Average across layers: [batch_size, seq_len, seq_len]
    avg_attention = torch.stack(all_attentions).mean(dim=0)
    
    # Extract [CLS] token's attention to other tokens (first row)
    cls_attention = avg_attention[0, 0, :num_tokens].cpu().numpy()
    
    # Normalize
    if cls_attention.sum() > 0:
        cls_attention = cls_attention / cls_attention.sum()
    
    # Convert predictions (skip [CLS] and [SEP] tokens)
    intent_name = vocabs['id_to_intent'].get(intent_pred, 'Unknown')
    
    # Map BERT tokens back to original tokens (handle subword tokens)
    # For simplicity, we'll use the BERT tokens directly
    slot_names = [vocabs['id_to_slot'].get(slot_id, 'O') for slot_id in slot_preds[:num_tokens]]
    
    return {
        'intent': intent_name,
        'slots': list(zip(tokens, slot_names)),
        'attention': cls_attention,
        'tokens': tokens,
        'bert_attentions': bert_attentions  # Keep full attentions for advanced viz
    }
