import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.data.loader import SLUDataLoader
from src.data.splitter import SLUDataSplitter
from src.data.feature_engineer import SLUFeatureEngineer
from src.data.builder import SLUDataBuilder
from src.data.data_utils import SLUDataset, BERTDataset, build_vocab, get_collate_fn, bert_collate_fn, save_vocab
from src.model.baseline import BaselineModel
from src.model.models import JointBiLSTM, JointBiLSTMAttn, JointBERTModel
from src.model.trainer import JointTrainer, BERTTrainer
from src.model.model_utils import load_bert_tokenizer, load_embeddings, download_glove
from src.pipeline import DataPipeline
from src.train_utils import run_training_loop
from src.evaluation import SLUEvaluator
from src.utils import load_config_file, save_model

import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == '__main__':
    ### CONFIGURATION ###
    DATA_DIR, CONFIG_PATH, SAVE_DIR = "./data", "./config.yaml", 'files/checkpoints'
    config = load_config_file(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### DATA PREPARATION ###
    pipeline = DataPipeline(config, DATA_DIR, SAVE_DIR)
    train_df, val_df, word_to_id, full_slot_mapping, intent_to_id, feat_outputs = pipeline.prepare_data()
    # Build intent dataset
    X_train_intent, y_train_intent, X_val_intent, y_val_intent, intent_encoder, slot_label_to_id, id_to_slot_label, vectorizer = feat_outputs
    intent_to_id = {label: idx for idx, label in enumerate(intent_encoder.classes_)}
    # Build slot dataset
    X_train_slot, y_train_slot, X_val_slot, y_val_slot = pipeline.build_slot_dataset(train_df, val_df)
    # Evaluator
    evaluator = SLUEvaluator(slot_vocab=full_slot_mapping, intent_vocab=intent_to_id)

    ### MODEL TRAINING: BASELINE ###
    # Train a baseline model (CRF Slot Filling + RF Intent)
    baseline_model = BaselineModel(slot_label_to_id, intent_encoder)
    baseline_model.train(X_train_intent, X_train_slot, y_train_intent, y_train_slot)
    baseline_results = baseline_model.evaluate(evaluator, X_val_intent, y_val_intent, X_val_slot, y_val_slot)
    baseline_model.vectorizer = vectorizer  # Save vectorizer for inference
    baseline_model.save(SAVE_DIR + '/baseline_model.joblib')

    ### MODEL TRAINING: JOINT BiLSTM ###
    # Load embeddings
    glove_path = download_glove()
    embed_matrix = load_embeddings(glove_path, word_to_id)
    # Dataloaders
    collate_fn = get_collate_fn(word_to_id['<PAD>'], full_slot_mapping['<PAD>'])
    train_loader = DataLoader(SLUDataset(train_df, word_to_id, full_slot_mapping, intent_to_id), 
                              batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SLUDataset(val_df, word_to_id, full_slot_mapping, intent_to_id), 
                            batch_size=config['batch_size'], collate_fn=collate_fn)
    # Define losses (intent + slot loss) and device (GPU/CPU)
    intent_criterion = nn.CrossEntropyLoss()
    slot_criterion = nn.CrossEntropyLoss(ignore_index=0)
    evaluator = SLUEvaluator(slot_vocab=full_slot_mapping, intent_vocab=intent_to_id)
    # Train a joint model (BiLSTM + two heads)
    jointbilstm_model = JointBiLSTM(
        vocab_size=len(word_to_id),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_slots=len(full_slot_mapping),
        num_intents=len(intent_to_id),
        embedding_matrix=embed_matrix,
        dropout=config['dropout'],
        pad_idx=word_to_id['<PAD>']
    ).to(device)
    optimizer = optim.Adam(jointbilstm_model.parameters(), lr=config['learning_rate'])
    trainer = JointTrainer(jointbilstm_model, optimizer, slot_criterion, intent_criterion, device, evaluator)
    run_training_loop(jointbilstm_model, trainer, train_loader, val_loader, 
                      config['num_epochs'], "jointbilstm", SAVE_DIR)

    ### MODEL TRAINING: JOINT BiLSTM WITH ATTENTION ###
    # Train a joint model (BiLSTM + two heads) with attention
    jointbilstm_attn_model = JointBiLSTMAttn(
        vocab_size=len(word_to_id),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_slots=len(full_slot_mapping),
        num_intents=len(intent_to_id),
        embedding_matrix=embed_matrix,
        dropout=config['dropout'],
        pad_idx=word_to_id['<PAD>']
    ).to(device)
    optimizer = optim.Adam(jointbilstm_attn_model.parameters(), lr=config['learning_rate'])
    trainer = JointTrainer(jointbilstm_attn_model, optimizer, slot_criterion, intent_criterion, device, evaluator)
    run_training_loop(jointbilstm_attn_model, trainer, train_loader, val_loader, 
                      config['num_epochs'], "jointbilstm_attn", SAVE_DIR)

    ### MODEL TRAINING: JOINT BERT ###
    # Create BERTDataset
    tokenizer = load_bert_tokenizer(config['bert_model_name'])
    bert_train_dataset = BERTDataset(train_df, tokenizer, full_slot_mapping, intent_to_id, max_length=config['bert_max_len'])
    bert_val_dataset = BERTDataset(val_df, tokenizer, full_slot_mapping, intent_to_id, max_length=config['bert_max_len'])
    # Create BERT data loaders
    bert_train_loader = DataLoader(
        bert_train_dataset, 
        batch_size=config['bert_batch_size'],  # Smaller batch size for BERT
        shuffle=True, 
        collate_fn=bert_collate_fn
    )
    bert_val_loader = DataLoader(
        bert_val_dataset, 
        batch_size=config['bert_batch_size'], 
        shuffle=False, 
        collate_fn=bert_collate_fn
    )
    # Train a BERT-based model
    slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore special tokens
    intent_criterion = nn.CrossEntropyLoss()
    bert_model = JointBERTModel(
        bert_model_name=config['bert_model_name'],
        num_intents=len(intent_to_id),
        num_slots=len(full_slot_mapping),
        dropout=config['dropout']
    ).to(device)
    optimizer = torch.optim.AdamW(
        bert_model.parameters(),
        lr=config['bert_learning_rate'],
        weight_decay=config['bert_weight_decay']
    )    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=config['bert_scheduler_start_factor'], 
        end_factor=config['bert_scheduler_end_factor'], 
        total_iters=config['bert_num_epochs']
    )
    trainer = BERTTrainer(bert_model, optimizer, scheduler, slot_criterion, intent_criterion, device, evaluator)
    run_training_loop(bert_model, trainer, bert_train_loader, bert_val_loader, 
                      config['bert_num_epochs'], "jointbert", SAVE_DIR)
