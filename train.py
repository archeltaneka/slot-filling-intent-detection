import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.data.loader import SLUDataLoader
from src.data.splitter import SLUDataSplitter
from src.data.feature_engineer import SLUFeatureEngineer
from src.data.builder import SLUDataBuilder
from src.data.data_utils import SLUDataset, BERTDataset, build_vocab, get_collate_fn, bert_collate_fn
from src.model.baseline import BaselineModel
from src.model.models import JointBiLSTM, JointBiLSTMAttn, JointBERTModel
from src.model.trainer import JointTrainer, BERTTrainer
from src.model.model_utils import load_bert_tokenizer, load_embeddings, download_glove
from src.evaluation import SLUEvaluator
from src.utils import load_config_file, save_model

import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == '__main__':
    DATA_DIR = "./data"
    CONFIG_PATH = "./config.yaml"
    SAVE_DIR = 'files/checkpoints'
    config = load_config_file(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a DataFrame to store the training data
    loader = SLUDataLoader(data_dir=DATA_DIR)
    df = loader.load_all_data()
    
    # Split into train and test sets
    splitter = SLUDataSplitter(df=df, test_size=config['test_size'], random_state=config['random_state'])
    train_df, val_df = splitter.split_data()
    
    # Build intent dataset
    feature_engineer = SLUFeatureEngineer(df, train_df, val_df)
    X_train_intent, y_train_intent, X_val_intent, y_val_intent, intent_encoder, slot_label_to_id, id_to_slot_label = feature_engineer.engineer_features()
    
    # Build slot dataset
    data_builder = SLUDataBuilder(train_df, val_df)
    X_train_slot, y_train_slot, tokens_train_slot = data_builder.build_crf_dataset(train_df)
    X_val_slot, y_val_slot, tokens_val_slot = data_builder.build_crf_dataset(val_df)
    intent_to_id = {label: idx for idx, label in enumerate(intent_encoder.classes_)}

    # # Train a baseline model (CRF Slot Filling + RF Intent)
    # evaluator = SLUEvaluator(slot_vocab=slot_label_to_id, intent_vocab=intent_to_id)
    # baseline_model = BaselineModel(slot_label_to_id, intent_encoder)
    # baseline_model.train(X_train_intent, X_train_slot, y_train_intent, y_train_slot)
    # baseline_results = baseline_model.evaluate(evaluator, X_val_intent, y_val_intent, X_val_slot, y_val_slot)
    # baseline_model.save(SAVE_DIR + '/baseline_model.joblib')

    # Create a dataloader
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    word_to_id, id_to_word = build_vocab(train_df) # Build vocabulary
    full_slot_mapping = {PAD_TOKEN: 0, **{k: v+1 for k, v in slot_label_to_id.items()}}
    train_ds = SLUDataset(train_df, word_to_id, full_slot_mapping, intent_to_id)
    val_ds = SLUDataset(val_df, word_to_id, full_slot_mapping, intent_to_id)
    collate_fn = get_collate_fn(word_to_id[PAD_TOKEN], full_slot_mapping[PAD_TOKEN])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    logging.info(f"Vocab size: {len(word_to_id)}, Slot labels: {len(full_slot_mapping)}, Intents: {len(intent_to_id)}")

    # # Load embeddings
    # glove_path = download_glove()
    # embed_matrix = load_embeddings(glove_path, word_to_id)    

    # # Define losses (intent + slot loss) and device (GPU/CPU)
    # intent_criterion = nn.CrossEntropyLoss()
    # slot_criterion = nn.CrossEntropyLoss(ignore_index=full_slot_mapping['<PAD>'])
    evaluator = SLUEvaluator(slot_vocab=full_slot_mapping, intent_vocab=intent_to_id)

    # # Train a joint model (BiLSTM + two heads)
    # jointbilstm_model = JointBiLSTM(
    #     vocab_size=len(word_to_id),
    #     embed_dim=config['embed_dim'],
    #     hidden_dim=config['hidden_dim'],
    #     num_layers=config['num_layers'],
    #     num_slots=len(full_slot_mapping),
    #     num_intents=len(intent_to_id),
    #     embedding_matrix=embed_matrix,
    #     dropout=config['dropout'],
    #     pad_idx=word_to_id[PAD_TOKEN]
    # ).to(device)
    # optimizer = optim.Adam(jointbilstm_model.parameters(), lr=config['learning_rate'])
    # trainer = JointTrainer(jointbilstm_model, optimizer, slot_criterion, intent_criterion, device, evaluator)
    # logging.info("Training JointBiLSTM model...")
    # for epoch in range(config['num_epochs']):
    #     train_loss = trainer.train_epoch(train_loader)
    #     jointbilstm_results = trainer.evaluate(val_loader)
    #     logging.info(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss = {train_loss:.4f}, Val Intent Acc: {jointbilstm_results['intent_accuracy']:.4f}, Val Slot F1: {jointbilstm_results['slot_f1']:.4f}, Entity F1: {jointbilstm_results['entity_f1']:.4f}")
    # logging.info("Final validation evaluation (JointBiLSTM model):")
    # logging.info(f"Intent accuracy: {jointbilstm_results['intent_accuracy']:.4f}")
    # logging.info(f"Slot F1: {jointbilstm_results['slot_f1']:.4f} | Entity F1: {jointbilstm_results['entity_f1']:.4f}")
    # save_model(jointbilstm_model, SAVE_DIR, 'jointbilstm_model.pth')
    # logging.info(f"JointBiLSTM model saved to {SAVE_DIR}")

    # # Train a joint model (BiLSTM + two heads) with attention
    # jointbilstm_attn_model = JointBiLSTMAttn(
    #     vocab_size=len(word_to_id),
    #     embed_dim=config['embed_dim'],
    #     hidden_dim=config['hidden_dim'],
    #     num_layers=config['num_layers'],
    #     num_slots=len(full_slot_mapping),
    #     num_intents=len(intent_to_id),
    #     embedding_matrix=embed_matrix,
    #     dropout=config['dropout'],
    #     pad_idx=word_to_id[PAD_TOKEN]
    # ).to(device)
    # optimizer = optim.Adam(jointbilstm_attn_model.parameters(), lr=config['learning_rate'])
    # trainer = JointTrainer(jointbilstm_attn_model, optimizer, slot_criterion, intent_criterion, device, evaluator)
    # logging.info("Training JointBiLSTM with attention model...")
    # for epoch in range(config['num_epochs']):
    #     train_loss = trainer.train_epoch(train_loader)
    #     jointbilstm_attn_results = trainer.evaluate(val_loader)
    #     logging.info(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss = {train_loss:.4f}, Val Intent Acc: {jointbilstm_attn_results['intent_accuracy']:.4f}, Val Slot F1: {jointbilstm_attn_results['slot_f1']:.4f}, Entity F1: {jointbilstm_attn_results['entity_f1']:.4f}")
    # logging.info("Final validation evaluation (JointBiLSTM with attention model):")
    # logging.info(f"Intent accuracy: {jointbilstm_attn_results['intent_accuracy']:.4f}")
    # logging.info(f"Slot F1: {jointbilstm_attn_results['slot_f1']:.4f} | Entity F1: {jointbilstm_attn_results['entity_f1']:.4f}")
    # save_model(jointbilstm_attn_model, SAVE_DIR, 'jointbilstm_attn_model.pth')
    # logging.info(f"JointBiLSTM with attention model saved to {SAVE_DIR}")

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
    logging.info("Training JointBERT model...")
    for epoch in range(config['bert_num_epochs']):
        train_loss = trainer.train_epoch(bert_train_loader)
        bert_results = trainer.evaluate(bert_val_loader)
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss = {train_loss:.4f}, Val Intent Acc: {bert_results['intent_accuracy']:.4f}, Val Slot F1: {bert_results['slot_f1']:.4f}, Entity F1: {bert_results['entity_f1']:.4f}")
    logging.info("Final validation evaluation (BERT model):")
    logging.info(f"Intent accuracy: {bert_results['intent_accuracy']:.4f}")
    logging.info(f"Slot F1: {bert_results['slot_f1']:.4f} | Entity F1: {bert_results['entity_f1']:.4f}")
    save_model(bert_model, SAVE_DIR, 'bert_model.pth')
    logging.info(f"BERT model saved to {SAVE_DIR}")
