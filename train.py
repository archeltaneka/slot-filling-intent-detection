from collections import Counter, defaultdict
import itertools
from difflib import SequenceMatcher
import logging
import os
import re
import requests
import string
from typing import Union, Any, List, Dict
import zipfile

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import make_scorer, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig

from src.data.loader import SLUDataLoader
from src.data.splitter import SLUDataSplitter
from src.data.feature_engineer import SLUFeatureEngineer
from src.data.builder import SLUDataBuilder
from src.model.baseline import BaselineModel
from src.evaluation import *

import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    DATA_DIR = "./data"
    TEST_SIZE = 0.2
    RANDOM_STATE = 1

    # Create a DataFrame to store the training data
    loader = SLUDataLoader(data_dir=DATA_DIR)
    df = loader.load_all_data()
    
    # Split into train and test sets
    splitter = SLUDataSplitter(df=df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df, val_df = splitter.split_data()
    
    # Build intent dataset
    feature_engineer = SLUFeatureEngineer(df, train_df, val_df)
    X_train_intent, y_train_intent, X_val_intent, y_val_intent, intent_encoder, slot_label_to_id, id_to_slot_label = feature_engineer.engineer_features()
    
    # Build slot dataset
    data_builder = SLUDataBuilder(train_df, val_df)
    X_train_slot, y_train_slot, tokens_train_slot = data_builder.build_crf_dataset(train_df)
    X_val_slot, y_val_slot, tokens_val_slot = data_builder.build_crf_dataset(val_df)

    # Train a baseline model (CRF Slot Filling + RF Intent)
    intent_to_id = {label: idx for idx, label in enumerate(intent_encoder.classes_)}
    evaluator = SLUEvaluator(slot_vocab=slot_label_to_id, intent_vocab=intent_to_id)
    baseline_model = BaselineModel(slot_label_to_id, intent_encoder)
    baseline_model.train(X_train_intent, X_train_slot, y_train_intent, y_train_slot)
    baseline_results = baseline_model.evaluate(evaluator, X_val_intent, y_val_intent, X_val_slot, y_val_slot)

    
