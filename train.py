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

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig

from src.data_loader import SLUDataLoader
from src.data_splitter import SLUDataSplitter
from src.feature_engineer import SLUFeatureEngineer
from src.data_builder import SLUDataBuilder
from src.evaluation import *

from src.model.crf import CRFModel
from src.utils import convert_slots_to_ids


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
    
    # Feature engineer
    feature_engineer = SLUFeatureEngineer(df, train_df, val_df)
    X_train_tfidf, X_val_tfidf, intent_encoder, slot_label_to_id, id_to_slot_label = feature_engineer.engineer_features()
    
    # Build CRF dataset
    data_builder = SLUDataBuilder(train_df, val_df)
    X_train_crf, y_train_crf, tokens_train_crf = data_builder.build_crf_dataset(train_df)
    X_val_crf, y_val_crf, tokens_val_crf = data_builder.build_crf_dataset(val_df)

    # Train a basic CRF model
    intent_to_id = {label: idx for idx, label in enumerate(intent_encoder.classes_)}
    evaluator = SLUEvaluator(slot_vocab=slot_label_to_id, intent_vocab=intent_to_id)
    crf_model = CRFModel(slot_label_to_id)
    crf_model.train(X_train_crf, y_train_crf)
    crf_results = crf_model.evaluate(evaluator, X_val_crf, y_val_crf)

    
