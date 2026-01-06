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

from sklearn_crfsuite import CRF, metrics
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

import scipy.stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import SLUDataLoader
from src.data_splitter import SLUDataSplitter
from src.feature_engineer import SLUFeatureEngineer
from src.data_builder import SLUDataBuilder
from src.evaluation import *

import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    data_dir = "./data"

    # Create a DataFrame to store the training data
    loader = SLUDataLoader(data_dir=data_dir)
    df = loader.load_all_data()
    
    # Split into train and test sets
    splitter = SLUDataSplitter(df=df, test_size=0.2, random_state=42)
    train_df, val_df = splitter.split_data()
    
    # Feature engineer
    feature_engineer = SLUFeatureEngineer(train_df, val_df)
    X_train_tfidf, X_val_tfidf, intent_encoder, slot_label_to_id, id_to_slot_label = feature_engineer.engineer_features()
    
    # Build CRF dataset
    data_builder = SLUDataBuilder(train_df, val_df)
    X_train_crf, y_train_crf, tokens_train_crf = data_builder.build_crf_dataset(train_df)
    X_val_crf, y_val_crf, tokens_val_crf = data_builder.build_crf_dataset(val_df)
    
