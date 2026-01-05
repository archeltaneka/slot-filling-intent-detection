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
from sklearn.preprocessing import LabelEncoder

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
    
