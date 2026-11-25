import os
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from preprocess import get_preprocessed_data
from utils import make_last_window
from model import GRURegressor
from train import set_seed

op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']

def extract_hidden(sub, mode, seed=0):
    train_df, test_df, feats, _, _ = get_preprocessed_data(sub, mode=mode)
    Xte, units = make_last_window(test_df, feats, window=30)
    Xte = Xte.reshape(-1, 30, len(feats))
    t_Xte = torch.tensor(Xte, dtype=torch.float32)
    
    # We can inspect the model representation using the GRURegressor definition
    # Let's load the model weights if saved or train a quick probe
    # Note: weights are saved or trained
    return Xte, units

print("Linear probe data prepared.")
