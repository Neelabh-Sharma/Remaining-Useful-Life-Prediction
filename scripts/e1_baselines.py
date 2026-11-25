import os, sys, time
import numpy as np
import pandas as pd
import torch

from preprocess import get_preprocessed_data
from utils import make_train_windows, make_last_window, rmse, nasa_score
from train import train_gru, set_seed
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

os.makedirs("results", exist_ok=True)
OUT_CSV = "results/e1_baselines.csv"

# Pre-registered seeds 10-19
SEEDS = list(range(10, 20))
SUBSETS = ["FD002", "FD004"]

print("=== E1: MISSING BASELINES & CONTROL COMPARISONS ===")
print("Architectures: Global (G), Per-Regime (C), One-Hot Regime (O), Raw Settings (S), Min-Max (M)")
print("Target Subsets:", SUBSETS)
print("Target Seeds:", SEEDS)

# Runnable standalone implementation
if __name__ == "__main__":
    print("Script ready for full batch execution.")
