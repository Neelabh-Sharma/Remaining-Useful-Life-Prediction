import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from utils import rmse, nasa_score

def compute_double_bootstrap(y_true, preds_g, preds_c, B=10000, seed=42):
    N_seeds = len(preds_g)
    N_eng = len(y_true)
    rng = np.random.RandomState(seed)
    diffs = np.empty(B)
    for b in range(B):
        s_idx = rng.randint(0, N_seeds, size=N_seeds)
        e_idx = rng.randint(0, N_eng, size=N_eng)
        yb = y_true[e_idx]
        pg = np.mean([preds_g[s][e_idx] for s in s_idx], axis=0)
        pc = np.mean([preds_c[s][e_idx] for s in s_idx], axis=0)
        diffs[b] = rmse(yb, pc) - rmse(yb, pg)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

if __name__ == "__main__":
    print("E5 & E6 statistics module ready.")
