import os, sys, time
import numpy as np
import pandas as pd
from scipy import stats

"""
E2 & E3: Label Scarcity with Nested Subsets and Pre-Registered TOST Equivalence Testing
Equivalence margin delta: 1.5 cycles (~5% of FD004 test RMSE).
Two One-Sided Tests (TOST):
  H0: |mu_cond@x% - mu_global@100%| >= delta
  H1: -delta < mu_cond@x% - mu_global@100% < delta
"""

TOST_MARGIN = 1.5  # Pre-registered 1.5 cycles
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
N_SPLITS = 5
SEEDS = list(range(10, 20))

def run_tost(diffs, delta=TOST_MARGIN):
    """
    Two One-Sided Tests (TOST) for equivalence.
    t1: test if mean > -delta (t = (mean - (-delta)) / se)
    t2: test if mean < +delta (t = (mean - delta) / se)
    """
    n = len(diffs)
    mean_d = np.mean(diffs)
    se_d = np.std(diffs, ddof=1) / np.sqrt(n)
    df = n - 1
    
    t1 = (mean_d - (-delta)) / se_d
    p1 = 1 - stats.t.cdf(t1, df=df)  # One-sided upper
    
    t2 = (mean_d - delta) / se_d
    p2 = stats.t.cdf(t2, df=df)      # One-sided lower
    
    p_tost = max(p1, p2)
    return mean_d, se_d, p_tost

if __name__ == "__main__":
    print(f"E2 & E3 TOST script loaded. Pre-registered margin delta = {TOST_MARGIN} cycles.")
