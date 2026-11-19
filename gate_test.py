import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from train import train_gru

torch.set_num_threads(os.cpu_count())

def run_gate_test():
    print("=" * 80, flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING QUICK GATE TEST ON FD004 (Seed 0)", flush=True)
    print("Methods: GRU_global (Mode G) vs GRU_condition_aware (Mode C)", flush=True)
    print("=" * 80, flush=True)

    # 1. Run GRU_global (Mode G)
    print("\n" + "-" * 70, flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running (B) GRU_global (Mode G) on FD004 seed 0...", flush=True)
    print("-" * 70, flush=True)
    b_rmse, b_nasa, b_elapsed, b_epochs, b_details = train_gru(
        subset="FD004",
        mode="G",
        seed=0,
        max_epochs=30,
        patience=8,
        batch_size=256,
        lr=2e-3,
        verbose=True,
        return_details=True
    )

    # 2. Run GRU_condition_aware (Mode C)
    print("\n" + "-" * 70, flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running (C) GRU_condition_aware (Mode C) on FD004 seed 0...", flush=True)
    print("-" * 70, flush=True)
    c_rmse, c_nasa, c_elapsed, c_epochs, c_details = train_gru(
        subset="FD004",
        mode="C",
        seed=0,
        max_epochs=30,
        patience=8,
        batch_size=256,
        lr=2e-3,
        verbose=True,
        return_details=True
    )

    # Trivial baseline
    trivial_rmse = b_details['trivial_rmse']
    true_test_rul = b_details['y_test']

    # 3. Print Results Summary
    print("\n" + "=" * 80, flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GATE TEST METRICS SUMMARY (FD004, Seed 0)", flush=True)
    print("=" * 80, flush=True)
    print(f"Trivial Baseline Test RMSE (predicting mean train RUL): {trivial_rmse:.4f}")
    print(f"Ground Truth Test RUL: min={np.min(true_test_rul):.1f}, max={np.max(true_test_rul):.1f}, mean={np.mean(true_test_rul):.1f}, std={np.std(true_test_rul):.1f}")
    print("-" * 80, flush=True)
    print(f"{'Metric':<30} | {'(B) GRU_global':<20} | {'(C) GRU_cond_aware':<20}")
    print("-" * 80, flush=True)
    print(f"{'Epochs Used':<30} | {b_epochs:<20} | {c_epochs:<20}")
    print(f"{'Seconds per Epoch':<30} | {b_details['sec_per_epoch']:<20.2f} | {c_details['sec_per_epoch']:<20.2f}")
    print(f"{'Total Elapsed Time (s)':<30} | {b_elapsed:<20.2f} | {c_elapsed:<20.2f}")
    print(f"{'Train RMSE':<30} | {b_details['train_rmse']:<20.4f} | {c_details['train_rmse']:<20.4f}")
    print(f"{'Validation RMSE':<30} | {b_details['val_rmse']:<20.4f} | {c_details['val_rmse']:<20.4f}")
    print(f"{'Test RMSE':<30} | {b_rmse:<20.4f} | {c_rmse:<20.4f}")
    print(f"{'NASA Score':<30} | {b_nasa:<20.1f} | {c_nasa:<20.1f}")
    print(f"{'Pred Min':<30} | {b_details['pred_min']:<20.2f} | {c_details['pred_min']:<20.2f}")
    print(f"{'Pred Max':<30} | {b_details['pred_max']:<20.2f} | {c_details['pred_max']:<20.2f}")
    print(f"{'Pred Mean':<30} | {b_details['pred_mean']:<20.2f} | {c_details['pred_mean']:<20.2f}")
    print(f"{'Pred Std':<30} | {b_details['pred_std']:<20.2f} | {c_details['pred_std']:<20.2f}")
    print("=" * 80, flush=True)

    # Check PASS criteria
    # Criterion 1: test RMSE of GRU_global is below trivial-baseline RMSE by a clear margin
    # Criterion 2: GRU_condition_aware is lower than GRU_global
    crit1 = (trivial_rmse - b_rmse) > 5.0
    crit2 = c_rmse < b_rmse

    print(f"Criterion 1 (GRU_global < Trivial [{trivial_rmse:.2f}] by clear margin): {'PASS' if crit1 else 'FAIL'} (Margin = {trivial_rmse - b_rmse:.2f})", flush=True)
    print(f"Criterion 2 (GRU_cond_aware [{c_rmse:.2f}] < GRU_global [{b_rmse:.2f}]): {'PASS' if crit2 else 'FAIL'} (Difference = {b_rmse - c_rmse:.2f})", flush=True)

    passed = crit1 and crit2
    if passed:
        print("\n>>> OVERALL GATE TEST RESULT: PASS <<<", flush=True)
    else:
        print("\n>>> OVERALL GATE TEST RESULT: FAIL <<<", flush=True)
    print("=" * 80, flush=True)
    return passed

if __name__ == "__main__":
    passed = run_gate_test()
    sys.exit(0 if passed else 1)

