import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocess import get_preprocessed_data
from utils import make_train_windows, make_last_window, rmse, nasa_score
from model import GRURegressor
from train import set_seed

def diagnose_fd004():
    print("=" * 85, flush=True)
    print("DIAGNOSIS ON FD004 SEED 0 (GRU & DATA INTEGRITY)", flush=True)
    print("=" * 85, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch Device: {device}", flush=True)

    subset = "FD004"
    seed = 0
    set_seed(seed)

    # 1. Load data
    train_df, test_df, feature_cols, k, n_sensors = get_preprocessed_data(subset, mode='G')
    num_features = len(feature_cols)
    print(f"\nSubset {subset}: Total features = {num_features} (3 op settings + {n_sensors} active sensors), Mode C k = {k}", flush=True)

    # 2b. Trivial model baseline: predict mean train RUL_clipped for every test engine
    mean_train_rul = train_df['RUL_clipped'].mean()
    test_rul_df = pd.read_csv(f"data_clean/test_RUL_{subset}.csv")
    test_engines = sorted(test_df['unit_number'].unique())
    y_test = np.array([test_rul_df.loc[test_rul_df['unit_number'] == u, 'true_RUL'].values[0] for u in test_engines], dtype=float)

    trivial_preds = np.full_like(y_test, mean_train_rul)
    trivial_rmse = rmse(y_test, trivial_preds)
    trivial_nasa = nasa_score(y_test, trivial_preds)
    print(f"\n--- 2b. TRIVIAL MODEL BASELINE (Predicting Mean Train RUL_clipped) ---", flush=True)
    print(f"  Mean Train RUL_clipped:          {mean_train_rul:.4f}", flush=True)
    print(f"  Trivial Model Test RMSE:         {trivial_rmse:.4f}", flush=True)
    print(f"  Trivial Model Test NASA Score:   {trivial_nasa:.1f}", flush=True)

    # (3) Engine lengths and padding check
    print(f"\n--- (3) ENGINE TRAJECTORY LENGTHS & PADDING CHECK ---", flush=True)
    train_lengths = train_df.groupby('unit_number').size()
    test_lengths = test_df.groupby('unit_number').size()

    train_short_engines = (train_lengths < 30).sum()
    test_short_engines = (test_lengths < 30).sum()

    print(f"  Train engines count: {len(train_lengths)} | Min cycles: {train_lengths.min()}, Max cycles: {train_lengths.max()}", flush=True)
  
    print(f"  Test engines count:  {len(test_lengths)} | Min cycles: {test_lengths.min()}, Max cycles: {test_lengths.max()}", flush=True)
    print(f"  Count of test engines shorter than 30 cycles: {test_short_engines}", flush=True)
    if test_short_engines > 0:
        short_test_units = test_lengths[test_lengths < 30]
        print(f"    Short test engine unit IDs and lengths: {dict(short_test_units)}")

    # Check how many training windows were built from padded sequences
    train_padded_windows = 0
    for u, group in train_df.groupby('unit_number'):
        if len(group) < 30:
            train_padded_windows += 1
    print(f"  Train windows built from padded sequences: {train_padded_windows}", flush=True)

    # 2d & (1) Window verification and unit alignment assertion
    print(f"\n--- 2d & (1) WINDOW VERIFICATION & ALIGNMENT ASSERTION ---", flush=True)
    X_tr_all, y_tr_all = make_train_windows(train_df, feature_cols, window=30, stride=5)
    X_te_all, te_units = make_last_window(test_df, feature_cols, window=30)

    # (1) Assert list(te_units) == list(test_engines)
    assert list(te_units) == list(test_engines), f"Unit ID alignment mismatch! te_units != test_engines"
    print(f"  ASSERTION PASSED: list(te_units) == list(test_engines) is True! ({len(te_units)} engines verified)", flush=True)

    print(f"  Train windows shape: {X_tr_all.shape} | Test last-windows shape: {X_te_all.shape}", flush=True)
    print(f"  Train window 0 (first 6 values):       {np.round(X_tr_all[0, :6], 4)}", flush=True)
    print(f"  Test engine 1 window (first 6 values): {np.round(X_te_all[0, :6], 4)}", flush=True)
    print(f"  Train window 0 (last 6 values):        {np.round(X_tr_all[0, -6:], 4)}", flush=True)
    print(f"  Test engine 1 window (last 6 values):  {np.round(X_te_all[0, -6:], 4)}", flush=True)

    # (2) Real scaling check: per-feature mean and std of all train windows vs all test last-windows (first 10 features)
    print(f"\n--- (2) REAL SCALING CHECK: TRAIN WINDOWS vs TEST LAST-WINDOWS (First 10 Features) ---", flush=True)
    # Reshape windows to (-1, num_features) to inspect feature distributions across all time steps
    X_tr_flat_steps = X_tr_all.reshape(-1, num_features)
    X_te_flat_steps = X_te_all.reshape(-1, num_features)

    print(f"{'Feature Name':<15} | {'Train Mean':<12} | {'Train Std':<12} | {'Test Mean':<12} | {'Test Std':<12}")
    print("-" * 75)
    for i in range(min(10, num_features)):
        fname = feature_cols[i]
        tr_m = X_tr_flat_steps[:, i].mean()
        tr_s = X_tr_flat_steps[:, i].std()
        te_m = X_te_flat_steps[:, i].mean()
        te_s = X_te_flat_steps[:, i].std()
        print(f"{fname:<15} | {tr_m:12.4f} | {tr_s:12.4f} | {te_m:12.4f} | {te_s:12.4f}", flush=True)

    # Run Epoch-by-Epoch Diagnosis on Mode G and Mode C
    for mode_name, mode_code in [("Mode G (Global)", "G"), ("Mode C (Condition-Aware)", "C")]:
        print(f"\n" + "=" * 85, flush=True)
        print(f"DIAGNOSIS FOR {mode_name} ON FD004 SEED 0", flush=True)
        print("=" * 85, flush=True)

        set_seed(seed)
        tr_df_m, te_df_m, feats, _, _ = get_preprocessed_data(subset, mode=mode_code)

        engines = sorted(tr_df_m['unit_number'].unique())
        tr_engs, val_engs = train_test_split(engines, test_size=0.2, random_state=seed)

        tr_data = tr_df_m[tr_df_m['unit_number'].isin(tr_engs)]
        val_data = tr_df_m[tr_df_m['unit_number'].isin(val_engs)]

        X_tr, y_tr = make_train_windows(tr_data, feats, window=30, stride=5)
        X_val, y_val = make_train_windows(val_data, feats, window=30, stride=5)
        X_te, test_u = make_last_window(te_df_m, feats, window=30)

        X_tr_3d = X_tr.reshape(-1, 30, num_features)
        X_val_3d = X_val.reshape(-1, 30, num_features)
        X_te_3d = X_te.reshape(-1, 30, num_features)

        t_X_tr = torch.tensor(X_tr_3d, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr, dtype=torch.float32)
        t_X_val = torch.tensor(X_val_3d, dtype=torch.float32)
        t_y_val = torch.tensor(y_val, dtype=torch.float32)
        t_X_te = torch.tensor(X_te_3d, dtype=torch.float32).to(device)

        train_loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=256, shuffle=True)
        val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=256, shuffle=False)

        model = GRURegressor(input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_loss = float('inf')
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
        early_stop_epoch = None
        best_epoch = 0

        print(f"\n--- 2a. EPOCH-BY-EPOCH TRAINING LOG ({mode_name}) ---", flush=True)
        for epoch in range(1, 41):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                preds = model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(by)
            train_loss /= len(t_X_tr)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    preds = model(bx)
                    loss = criterion(preds, by)
                    val_loss += loss.item() * len(by)
            val_loss /= len(t_X_val)
            val_rmse = np.sqrt(val_loss)

            print(f"Epoch {epoch:2d}/40 | Train Loss: {train_loss:8.2f} (RMSE: {np.sqrt(train_loss):6.2f}) | Val Loss: {val_loss:8.2f} (RMSE: {val_rmse:6.2f})", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 6:
                    early_stop_epoch = epoch
                    print(f"  -> Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val RMSE: {np.sqrt(best_val_loss):.2f}", flush=True)
                    break

        if early_stop_epoch is None:
            early_stop_epoch = 40
            print(f"  -> Reached maximum epochs (40). Best epoch was {best_epoch} with Val RMSE: {np.sqrt(best_val_loss):.2f}", flush=True)

        model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            tr_eval_preds = model(t_X_tr.to(device)).cpu().numpy()
            val_eval_preds = model(t_X_val.to(device)).cpu().numpy()
            test_preds = model(t_X_te).cpu().numpy()

        tr_eval_preds = np.clip(tr_eval_preds, 0, None)
        val_eval_preds = np.clip(val_eval_preds, 0, None)
        test_preds = np.clip(test_preds, 0, None)

        train_rmse_fit = rmse(y_tr, tr_eval_preds)
        val_rmse_fit = rmse(y_val, val_eval_preds)
        test_rmse_fit = rmse(y_test, test_preds)
        test_nasa_fit = nasa_score(y_test, test_preds)

        print(f"\n--- 2c. FIT CHECK ON TRAIN, VAL, TEST ({mode_name}) ---", flush=True)
        print(f"  GRU RMSE on its own TRAIN windows: {train_rmse_fit:.4f}", flush=True)
        print(f"  GRU RMSE on VALIDATION windows:    {val_rmse_fit:.4f}", flush=True)
        print(f"  GRU RMSE on TEST last-windows:     {test_rmse_fit:.4f}", flush=True)
        print(f"  GRU NASA Score on TEST:            {test_nasa_fit:.1f}", flush=True)

        print(f"\n--- 2e. PREDICTION vs TRUE RUL DISTRIBUTION ({mode_name}) ---", flush=True)
        print(f"  True Test RUL: min={np.min(y_test):.2f}, max={np.max(y_test):.2f}, mean={np.mean(y_test):.2f}, std={np.std(y_test):.2f}", flush=True)
        print(f"  Test Preds:    min={np.min(test_preds):.2f}, max={np.max(test_preds):.2f}, mean={np.mean(test_preds):.2f}, std={np.std(test_preds):.2f}", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("DIAGNOSIS COMPLETED", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    diagnose_fd004()
