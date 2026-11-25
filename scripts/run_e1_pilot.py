"""
E1 Pilot Script: Testing Operating-Condition Normalization vs. Auxiliary Regime Inputs on FD004.

Conditions:
  C1: Global normalization (existing baseline)
  C2: Per-regime normalization, k=6 (existing condition-aware)
  C3: Global normalization + regime one-hot (6 dims) appended to every timestep
  C4: Global normalization + 3 raw operating settings (globally standardized) appended
  C5: Per-regime normalization + regime one-hot (both)

All runs use identical:
  - GRU (2 layers, hidden=64, dropout=0.2)
  - Window=30, Stride=5
  - Target scaling: y / 125.0
  - Adam lr=2e-3, batch=512, max_epochs=40, patience=8
  - Engine-level holdout validation split (20%, seed-dependent: 199 train, 50 val)
  - KMeans k=6 fit on train operating settings only
  - Evaluation on last window of test engines: Protocol A (unclipped) and Protocol B (clipped at 125)
"""

import os
import sys
import time
import copy
import random
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure repo root is on sys.path
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# Environment setup
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 8)
torch.set_num_threads(os.cpu_count() or 8)

from utils import make_train_windows, make_last_window, rmse, nasa_score
from model import GRURegressor

def get_git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode("ascii").strip()
        return h
    except Exception:
        return "unknown"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def predict_batches(model, tensor_X, device, batch_size=2048):
    model.eval()
    dataset = TensorDataset(tensor_X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds_list = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            out = model(bx)
            preds_list.append(out.cpu().numpy())
    raw_preds = np.concatenate(preds_list, axis=0)
    preds_scaled_back = raw_preds * 125.0
    return np.clip(preds_scaled_back, 0.0, None)

def prepare_e1_condition_data(condition, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(repo_dir, 'data_clean')

    train_path = os.path.join(data_dir, "train_FD004.csv")
    test_path = os.path.join(data_dir, "test_FD004.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Drop sensors that are constant in train data (std < 1e-10)
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    stds = train_df[sensor_cols].std()
    const_sensors = stds[stds < 1e-10].index.tolist()
    active_sensors = [s for s in sensor_cols if s not in const_sensors]
    op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']

    train_proc = train_df.copy()
    test_proc = test_df.copy()

    # Float64 precision
    train_proc[op_cols + active_sensors] = train_proc[op_cols + active_sensors].astype(np.float64)
    test_proc[op_cols + active_sensors] = test_proc[op_cols + active_sensors].astype(np.float64)

    # Fit KMeans strictly on training operating settings
    kmeans = KMeans(n_clusters=6, random_state=0, n_init=10)
    kmeans.fit(train_df[op_cols])
    train_clusters = kmeans.predict(train_df[op_cols])
    test_clusters = kmeans.predict(test_df[op_cols])

    if condition == 'C1':
        # Global normalization: standard pipeline (3 standardized op_settings + 21 standardized active sensors)
        feature_cols = op_cols + active_sensors
        scaler = StandardScaler()
        scaler.fit(train_proc[feature_cols])
        train_proc[feature_cols] = scaler.transform(train_proc[feature_cols])
        test_proc[feature_cols] = scaler.transform(test_proc[feature_cols])
        return train_proc, test_proc, feature_cols

    elif condition == 'C2':
        # Per-regime normalization, k=6 (existing condition-aware)
        feature_cols = op_cols + active_sensors
        op_scaler = StandardScaler()
        op_scaler.fit(train_proc[op_cols])
        train_proc[op_cols] = op_scaler.transform(train_proc[op_cols])
        test_proc[op_cols] = op_scaler.transform(test_proc[op_cols])

        for s in active_sensors:
            for c in range(6):
                tr_mask = (train_clusters == c)
                te_mask = (test_clusters == c)
                c_mean = train_df.loc[tr_mask, s].mean()
                c_std = train_df.loc[tr_mask, s].std()
                if c_std < 1e-10 or np.isnan(c_std):
                    c_std = 1.0
                train_proc.loc[tr_mask, s] = (train_df.loc[tr_mask, s].astype(float) - c_mean) / c_std
                test_proc.loc[te_mask, s] = (test_df.loc[te_mask, s].astype(float) - c_mean) / c_std
        return train_proc, test_proc, feature_cols

    elif condition == 'C3':
        # Global normalization + regime one-hot (6 dims) appended
        base_feature_cols = op_cols + active_sensors
        scaler = StandardScaler()
        scaler.fit(train_proc[base_feature_cols])
        train_proc[base_feature_cols] = scaler.transform(train_proc[base_feature_cols])
        test_proc[base_feature_cols] = scaler.transform(test_proc[base_feature_cols])

        one_hot_cols = [f"regime_oh_{c}" for c in range(6)]
        for c in range(6):
            train_proc[f"regime_oh_{c}"] = (train_clusters == c).astype(np.float64)
            test_proc[f"regime_oh_{c}"] = (test_clusters == c).astype(np.float64)
        feature_cols = base_feature_cols + one_hot_cols
        return train_proc, test_proc, feature_cols

    elif condition == 'C4':
        # Global normalization + raw 3 operating settings (globally standardized) appended
        base_feature_cols = op_cols + active_sensors
        scaler = StandardScaler()
        scaler.fit(train_proc[base_feature_cols])
        train_proc[base_feature_cols] = scaler.transform(train_proc[base_feature_cols])
        test_proc[base_feature_cols] = scaler.transform(test_proc[base_feature_cols])

        raw_op_cols = [f"raw_op_{i}" for i in [1, 2, 3]]
        op_scaler = StandardScaler()
        op_scaler.fit(train_df[op_cols])
        tr_std_op = op_scaler.transform(train_df[op_cols])
        te_std_op = op_scaler.transform(test_df[op_cols])
        for idx, col_name in enumerate(raw_op_cols):
            train_proc[col_name] = tr_std_op[:, idx]
            test_proc[col_name] = te_std_op[:, idx]
        feature_cols = base_feature_cols + raw_op_cols
        return train_proc, test_proc, feature_cols

    elif condition == 'C5':
        # Per-regime normalization + regime one-hot (both)
        base_feature_cols = op_cols + active_sensors
        op_scaler = StandardScaler()
        op_scaler.fit(train_proc[op_cols])
        train_proc[op_cols] = op_scaler.transform(train_proc[op_cols])
        test_proc[op_cols] = op_scaler.transform(test_proc[op_cols])

        for s in active_sensors:
            for c in range(6):
                tr_mask = (train_clusters == c)
                te_mask = (test_clusters == c)
                c_mean = train_df.loc[tr_mask, s].mean()
                c_std = train_df.loc[tr_mask, s].std()
                if c_std < 1e-10 or np.isnan(c_std):
                    c_std = 1.0
                train_proc.loc[tr_mask, s] = (train_df.loc[tr_mask, s].astype(float) - c_mean) / c_std
                test_proc.loc[te_mask, s] = (test_df.loc[te_mask, s].astype(float) - c_mean) / c_std

        one_hot_cols = [f"regime_oh_{c}" for c in range(6)]
        for c in range(6):
            train_proc[f"regime_oh_{c}"] = (train_clusters == c).astype(np.float64)
            test_proc[f"regime_oh_{c}"] = (test_clusters == c).astype(np.float64)
        feature_cols = base_feature_cols + one_hot_cols
        return train_proc, test_proc, feature_cols

    else:
        raise ValueError(f"Unknown condition: {condition}")

def run_single_e1(condition, seed, device=None):
    start_t = time.time()
    set_seed(seed)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, test_df, feature_cols = prepare_e1_condition_data(condition)
    num_features = len(feature_cols)

    # 20% engine holdout split strictly by seed
    engines = sorted(train_df['unit_number'].unique())
    tr_engines, val_engines = train_test_split(engines, test_size=0.2, random_state=seed)

    tr_df = train_df[train_df['unit_number'].isin(tr_engines)]
    val_df = train_df[train_df['unit_number'].isin(val_engines)]

    # Windows: window=30, stride=5
    X_tr, y_tr = make_train_windows(tr_df, feature_cols, window=30, stride=5)
    X_val, y_val = make_train_windows(val_df, feature_cols, window=30, stride=5)
    X_te, test_units = make_last_window(test_df, feature_cols, window=30)

    X_tr = X_tr.reshape(-1, 30, num_features)
    X_val = X_val.reshape(-1, 30, num_features)
    X_te = X_te.reshape(-1, 30, num_features)

    t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr / 125.0, dtype=torch.float32)
    t_X_val = torch.tensor(X_val, dtype=torch.float32)
    t_y_val = torch.tensor(y_val / 125.0, dtype=torch.float32)
    t_X_te = torch.tensor(X_te, dtype=torch.float32)

    test_rul_path = os.path.join(repo_dir, "data_clean/test_RUL_FD004.csv")
    test_rul_df = pd.read_csv(test_rul_path)
    assert list(test_units) == list(test_rul_df['unit_number'].tolist()), "Test engine ordering mismatch!"
    y_test_A = test_rul_df['true_RUL'].values.astype(float)
    y_test_B = np.minimum(y_test_A, 125.0)

    train_loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=512, shuffle=False)

    model = GRURegressor(input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    actual_epochs = 0

    for epoch in range(1, 41):
        actual_epochs = epoch
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * len(by)
        val_loss /= len(t_X_val)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                break

    model.load_state_dict(best_weights)
    test_preds = predict_batches(model, t_X_te, device, batch_size=2048)

    # Metrics
    best_val_rmse = float(np.sqrt(best_val_loss) * 125.0)
    te_rmse_A = rmse(y_test_A, test_preds)
    te_rmse_B = rmse(y_test_B, test_preds)
    te_nasa = nasa_score(y_test_A, test_preds)
    elapsed = time.time() - start_t
    git_h = get_git_hash()

    # Save predictions
    preds_dir = os.path.join(repo_dir, "results/preds")
    os.makedirs(preds_dir, exist_ok=True)
    pd.DataFrame({"unit_number": test_units, "true_RUL": y_test_A, "pred": test_preds}).to_csv(
        os.path.join(preds_dir, f"FD004_{condition}_s{seed}.csv"), index=False
    )

    return {
        "condition": condition,
        "seed": seed,
        "epochs": actual_epochs,
        "best_val_rmse": best_val_rmse,
        "rmse_A": te_rmse_A,
        "rmse_B": te_rmse_B,
        "nasa_score": te_nasa,
        "seconds": elapsed,
        "git_hash": git_h
    }

if __name__ == "__main__":
    OUT_CSV = os.path.join(repo_dir, "results/e1_pilot_runs.csv")
    COLS = ["condition", "seed", "epochs", "best_val_rmse", "rmse_A", "rmse_B", "nasa_score", "seconds", "git_hash"]

    if os.path.exists(OUT_CSV):
        df_exist = pd.read_csv(OUT_CSV)
        done_keys = set(zip(df_exist["condition"], df_exist["seed"]))
    else:
        pd.DataFrame(columns=COLS).to_csv(OUT_CSV, index=False)
        done_keys = set()

    conditions = ["C1", "C2", "C3", "C4", "C5"]
    seeds = [0, 1, 2]

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting E1 Pilot Execution (5 conditions x 3 seeds)...", flush=True)

    for cond in conditions:
        for seed in seeds:
            key = (cond, seed)
            if key in done_keys:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping {cond} seed {seed} (already done).", flush=True)
                continue

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {cond} seed {seed}...", flush=True)
            res = run_single_e1(cond, seed)
            row = [
                res["condition"],
                res["seed"],
                res["epochs"],
                res["best_val_rmse"],
                res["rmse_A"],
                res["rmse_B"],
                res["nasa_score"],
                res["seconds"],
                res["git_hash"]
            ]
            pd.DataFrame([row], columns=COLS).to_csv(OUT_CSV, mode="a", header=False, index=False)
            done_keys.add(key)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed {cond} seed {seed}: "
                  f"Val RMSE={res['best_val_rmse']:.2f}, Test RMSE(A)={res['rmse_A']:.2f}, "
                  f"Test RMSE(B)={res['rmse_B']:.2f}, NASA={res['nasa_score']:.1f}, Epochs={res['epochs']} ({res['seconds']:.1f}s)", flush=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All E1 pilot runs finished successfully!", flush=True)
