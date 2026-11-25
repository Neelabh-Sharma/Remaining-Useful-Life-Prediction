import os, sys, time, copy, random, subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 8)
torch.set_num_threads(os.cpu_count() or 8)

from utils import make_train_windows, make_last_window, rmse, nasa_score
from model import GRURegressor
from scripts.run_e1_pilot import prepare_e1_condition_data, predict_batches, set_seed, get_git_hash

def run_single_1a(condition, seed, max_epochs=100, patience=8, device=None):
    start_t = time.time()
    set_seed(seed)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, test_df, feature_cols = prepare_e1_condition_data(condition)
    num_features = len(feature_cols)

    engines = sorted(train_df['unit_number'].unique())
    tr_engines, val_engines = train_test_split(engines, test_size=0.2, random_state=seed)

    tr_df = train_df[train_df['unit_number'].isin(tr_engines)]
    val_df = train_df[train_df['unit_number'].isin(val_engines)]

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

    for epoch in range(1, max_epochs + 1):
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
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    test_preds = predict_batches(model, t_X_te, device, batch_size=2048)

    best_val_rmse = float(np.sqrt(best_val_loss) * 125.0)
    te_rmse_A = rmse(y_test_A, test_preds)
    te_rmse_B = rmse(y_test_B, test_preds)
    te_nasa = nasa_score(y_test_A, test_preds)
    elapsed = time.time() - start_t
    git_h = get_git_hash()

    preds_dir = os.path.join(repo_dir, "results/preds")
    os.makedirs(preds_dir, exist_ok=True)
    pd.DataFrame({"unit_number": test_units, "true_RUL": y_test_A, "pred": test_preds}).to_csv(
        os.path.join(preds_dir, f"FD004_100ep_{condition}_s{seed}.csv"), index=False
    )

    return {
        "condition": condition,
        "seed": seed,
        "max_epochs_cap": max_epochs,
        "epochs": actual_epochs,
        "best_val_rmse": best_val_rmse,
        "rmse_A": te_rmse_A,
        "rmse_B": te_rmse_B,
        "nasa_score": te_nasa,
        "seconds": elapsed,
        "git_hash": git_h
    }

if __name__ == "__main__":
    OUT_CSV = os.path.join(repo_dir, "results/tier1a_100epochs_runs.csv")
    COLS = ["condition", "seed", "max_epochs_cap", "epochs", "best_val_rmse", "rmse_A", "rmse_B", "nasa_score", "seconds", "git_hash"]

    if os.path.exists(OUT_CSV):
        df_exist = pd.read_csv(OUT_CSV)
        done_keys = set(zip(df_exist["condition"], df_exist["seed"]))
    else:
        pd.DataFrame(columns=COLS).to_csv(OUT_CSV, index=False)
        done_keys = set()

    conditions = ["C1", "C2", "C3", "C4"]
    seeds = [0, 1, 2]

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Tier 1A: 100-Epoch Sensitivity Runs...", flush=True)

    for cond in conditions:
        for seed in seeds:
            key = (cond, seed)
            if key in done_keys:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping {cond} seed {seed} (already done).", flush=True)
                continue

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {cond} seed {seed} (max 100 epochs)...", flush=True)
            res = run_single_1a(cond, seed, max_epochs=100, patience=8)
            row = [
                res["condition"],
                res["seed"],
                res["max_epochs_cap"],
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
                  f"Stopped at Epoch {res['epochs']}, Val RMSE={res['best_val_rmse']:.2f}, "
                  f"Test RMSE(A)={res['rmse_A']:.2f}, Test RMSE(B)={res['rmse_B']:.2f}, "
                  f"NASA={res['nasa_score']:.1f} ({res['seconds']:.1f}s)", flush=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Tier 1A completed successfully!", flush=True)
