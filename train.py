import os
import time
import copy
import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocess import get_preprocessed_data
from utils import make_train_windows, make_last_window, rmse, nasa_score
from model import GRURegressor

# Visibility & CPU optimization
torch.set_num_threads(os.cpu_count())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_batches(model, tensor_X, device, batch_size=2048):
    """Evaluates tensor_X in batches to avoid large memory allocations."""
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
    # Target was scaled to RUL_clipped / 125.0 during training; scale back by 125.0
    preds_scaled_back = raw_preds * 125.0
    return np.clip(preds_scaled_back, 0.0, None)


def train_gru(subset, mode, seed, max_epochs=30, patience=8, batch_size=256, lr=2e-3,
              hidden_size=64, min_epochs=1,
              device=None, verbose=False, return_details=False,
              k_override=None, label_frac=1.0, save_preds=True):
    """
    Trains GRURegressor on a given subset and normalization mode with seed.
    Target scaled to [0, 1] via y / 125.0.
    Adam lr=2e-3 with ReduceLROnPlateau(patience=3, factor=0.5), gradient clipping at 1.0.
    Validation: holds out 20% of train ENGINES, chosen by seed.
    Evaluates on the last window of every test engine vs true_RUL.

    k_override: number of KMeans clusters for mode 'C' (used for the ablation study).
    label_frac: fraction of the training engines that are kept (label-efficiency study).
                The validation engines are never reduced.
    """
    start_t = time.time()
    set_seed(seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Preprocess data
    train_df, test_df, feature_cols, _, _ = get_preprocessed_data(
        subset, mode=mode, k_override=k_override)
    num_features = len(feature_cols)

    # 2. Hold out 20% of train engines by seed
    engines = sorted(train_df['unit_number'].unique())
    tr_engines, val_engines = train_test_split(engines, test_size=0.2, random_state=seed)

    # Keep only a fraction of the training engines (validation engines unchanged)
    if label_frac < 1.0:
        rng = np.random.RandomState(seed)
        n_keep = max(2, int(round(label_frac * len(tr_engines))))
        tr_engines = list(rng.choice(tr_engines, size=n_keep, replace=False))

    tr_df = train_df[train_df['unit_number'].isin(tr_engines)]
    val_df = train_df[train_df['unit_number'].isin(val_engines)]

    # 3. Create windows (window=30, stride=5)
    X_tr, y_tr = make_train_windows(tr_df, feature_cols, window=30, stride=5)
    X_val, y_val = make_train_windows(val_df, feature_cols, window=30, stride=5)
    X_te, test_units = make_last_window(test_df, feature_cols, window=30)

    # Reshape from (N, 30*F) to (N, 30, F)
    X_tr = X_tr.reshape(-1, 30, num_features)
    X_val = X_val.reshape(-1, 30, num_features)
    X_te = X_te.reshape(-1, 30, num_features)

    # Convert to PyTorch tensors with target scaled to [0, 1]
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr / 125.0, dtype=torch.float32)

    t_X_val = torch.tensor(X_val, dtype=torch.float32)
    t_y_val = torch.tensor(y_val / 125.0, dtype=torch.float32)

    t_X_te = torch.tensor(X_te, dtype=torch.float32)

    # Ground truth test RUL
    test_rul_df = pd.read_csv(f"data_clean/test_RUL_{subset}.csv")
    assert list(test_units) == list(test_rul_df['unit_number'].tolist()), "Test engine ordering mismatch!"
    y_test = test_rul_df['true_RUL'].values.astype(float)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=batch_size, shuffle=False)

    # 4. Model, loss, optimizer, scheduler
    model = GRURegressor(input_size=num_features, hidden_size=hidden_size, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    actual_epochs = 0
    epoch_times = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        actual_epochs = epoch
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            # Gradient clipping at 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(t_X_tr)
        train_loss_history.append(train_loss)

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
        val_loss_history.append(val_loss)

        # Step LR scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        ep_duration = time.time() - ep_t0
        epoch_times.append(ep_duration)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if verbose:
            print(f"[{now_str}] Epoch {epoch:2d}/{max_epochs} | Train Loss (scaled): {train_loss:.5f} | "
                  f"Val Loss (scaled): {val_loss:.5f} | Val RMSE: {np.sqrt(val_loss) * 125:.2f} | "
                  f"lr: {current_lr:.1e} | ({ep_duration:.2f}s)", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= min_epochs:
                if verbose:
                    print(f"[{now_str}] Early stopping triggered at epoch {epoch} "
                          f"(no improvement for {patience} epochs, min_epochs={min_epochs}).", flush=True)
                break

    # Restore best model weights
    model.load_state_dict(best_weights)

    # Evaluate in batches of 2048
    test_preds = predict_batches(model, t_X_te, device, batch_size=2048)

    te_rmse = rmse(y_test, test_preds)
    te_nasa = nasa_score(y_test, test_preds)
    if save_preds:
        os.makedirs("results/preds", exist_ok=True)
        pd.DataFrame({"unit_number": test_units, "true_RUL": y_test, "pred": test_preds}).to_csv(
            f"results/preds/{subset}_{mode}_k{k_override}_f{label_frac}_s{seed}.csv", index=False)
    elapsed = time.time() - start_t

    if return_details:
        train_preds = predict_batches(model, t_X_tr, device, batch_size=2048)
        val_preds = predict_batches(model, t_X_val, device, batch_size=2048)
        train_rmse_val = rmse(y_tr, train_preds)
        val_rmse_val = rmse(y_val, val_preds)

        trivial_pred = np.full_like(y_test, np.mean(y_tr))
        trivial_rmse_val = rmse(y_test, trivial_pred)

        details = {
            'train_rmse': train_rmse_val,
            'val_rmse': val_rmse_val,
            'best_val_loss': best_val_loss,
            'best_val_rmse': float(np.sqrt(best_val_loss) * 125.0),
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'trivial_rmse': trivial_rmse_val,
            'test_preds': test_preds,
            'y_test': y_test,
            'epoch_times': epoch_times,
            'sec_per_epoch': np.mean(epoch_times) if epoch_times else 0.0,
            'pred_min': float(np.min(test_preds)),
            'pred_max': float(np.max(test_preds)),
            'pred_mean': float(np.mean(test_preds)),
            'pred_std': float(np.std(test_preds)),
        }
        return te_rmse, te_nasa, elapsed, actual_epochs, details

    return te_rmse, te_nasa, elapsed, actual_epochs


if __name__ == "__main__":
    print("Testing train.py on FD001 (Mode C, seed 0, 2 epochs)...", flush=True)
    rmse_val, nasa_val, elap, ep = train_gru("FD001", mode="C", seed=0, max_epochs=2, patience=2, verbose=True)
    print(f"Verified! Test RMSE: {rmse_val:.2f} | NASA: {nasa_val:.1f} in {elap:.2f}s ({ep} epochs)", flush=True)