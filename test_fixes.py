import os
import time
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

def run_test_fixes(subset="FD004", seed=0, stride=5, max_epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing fixes on {subset} seed {seed} (device: {device}, stride: {stride}, max_epochs: {max_epochs}, patience: {patience})")

    results = {}
    for mode_name, mode_code in [("GRU_global", "G"), ("GRU_condition_aware", "C")]:
        set_seed(seed)
        t0 = time.time()
        
        train_df, test_df, feature_cols, _, _ = get_preprocessed_data(subset, mode=mode_code)
        num_features = len(feature_cols)

        # 80/20 train/val engine split
        engines = sorted(train_df['unit_number'].unique())
        tr_engines, val_engines = train_test_split(engines, test_size=0.2, random_state=seed)

        tr_df = train_df[train_df['unit_number'].isin(tr_engines)]
        val_df = train_df[train_df['unit_number'].isin(val_engines)]

        X_tr, y_tr = make_train_windows(tr_df, feature_cols, window=30, stride=stride)
        X_val, y_val = make_train_windows(val_df, feature_cols, window=30, stride=stride)
        X_te, test_units = make_last_window(test_df, feature_cols, window=30)

        # Scale targets by 125
        y_tr_scaled = y_tr / 125.0
        y_val_scaled = y_val / 125.0

        X_tr_3d = X_tr.reshape(-1, 30, num_features)
        X_val_3d = X_val.reshape(-1, 30, num_features)
        X_te_3d = X_te.reshape(-1, 30, num_features)

        t_X_tr = torch.tensor(X_tr_3d, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr_scaled, dtype=torch.float32)
        t_X_val = torch.tensor(X_val_3d, dtype=torch.float32)
        t_y_val = torch.tensor(y_val_scaled, dtype=torch.float32)
        t_X_te = torch.tensor(X_te_3d, dtype=torch.float32).to(device)

        test_rul_df = pd.read_csv(f"data_clean/test_RUL_{subset}.csv")
        y_test = np.array([test_rul_df.loc[test_rul_df['unit_number'] == u, 'true_RUL'].values[0] for u in test_units], dtype=float)

        train_loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=256, shuffle=True)
        val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=256, shuffle=False)

        model = GRURegressor(input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)

        best_val_loss = float('inf')
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
        actual_epochs = 0
        best_epoch = 0

        for epoch in range(1, max_epochs + 1):
            actual_epochs = epoch
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                preds = model(bx)
                loss = criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            if epoch % 5 == 0 or epoch <= 5:
                print(f"[{mode_name}] Ep {epoch:3d} | Tr Loss: {train_loss:.5f} (RMSE: {np.sqrt(train_loss)*125:.2f}) | Val Loss: {val_loss:.5f} (RMSE: {np.sqrt(val_loss)*125:.2f}) | lr: {current_lr:.1e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[{mode_name}] Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val RMSE: {np.sqrt(best_val_loss)*125:.2f}")
                    break

        model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            test_preds_scaled = model(t_X_te).cpu().numpy()
            test_preds = np.clip(test_preds_scaled * 125.0, 0, None)

        test_rmse = rmse(y_test, test_preds)
        test_nasa = nasa_score(y_test, test_preds)
        elapsed = time.time() - t0

        print(f"[{mode_name}] Finished in {elapsed:.1f}s ({actual_epochs} epochs) | Test RMSE: {test_rmse:.2f} | Test NASA: {test_nasa:.1f}")
        print(f"  Pred stats: min={test_preds.min():.2f}, max={test_preds.max():.2f}, mean={test_preds.mean():.2f}, std={test_preds.std():.2f}")
        results[mode_name] = {
            'rmse': test_rmse,
            'nasa': test_nasa,
            'epochs': actual_epochs,
            'time': elapsed
        }

    print("\nSummary of Test Fixes on FD004 Seed 0:")
    for k, v in results.items():
        print(f"  {k}: RMSE = {v['rmse']:.2f}, NASA = {v['nasa']:.1f}, Epochs = {v['epochs']}, Time = {v['time']:.1f}s")

if __name__ == "__main__":
    run_test_fixes()

