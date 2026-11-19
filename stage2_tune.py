import os
import sys
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

from train import train_gru

RESULTS_CSV = "results/stage2_tuning.csv"
HISTORY_JSON = "results/stage2_history.json"
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Define Configs A-D
CONFIGS = {
    "A": {
        "desc": "Current default (lr=2e-3, h=64, patience=8, min_epochs=1)",
        "lr": 2e-3,
        "hidden_size": 64,
        "patience": 8,
        "min_epochs": 1,
        "max_epochs": 40,
        "batch_size": 512,
    },
    "B": {
        "desc": "Lower LR (lr=1e-3, h=64, patience=8, min_epochs=1)",
        "lr": 1e-3,
        "hidden_size": 64,
        "patience": 8,
        "min_epochs": 1,
        "max_epochs": 40,
        "batch_size": 512,
    },
    "C": {
        "desc": "Larger capacity (lr=2e-3, h=128, patience=8, min_epochs=1)",
        "lr": 2e-3,
        "hidden_size": 128,
        "patience": 8,
        "min_epochs": 1,
        "max_epochs": 40,
        "batch_size": 512,
    },
    "D": {
        "desc": "Extended patience (lr=2e-3, h=64, patience=15, min_epochs=25)",
        "lr": 2e-3,
        "hidden_size": 64,
        "patience": 15,
        "min_epochs": 25,
        "max_epochs": 40,
        "batch_size": 512,
    },
}

cols = [
    "config", "subset", "method", "mode", "seed",
    "lr", "hidden_size", "patience", "min_epochs",
    "val_rmse", "best_val_loss", "test_rmse", "nasa_score",
    "actual_epochs", "elapsed_sec"
]

if os.path.exists(RESULTS_CSV):
    df_existing = pd.read_csv(RESULTS_CSV)
    done_keys = set(zip(df_existing["config"], df_existing["subset"], df_existing["method"], df_existing["seed"]))
else:
    df_existing = pd.DataFrame(columns=cols)
    df_existing.to_csv(RESULTS_CSV, index=False)
    done_keys = set()

history_dict = {}
if os.path.exists(HISTORY_JSON):
    try:
        with open(HISTORY_JSON, "r") as f:
            history_dict = json.load(f)
    except Exception:
        history_dict = {}

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting STAGE 2 Tuning Runs...", flush=True)

# 1. FD004 Runs: 4 configs x 2 methods x 2 seeds (0, 1)
fd004_targets = []
for cfg_name in ["A", "B", "C", "D"]:
    for method, mode in [("GRU_global", "G"), ("GRU_cond", "C")]:
        for seed in [0, 1]:
            fd004_targets.append((cfg_name, "FD004", method, mode, seed))

# 2. FD003 Analysis Runs (to diagnose early stopping instability)
fd003_targets = []
for cfg_name in ["A", "D"]:
    for method, mode in [("GRU_cond", "C")]:
        for seed in [0, 1, 2]:
            fd003_targets.append((cfg_name, "FD003", method, mode, seed))

all_targets = fd004_targets + fd003_targets

for cfg_name, sub, method, mode, seed in all_targets:
    key = (cfg_name, sub, method, seed)
    if key in done_keys:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping Config {cfg_name} {sub} {method} seed {seed} (already completed).", flush=True)
        continue

    cfg = CONFIGS[cfg_name]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running Config {cfg_name} ({cfg['desc']}) on {sub} {method} seed {seed}...", flush=True)
    t0 = time.time()
    
    te_rmse, te_nasa, elapsed, ep, details = train_gru(
        subset=sub,
        mode=mode,
        seed=seed,
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        min_epochs=cfg["min_epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        hidden_size=cfg["hidden_size"],
        verbose=False,
        return_details=True,
        save_preds=False  # Do not overwrite main benchmark predictions
    )
    
    val_rmse = details["val_rmse"]
    best_val_loss = details["best_val_loss"]
    
    row = [
        cfg_name, sub, method, mode, seed,
        cfg["lr"], cfg["hidden_size"], cfg["patience"], cfg["min_epochs"],
        val_rmse, best_val_loss, te_rmse, te_nasa,
        ep, elapsed
    ]
    pd.DataFrame([row], columns=cols).to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    done_keys.add(key)
    
    # Save history
    hist_key = f"{cfg_name}_{sub}_{method}_s{seed}"
    history_dict[hist_key] = {
        "val_loss": details["val_loss_history"],
        "train_loss": details["train_loss_history"],
        "epoch_times": details["epoch_times"],
        "val_rmse": val_rmse,
        "test_rmse": te_rmse,
        "epochs": ep
    }
    with open(HISTORY_JSON, "w") as f:
        json.dump(history_dict, f, indent=2)
        
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done Config {cfg_name} {sub} {method} seed {seed}: "
          f"Val RMSE={val_rmse:.2f}, Test RMSE={te_rmse:.2f}, Epochs={ep} ({elapsed:.1f}s)", flush=True)

print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALL STAGE 2 RUNS COMPLETED SUCCESSFULLY!", flush=True)
