import os
import sys
import time
import ctypes
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Prevent Windows from sleeping during long computation
try:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    print("Windows sleep prevention activated (SetThreadExecutionState).", flush=True)
except Exception as e:
    print(f"Warning: Could not set execution state: {e}", flush=True)

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

from train import train_gru

OUT_CSV = "results/gru_runs_v3.csv"
PREDS_DIR = "results/preds"
os.makedirs("results", exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

cols = ["subset", "method", "hidden", "seed", "rmse", "nasa_score", "epochs", "seconds"]

# Initialize or load existing gru_runs_v3.csv
if os.path.exists(OUT_CSV):
    df_v3 = pd.read_csv(OUT_CSV)
    done_keys = set(zip(df_v3["subset"], df_v3["method"], df_v3["hidden"].astype(int), df_v3["seed"].astype(int)))
else:
    df_v3 = pd.DataFrame(columns=cols)
    df_v3.to_csv(OUT_CSV, index=False)
    done_keys = set()

# Seed 0-2 for GRU_cond (hidden=64): reuse existing runs
existing_runs = pd.read_csv("results/gru_runs.csv")
for sub in ["FD004", "FD002"]:
    for seed in [0, 1, 2]:
        key = (sub, "GRU_cond", 64, seed)
        pred_path_h64 = os.path.join(PREDS_DIR, f"{sub}_C_h64_s{seed}.csv")
        pred_path_orig = os.path.join(PREDS_DIR, f"{sub}_C_kNone_f1.0_s{seed}.csv")
        
        # Ensure pred file exists
        if os.path.exists(pred_path_orig) and not os.path.exists(pred_path_h64):
            pd.read_csv(pred_path_orig).to_csv(pred_path_h64, index=False)
            
        if key not in done_keys:
            r = existing_runs[(existing_runs["subset"] == sub) & 
                              (existing_runs["method"] == "GRU_cond") & 
                              (existing_runs["seed"] == seed)].iloc[0]
            row = [sub, "GRU_cond", 64, seed, float(r["rmse"]), float(r["nasa_score"]), int(r["epochs"]), float(r["seconds"])]
            pd.DataFrame([row], columns=cols).to_csv(OUT_CSV, mode="a", header=False, index=False)
            done_keys.add(key)
            print(f"Loaded existing GRU_cond seed {seed} for {sub} (RMSE={r['rmse']:.2f}, NASA={r['nasa_score']:.1f})", flush=True)

# Build run list for Stage 3
# Order: FD004 then FD002; cond seeds 3-9 first, then global seeds 0-9
tasks = []
for sub in ["FD004", "FD002"]:
    # GRU_cond (hidden=64, seeds 3-9)
    for seed in range(3, 10):
        tasks.append((sub, "GRU_cond", "C", 64, seed))
        
for sub in ["FD004", "FD002"]:
    # GRU_global (hidden=128, seeds 0-9)
    for seed in range(0, 10):
        tasks.append((sub, "GRU_global", "G", 128, seed))

print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting STAGE 3 Runs (Total planned tasks: {len(tasks)})...", flush=True)

last_step_time = time.time()

for sub, method, mode, hidden, seed in tasks:
    key = (sub, method, hidden, seed)
    pred_path_h = os.path.join(PREDS_DIR, f"{sub}_{mode}_h{hidden}_s{seed}.csv")
    
    if key in done_keys and os.path.exists(pred_path_h):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping {sub} {method} h={hidden} seed {seed} (already done).", flush=True)
        continue

    # Watchdog for sleep gap
    now_t = time.time()
    time_gap = now_t - last_step_time
    if time_gap > 900:
        print(f"\n[WARNING] Time gap of {time_gap/60:.1f} minutes detected between runs! Laptop may have slept.\n", flush=True)
    last_step_time = now_t

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {sub} {method} (hidden={hidden}, seed={seed})...", flush=True)
    t0 = time.time()
    
    te_rmse, te_nasa, elapsed, ep, details = train_gru(
        subset=sub,
        mode=mode,
        seed=seed,
        max_epochs=40,
        patience=8,
        min_epochs=1,
        batch_size=512,
        lr=2e-3,
        hidden_size=hidden,
        verbose=False,
        return_details=True,
        save_preds=False
    )
    
    # Save predictions
    test_rul_df = pd.read_csv(f"data_clean/test_RUL_{sub}.csv")
    df_pred = pd.DataFrame({
        "unit_number": test_rul_df["unit_number"],
        "true_RUL": test_rul_df["true_RUL"],
        "pred": details["test_preds"]
    })
    df_pred.to_csv(pred_path_h, index=False)
    if method == "GRU_cond":
        # Also maintain standard naming
        df_pred.to_csv(os.path.join(PREDS_DIR, f"{sub}_C_kNone_f1.0_s{seed}.csv"), index=False)
        
    row = [sub, method, hidden, seed, te_rmse, te_nasa, ep, elapsed]
    pd.DataFrame([row], columns=cols).to_csv(OUT_CSV, mode="a", header=False, index=False)
    done_keys.add(key)
    
    last_step_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished {sub} {method} h={hidden} seed {seed}: "
          f"RMSE={te_rmse:.4f}, NASA={te_nasa:.1f}, Epochs={ep} ({elapsed:.1f}s)", flush=True)

print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALL STAGE 3 TRAINING RUNS COMPLETED!", flush=True)
