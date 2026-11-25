import os
import sys
import time
from datetime import datetime
import pandas as pd
from run_e1_pilot import run_single_e1

os.makedirs("results", exist_ok=True)
os.makedirs("results/preds", exist_ok=True)
os.makedirs("logs", exist_ok=True)

OUT_CSV = "results/e1_pilot_runs.csv"
COLS = ["condition", "seed", "epochs", "best_val_rmse", "rmse_A", "rmse_B", "nasa_score", "seconds", "git_hash"]

# Load already finished runs to allow resume
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
