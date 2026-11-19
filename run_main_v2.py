import os
import time
from datetime import datetime
import pandas as pd
from train import train_gru

os.makedirs("results", exist_ok=True)
os.makedirs("results/preds", exist_ok=True)
os.makedirs("logs", exist_ok=True)

out = "results/gru_runs_v2.csv"
cols = ["subset", "method", "seed", "rmse", "nasa_score", "epochs", "seconds"]

if os.path.exists(out):
    d = pd.read_csv(out)
    done = set(zip(d["subset"], d["method"], d["seed"]))
else:
    pd.DataFrame(columns=cols).to_csv(out, index=False)
    done = set()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting run_main_v2.py (Running FD001 & FD003, skipping existing)", flush=True)

for subset in ["FD004", "FD002", "FD001", "FD003"]:
    for seed in [0, 1, 2]:
        for method, mode in [("GRU_global", "G"), ("GRU_cond", "C")]:
            key = (subset, method, seed)
            pred_file = f"results/preds/{subset}_{mode}_kNone_f1.0_s{seed}.csv"
            if key in done and os.path.exists(pred_file):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping {subset} {method} seed {seed} (already done and pred saved)", flush=True)
                continue

            t0 = time.time()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {subset} {method} seed {seed}...", flush=True)
            r, s, el, ep = train_gru(
                subset=subset,
                mode=mode,
                seed=seed,
                max_epochs=40,
                patience=8,
                batch_size=512,
                lr=2e-3,
                verbose=False
            )
            row = [subset, method, seed, r, s, ep, el]
            pd.DataFrame([row], columns=cols).to_csv(out, mode="a", header=False, index=False)
            done.add(key)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done {subset} {method} seed {seed}: RMSE={r:.2f}, NASA={s:.1f}, Epochs={ep} ({el:.1f}s)", flush=True)

print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] run_main_v2.py COMPLETED ALL RUNS!\n", flush=True)
d = pd.read_csv(out)
d = d.drop_duplicates(subset=["subset", "method", "seed"], keep="last")
d.to_csv(out, index=False)
print(d.groupby(["subset", "method"])[["rmse", "nasa_score"]].agg(["mean", "std"]).round(2), flush=True)
