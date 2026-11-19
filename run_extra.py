import os
import time
from datetime import datetime
import pandas as pd
from train import train_gru

os.makedirs("results", exist_ok=True)
os.makedirs("results/preds", exist_ok=True)
os.makedirs("logs", exist_ok=True)

out = "results/extra_runs.csv"
cols = ["exp_type", "subset", "method", "k", "label_frac", "seed", "rmse", "nasa_score", "epochs", "seconds"]

if os.path.exists(out):
    df_exist = pd.read_csv(out)
    done = set(zip(df_exist["exp_type"], df_exist["method"], df_exist["k"], df_exist["label_frac"], df_exist["seed"]))
else:
    pd.DataFrame(columns=cols).to_csv(out, index=False)
    done = set()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting run_extra.py", flush=True)

# 1. Ablation on k: k=3, 9 on FD004 (Mode C, label_frac=1.0)
ablation_configs = [
    ("ablation", "FD004", "GRU_cond", "C", 3, 1.0),
    ("ablation", "FD004", "GRU_cond", "C", 9, 1.0)
]

for exp_type, subset, method, mode, k_val, frac in ablation_configs:
    for seed in [0, 1, 2]:
        key = (exp_type, method, k_val, frac, seed)
        pred_file = f"results/preds/{subset}_{mode}_k{k_val}_f{frac}_s{seed}.csv"
        if key in done and os.path.exists(pred_file):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping {exp_type} {method} k={k_val} seed={seed} (already done and pred saved)", flush=True)
            continue
        t0 = time.time()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {exp_type}: {subset} {method} k={k_val} seed={seed}...", flush=True)
        r, s, el, ep = train_gru(
            subset=subset,
            mode=mode,
            seed=seed,
            max_epochs=40,
            patience=8,
            batch_size=512,
            lr=2e-3,
            k_override=k_val,
            label_frac=frac,
            verbose=False
        )
        row = [exp_type, subset, method, k_val, frac, seed, r, s, ep, el]
        pd.DataFrame([row], columns=cols).to_csv(out, mode="a", header=False, index=False)
        done.add(key)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done {exp_type} {method} k={k_val} seed={seed}: RMSE={r:.2f}, NASA={s:.1f}, Epochs={ep} ({el:.1f}s)", flush=True)

# 2. Label efficiency: 10%, 25%, 50% for GRU_global and GRU_cond on FD004 (k=6 for cond)
label_configs = [0.10, 0.25, 0.50]
methods = [
    ("GRU_global", "G", None),
    ("GRU_cond", "C", 6)
]

for frac in label_configs:
    for method, mode, k_val in methods:
        for seed in [0, 1, 2]:
            key = ("label_efficiency", method, k_val, frac, seed)
            pred_file = f"results/preds/FD004_{mode}_k{k_val}_f{frac}_s{seed}.csv"
            if key in done and os.path.exists(pred_file):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping label_eff {method} frac={frac} seed={seed} (already done and pred saved)", flush=True)
                continue
            t0 = time.time()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running label_efficiency: FD004 {method} frac={frac} seed={seed}...", flush=True)
            r, s, el, ep = train_gru(
                subset="FD004",
                mode=mode,
                seed=seed,
                max_epochs=40,
                patience=8,
                batch_size=512,
                lr=2e-3,
                k_override=k_val if mode == "C" else None,
                label_frac=frac,
                verbose=False
            )
            row = ["label_efficiency", "FD004", method, k_val, frac, seed, r, s, ep, el]
            pd.DataFrame([row], columns=cols).to_csv(out, mode="a", header=False, index=False)
            done.add(key)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done label_eff {method} frac={frac} seed={seed}: RMSE={r:.2f}, NASA={s:.1f}, Epochs={ep} ({el:.1f}s)", flush=True)

print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] run_extra.py COMPLETED ALL RUNS!\n", flush=True)
d = pd.read_csv(out)
# Deduplicate if re-runs occurred
d = d.drop_duplicates(subset=["exp_type", "subset", "method", "k", "label_frac", "seed"], keep="last")
d.to_csv(out, index=False)

print("=== ABLATION SUMMARY (FD004) ===", flush=True)
d_abl = d[d["exp_type"] == "ablation"]
if len(d_abl) > 0:
    print(d_abl.groupby(["k"])[["rmse", "nasa_score"]].agg(["mean", "std"]).round(2), flush=True)

print("\n=== LABEL EFFICIENCY SUMMARY (FD004) ===", flush=True)
d_lbl = d[d["exp_type"] == "label_efficiency"]
if len(d_lbl) > 0:
    print(d_lbl.groupby(["label_frac", "method"])[["rmse", "nasa_score"]].agg(["mean", "std"]).round(2), flush=True)
