import os, time
import pandas as pd
from train import train_gru

os.makedirs("results", exist_ok=True)
out = "results/gru_runs.csv"
cols = ["subset", "method", "seed", "rmse", "nasa_score", "epochs", "seconds"]
if os.path.exists(out):
    d = pd.read_csv(out)
    done = set(zip(d["subset"], d["method"], d["seed"]))
else:
    pd.DataFrame(columns=cols).to_csv(out, index=False)
    done = set()

for subset in ["FD004", "FD002", "FD001", "FD003"]:
    for seed in [0, 1, 2]:
        for method, mode in [("GRU_global", "G"), ("GRU_cond", "C")]:
            if (subset, method, seed) in done:
                continue
            r, s, el, ep = train_gru(subset, mode=mode, seed=seed, max_epochs=40,
                                     patience=8, batch_size=512, lr=2e-3, verbose=False)
            pd.DataFrame([[subset, method, seed, r, s, ep, el]], columns=cols).to_csv(
                out, mode="a", header=False, index=False)
            print(f"{subset} {method} seed {seed}: RMSE={r:.2f} NASA={s:.1f} "
                  f"epochs={ep} ({el:.0f}s)", flush=True)

d = pd.read_csv(out)
print(d.groupby(["subset", "method"])[["rmse", "nasa_score"]].agg(["mean", "std"]).round(2))