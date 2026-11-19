import os, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from preprocess import get_preprocessed_data
from utils import make_train_windows, make_last_window, rmse, nasa_score

os.makedirs("results", exist_ok=True)
out = "results/rf_cond_runs.csv"
if os.path.exists(out):
    d = pd.read_csv(out)
    done = set(zip(d["subset"], d["seed"]))
else:
    pd.DataFrame(columns=["subset", "method", "seed", "rmse", "nasa_score"]).to_csv(out, index=False)
    done = set()

for subset in ["FD004", "FD002", "FD001", "FD003"]:
    train_df, test_df, feats, k, n_s = get_preprocessed_data(subset, mode="C")
    X, y = make_train_windows(train_df, feats, window=30, stride=5)
    Xte, units = make_last_window(test_df, feats, window=30)
    truth = pd.read_csv(f"data_clean/test_RUL_{subset}.csv").set_index("unit_number")["true_RUL"]
    y_true = np.array([truth.loc[u] for u in units], dtype=float)
    for seed in [0, 1, 2]:
        if (subset, seed) in done:
            continue
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=seed)
        rf.fit(X, y)
        p = np.clip(rf.predict(Xte), 0, None)
        r, s = rmse(y_true, p), nasa_score(y_true, p)
        pd.DataFrame([[subset, "RF_cond", seed, r, s]]).to_csv(out, mode="a", header=False, index=False)
        print(f"{subset} seed {seed} k={k} RMSE={r:.2f} NASA={s:.1f} ({time.time()-t0:.0f}s)", flush=True)

d = pd.read_csv(out)
print(d.groupby("subset")[["rmse", "nasa_score"]].agg(["mean", "std"]))