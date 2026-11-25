import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def compute_variance_explained(subset="FD004"):
    train_raw = pd.read_csv(f"data_clean/train_{subset}.csv")
    op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    km = KMeans(n_clusters=6, random_state=0, n_init=10).fit(train_raw[op_cols])
    regimes = km.predict(train_raw[op_cols])
    
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    stds = train_raw[sensor_cols].std()
    active_sensors = [s for s in sensor_cols if stds[s] >= 1e-10]
    
    records = []
    for s in active_sensors:
        y = train_raw[s].values
        gm = np.mean(y)
        ss_tot = np.sum((y - gm) ** 2)
        ss_bet = sum(len(y[regimes == c]) * (np.mean(y[regimes == c]) - gm) ** 2 for c in range(6))
        eta2 = ss_bet / ss_tot if ss_tot > 0 else 0.0
        records.append({"sensor": s, "eta2": eta2})
    return pd.DataFrame(records)

if __name__ == "__main__":
    for sub in ["FD002", "FD004"]:
        df = compute_variance_explained(sub)
        print(f"[{sub}] Mean variance explained by regime: {df['eta2'].mean():.4f}")
