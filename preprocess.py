import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_preprocessed_data(subset, mode='G', data_dir='data_clean', k_override=None):
    """
    Loads train and test for a given subset (FD001-FD004) and normalizes features.
    Modes:
      (G) Global: StandardScaler fit on TRAIN only.
      (C) Condition-aware: KMeans (k=6 for FD002/FD004, k=1 for FD001/FD003, random_state=0)
          fit on the 3 operating settings of TRAIN only. Assign every row (train and test)
          to its nearest cluster. Standardize each sensor separately inside each cluster
          using TRAIN statistics of that cluster. For FD001/FD003 (k=1), C equals G.
    k_override: if given, replaces the default number of clusters (used for the ablation).
    Drops sensors that are constant within that subset's train data.
    Returns:
      train_df_proc, test_df_proc, feature_cols, k, len(active_sensors)
    """
    train_path = os.path.join(data_dir, f"train_{subset}.csv")
    test_path = os.path.join(data_dir, f"test_{subset}.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Drop sensors that are constant within train data (std < 1e-10)
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    stds = train_df[sensor_cols].std()
    const_sensors = stds[stds < 1e-10].index.tolist()
    active_sensors = [s for s in sensor_cols if s not in const_sensors]

    op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    feature_cols = op_cols + active_sensors

    train_proc = train_df.copy()
    test_proc = test_df.copy()

    # Ensure float64 precision
    train_proc[feature_cols] = train_proc[feature_cols].astype(np.float64)
    test_proc[feature_cols] = test_proc[feature_cols].astype(np.float64)

    k = 1 if subset in ['FD001', 'FD003'] else 6
    if k_override is not None:
        k = k_override

    if mode == 'G':
        # Global StandardScaler fit on train only
        scaler = StandardScaler()
        scaler.fit(train_proc[feature_cols])
        train_proc[feature_cols] = scaler.transform(train_proc[feature_cols])
        test_proc[feature_cols] = scaler.transform(test_proc[feature_cols])

    elif mode == 'C':
        # Condition-aware normalization
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(train_df[op_cols])

        train_clusters = kmeans.predict(train_df[op_cols])
        test_clusters = kmeans.predict(test_df[op_cols])

        # Standardize op settings globally
        op_scaler = StandardScaler()
        op_scaler.fit(train_proc[op_cols])
        train_proc[op_cols] = op_scaler.transform(train_proc[op_cols])
        test_proc[op_cols] = op_scaler.transform(test_proc[op_cols])

        # Standardize each sensor separately inside each cluster using TRAIN statistics
        for s in active_sensors:
            for c in range(k):
                tr_mask = (train_clusters == c)
                te_mask = (test_clusters == c)

                c_mean = train_df.loc[tr_mask, s].mean()
                c_std = train_df.loc[tr_mask, s].std()
                if c_std < 1e-10 or np.isnan(c_std):
                    c_std = 1.0

                train_proc.loc[tr_mask, s] = (train_df.loc[tr_mask, s].astype(float) - c_mean) / c_std
                test_proc.loc[te_mask, s] = (test_df.loc[te_mask, s].astype(float) - c_mean) / c_std
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'G' or 'C'.")

    return train_proc, test_proc, feature_cols, k, len(active_sensors)

def print_preprocessing_summary():
    print("=" * 70)
    print("TASK 1: PREPROCESSING SUMMARY PER SUBSET")
    print("=" * 70)
    print(f"{'Subset':<10} | {'Mode C Clusters (k)':<20} | {'Sensors Kept':<15} | {'Constant Sensors Dropped'}")
    print("-" * 70)
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    for sub in subsets:
        tr_df = pd.read_csv(f"data_clean/train_{sub}.csv")
        s_cols = [f"sensor_{i}" for i in range(1, 22)]
        stds = tr_df[s_cols].std()
        const = stds[stds < 1e-10].index.tolist()
        k = 1 if sub in ['FD001', 'FD003'] else 6
        sensors_kept = 21 - len(const)
        print(f"{sub:<10} | {k:<20} | {sensors_kept:<15} | {const}")
    print("=" * 70)

if __name__ == "__main__":
    print_preprocessing_summary()
    print("\nVerifying data consistency across modes G and C...")
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for m in ["G", "C"]:
            tr, te, feats, k_val, n_sensors = get_preprocessed_data(sub, mode=m)
            assert not tr[feats].isna().any().any(), f"NaNs found in train {sub} mode {m}"
            assert not te[feats].isna().any().any(), f"NaNs found in test {sub} mode {m}"
    print("All preprocessing verifications passed successfully!")