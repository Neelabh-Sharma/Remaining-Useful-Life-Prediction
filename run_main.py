import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from preprocess import get_preprocessed_data
from utils import make_train_windows, make_last_window, rmse, nasa_score
from train import train_gru, set_seed

def get_existing_runs(runs_csv_path):
    if not os.path.exists(runs_csv_path):
        return set(), []
    try:
        df = pd.read_csv(runs_csv_path)
        existing = set()
        for _, row in df.iterrows():
            existing.add((str(row['subset']), str(row['method']), int(row['seed'])))
        return existing, df.to_dict('records')
    except Exception:
        return set(), []

def append_run(runs_csv_path, record):
    df_row = pd.DataFrame([record])
    if not os.path.exists(runs_csv_path):
        df_row.to_csv(runs_csv_path, index=False)
    else:
        df_row.to_csv(runs_csv_path, mode='a', header=False, index=False)

def run_rf(subset, mode, seed):
    """
    Runs Random Forest with condition-aware normalization (Mode C).
    """
    t0 = time.time()
    train_df, test_df, feature_cols, _, _ = get_preprocessed_data(subset, mode=mode)
    test_rul_df = pd.read_csv(f"data_clean/test_RUL_{subset}.csv")

    X_train, y_train = make_train_windows(train_df, feature_cols, window=30, stride=5)
    X_test, test_units = make_last_window(test_df, feature_cols, window=30)

    y_test = np.array([test_rul_df.loc[test_rul_df['unit_number'] == u, 'true_RUL'].values[0] for u in test_units], dtype=float)

    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    preds = np.clip(preds, 0, None)

    te_rmse = rmse(y_test, preds)
    te_nasa = nasa_score(y_test, preds)
    elapsed = time.time() - t0
    return te_rmse, te_nasa, elapsed

def run_pipeline(target_subsets=None):
    res_dir = "results"
    os.makedirs(res_dir, exist_ok=True)
    runs_csv_path = os.path.join(res_dir, "main_runs.csv")

    if target_subsets is None:
        target_subsets = ["FD004", "FD002", "FD001", "FD003"]

    existing_keys, _ = get_existing_runs(runs_csv_path)
    print("=" * 75, flush=True)
    print(f"MAIN RUNS PIPELINE: Target Subsets = {target_subsets}", flush=True)
    print(f"Initial check: {len(existing_keys)} runs already in {runs_csv_path}.", flush=True)
    print("=" * 75, flush=True)

    seeds = [0, 1, 2]
    methods = [
        ('GRU_global', 'G', 'gru'),
        ('GRU_condition_aware', 'C', 'gru'),
        ('RF_condition_aware', 'C', 'rf')
    ]

    total_start = time.time()

    for sub in target_subsets:
        sub_start = time.time()
        print(f"\n==================================================", flush=True)
        print(f">>> STARTING SUBSET: {sub} <<<", flush=True)
        print(f"==================================================", flush=True)

        for method_name, norm_mode, model_type in methods:
            for seed in seeds:
                run_key = (sub, method_name, seed)
                if run_key in existing_keys:
                    print(f"[{sub}] {method_name} (seed {seed}) ALREADY COMPLETED. Skipping.", flush=True)
                    continue

                t_run_0 = time.time()
                print(f"[{sub}] Running {method_name} | Seed {seed} ...", flush=True)

                if model_type == 'gru':
                    te_rmse, te_nasa, elapsed, epochs = train_gru(
                        subset=sub,
                        mode=norm_mode,
                        seed=seed,
                        max_epochs=40,
                        patience=6,
                        batch_size=256,
                        lr=1e-3,
                        verbose=False
                    )
                    run_time = time.time() - t_run_0
                    print(f"  -> [{sub}] {method_name} seed {seed} finished in {run_time:.2f}s ({epochs} epochs) | RMSE: {te_rmse:.4f} | NASA: {te_nasa:.1f}", flush=True)
                else: # rf
                    te_rmse, te_nasa, elapsed = run_rf(subset=sub, mode=norm_mode, seed=seed)
                    run_time = time.time() - t_run_0
                    print(f"  -> [{sub}] {method_name} seed {seed} finished in {run_time:.2f}s | RMSE: {te_rmse:.4f} | NASA: {te_nasa:.1f}", flush=True)

                record = {
                    'subset': sub,
                    'method': method_name,
                    'seed': seed,
                    'rmse': te_rmse,
                    'nasa_score': te_nasa
                }
                append_run(runs_csv_path, record)
                existing_keys.add(run_key)

        print(f"\n--- SUBSET {sub} COMPLETED in {time.time()-sub_start:.2f}s ---", flush=True)
        print_subset_summary(runs_csv_path, sub)

    total_elapsed = time.time() - total_start
    print(f"\nAll specified subsets {target_subsets} finished in {total_elapsed:.2f}s ({total_elapsed/60:.2f} min).", flush=True)

def print_subset_summary(runs_csv_path, subset):
    if not os.path.exists(runs_csv_path):
        return
    df = pd.read_csv(runs_csv_path)
    sub_df = df[df['subset'] == subset]
    if len(sub_df) == 0:
        return

    print(f"\nSummary for {subset}:", flush=True)
    for m in ['GRU_global', 'GRU_condition_aware', 'RF_condition_aware']:
        m_df = sub_df[sub_df['method'] == m]
        if len(m_df) > 0:
            m_rmse = m_df['rmse'].mean()
            s_rmse = m_df['rmse'].std()
            m_nasa = m_df['nasa_score'].mean()
            s_nasa = m_df['nasa_score'].std()
            print(f"  {m:<22}: RMSE = {m_rmse:.2f} ± {s_rmse:.2f} | NASA = {m_nasa:.1f} ± {s_nasa:.1f} (seeds: {list(m_df['seed'])})", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=["FD004", "FD002", "FD001", "FD003"], help="List of subsets to run in order")
    args = parser.parse_args()

    run_pipeline(target_subsets=args.subsets)
