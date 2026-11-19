import os
import time
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from utils import nasa_score, rmse, make_last_window, make_train_windows

def get_existing_runs(runs_csv_path):
    if not os.path.exists(runs_csv_path):
        return set(), []
    try:
        df = pd.read_csv(runs_csv_path)
        existing = set()
        for _, row in df.iterrows():
            seed_val = str(row['seed'])
            existing.add((row['subset'], row['model'], seed_val))
        return existing, df.to_dict('records')
    except Exception:
        return set(), []

def append_run(runs_csv_path, run_record):
    df_row = pd.DataFrame([run_record])
    if not os.path.exists(runs_csv_path):
        df_row.to_csv(runs_csv_path, index=False)
    else:
        df_row.to_csv(runs_csv_path, mode='a', header=False, index=False)

def run_baseline():
    total_start_time = time.time()
    print("=" * 80, flush=True)
    print("TASK 3 & 4: BASELINE MODELS & EVALUATION FOR NASA C-MAPSS", flush=True)
    print("Configuration: RF n_estimators=100, max_depth=20, seeds=[0, 1, 2], stride=5", flush=True)
    print("=" * 80, flush=True)

    data_dir = "data_clean"
    res_dir = "results"
    fig_dir = "figures"
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    rf_seeds = [0, 1, 2]
    
    runs_csv_path = os.path.join(res_dir, "baseline_runs.csv")
    existing_keys, all_runs = get_existing_runs(runs_csv_path)
    print(f"Initial check: {len(existing_keys)} runs already recorded in {runs_csv_path}.", flush=True)

    rf_seed0_preds = {} # For Task 4 figures

    for sub in subsets:
        print(f"\n>>> SUBSET: {sub} <<<", flush=True)
        sub_start_time = time.time()
        
        train_path = os.path.join(data_dir, f"train_{sub}.csv")
        test_path = os.path.join(data_dir, f"test_{sub}.csv")
        rul_path = os.path.join(data_dir, f"test_RUL_{sub}.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        test_rul_df = pd.read_csv(rul_path)

        # 1. Identify and drop constant sensors in train
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
        stds = train_df[sensor_cols].std()
        const_sensors = stds[stds < 1e-10].index.tolist()
        active_sensors = [s for s in sensor_cols if s not in const_sensors]
        feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + active_sensors
        print(f"  Features ({len(feature_cols)}): dropped {len(const_sensors)} constant sensors {const_sensors}", flush=True)

        # 2. Standardize features using train set only
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols])

        train_df_scaled = train_df.copy()
        train_df_scaled[feature_cols] = scaler.transform(train_df[feature_cols])

        test_df_scaled = test_df.copy()
        test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

        # 3. Create windows (window=30, stride=5)
        t_w0 = time.time()
        X_train, y_train = make_train_windows(train_df_scaled, feature_cols, window=30, stride=5)
        X_test, test_units = make_last_window(test_df_scaled, feature_cols, window=30)
        print(f"  Windows prepared in {time.time()-t_w0:.2f}s | Train X: {X_train.shape} | Test X: {X_test.shape}", flush=True)

        # Ground truth aligned by test_units
        y_test = [test_rul_df.loc[test_rul_df['unit_number'] == u, 'true_RUL'].values[0] for u in test_units]
        y_test = np.array(y_test, dtype=float)

        # 4. Model (A): Ridge Regression
        ridge_key = (sub, 'Ridge', 'None')
        if ridge_key in existing_keys:
            print(f"  [Ridge] (subset={sub}) ALREADY RUN. Skipping.", flush=True)
        else:
            t_r0 = time.time()
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_train, y_train)
            y_pred_ridge = np.clip(ridge.predict(X_test), 0, None)
            r_rmse = rmse(y_test, y_pred_ridge)
            r_nasa = nasa_score(y_test, y_pred_ridge)
            elapsed = time.time() - t_r0
            print(f"  [Ridge] elapsed: {elapsed:.2f}s | RMSE: {r_rmse:.4f} | NASA Score: {r_nasa:.1f}", flush=True)
            record = {
                'subset': sub,
                'model': 'Ridge',
                'seed': 'None',
                'rmse': r_rmse,
                'nasa_score': r_nasa
            }
            append_run(runs_csv_path, record)
            existing_keys.add(ridge_key)
            all_runs.append(record)

        # 5. Model (B): Random Forest (seeds 0, 1, 2)
        for seed in rf_seeds:
            rf_key = (sub, 'RandomForest', str(seed))
            if rf_key in existing_keys:
                print(f"  [RandomForest] (subset={sub}, seed={seed}) ALREADY RUN. Skipping.", flush=True)
                # If seed == 0 and needed for figure, we can fit or load; let's fit if seed==0 and not yet stored
                if seed == 0 and sub in ['FD001', 'FD004'] and sub not in rf_seed0_preds:
                    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0, n_jobs=-1)
                    rf.fit(X_train, y_train)
                    rf_seed0_preds[sub] = (y_test, np.clip(rf.predict(X_test), 0, None))
            else:
                t_rf0 = time.time()
                rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1)
                rf.fit(X_train, y_train)
                y_pred_rf = np.clip(rf.predict(X_test), 0, None)
                rf_rmse = rmse(y_test, y_pred_rf)
                rf_nasa = nasa_score(y_test, y_pred_rf)
                elapsed = time.time() - t_rf0
                print(f"  [RandomForest seed={seed}] elapsed: {elapsed:.2f}s | RMSE: {rf_rmse:.4f} | NASA Score: {rf_nasa:.1f}", flush=True)
                record = {
                    'subset': sub,
                    'model': 'RandomForest',
                    'seed': str(seed),
                    'rmse': rf_rmse,
                    'nasa_score': rf_nasa
                }
                append_run(runs_csv_path, record)
                existing_keys.add(rf_key)
                all_runs.append(record)
                
                if seed == 0 and sub in ['FD001', 'FD004']:
                    rf_seed0_preds[sub] = (y_test, y_pred_rf)

        print(f"Subset {sub} completed in {time.time()-sub_start_time:.2f}s", flush=True)

    # Reload all runs to ensure exact state
    runs_df = pd.read_csv(runs_csv_path)

    # Compute Summary Table
    summary_rows = []
    for sub in subsets:
        # Ridge
        r_df = runs_df[(runs_df['subset'] == sub) & (runs_df['model'] == 'Ridge')]
        r_rmse = r_df['rmse'].iloc[0]
        r_nasa = r_df['nasa_score'].iloc[0]
        summary_rows.append({
            'Subset': sub,
            'Model': 'Ridge',
            'RMSE (Formatted)': f"{r_rmse:.2f}",
            'NASA (Formatted)': f"{r_nasa:.1f}",
            'rmse_mean': r_rmse,
            'rmse_std': 0.0
        })

        # Random Forest
        rf_df = runs_df[(runs_df['subset'] == sub) & (runs_df['model'] == 'RandomForest')]
        rf_rmse_m = rf_df['rmse'].mean()
        rf_rmse_s = rf_df['rmse'].std()
        rf_nasa_m = rf_df['nasa_score'].mean()
        rf_nasa_s = rf_df['nasa_score'].std()
        summary_rows.append({
            'Subset': sub,
            'Model': 'Random Forest',
            'RMSE (Formatted)': f"{rf_rmse_m:.2f} ± {rf_rmse_s:.2f}",
            'NASA (Formatted)': f"{rf_nasa_m:.1f} ± {rf_nasa_s:.1f}",
            'rmse_mean': rf_rmse_m,
            'rmse_std': rf_rmse_s
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80, flush=True)
    print("BASELINE SUMMARY TABLE", flush=True)
    print("=" * 80, flush=True)
    print(summary_df[['Subset', 'Model', 'RMSE (Formatted)', 'NASA (Formatted)']].to_string(index=False), flush=True)

    # Save results/baseline_summary.md
    summary_md_path = os.path.join(res_dir, "baseline_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# NASA C-MAPSS Baseline Benchmark Summary\n\n")
        f.write("Evaluation on test engine final cycles (window=30, stride=5, target clipped at 125, RF n_estimators=100, max_depth=20, seeds 0, 1, 2).\n\n")
        f.write("| Subset | Model | Test RMSE | Test NASA Score |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for _, row in summary_df.iterrows():
            f.write(f"| {row['Subset']} | {row['Model']} | {row['RMSE (Formatted)']} | {row['NASA (Formatted)']} |\n")
        f.write("\n*Notes: Random Forest metrics are reported as mean ± std across 3 random seeds (0, 1, 2). Ridge regression is deterministic.*\n")
    print(f"Saved {summary_md_path}", flush=True)

    # TASK 4: SANITY FIGURES
    print("\n--- GENERATING SANITY FIGURES ---", flush=True)
    
    # 1. Scatter plots: predicted vs true RUL for RF seed 0 on FD001 and FD004
    # Ensure predictions are available for FD001 and FD004
    for sub in ["FD001", "FD004"]:
        if sub not in rf_seed0_preds:
            train_df = pd.read_csv(os.path.join(data_dir, f"train_{sub}.csv"))
            test_df = pd.read_csv(os.path.join(data_dir, f"test_{sub}.csv"))
            test_rul_df = pd.read_csv(os.path.join(data_dir, f"test_RUL_{sub}.csv"))
            sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
            stds = train_df[sensor_cols].std()
            active_sensors = [s for s in sensor_cols if s not in stds[stds < 1e-10].index.tolist()]
            feats = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + active_sensors
            scaler = StandardScaler().fit(train_df[feats])
            tr_s = train_df.copy(); tr_s[feats] = scaler.transform(train_df[feats])
            te_s = test_df.copy(); te_s[feats] = scaler.transform(test_df[feats])
            X_tr, y_tr = make_train_windows(tr_s, feats, window=30, stride=5)
            X_te, t_u = make_last_window(te_s, feats, window=30)
            y_te = np.array([test_rul_df.loc[test_rul_df['unit_number'] == u, 'true_RUL'].values[0] for u in t_u], dtype=float)
            rf0 = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
            rf_seed0_preds[sub] = (y_te, np.clip(rf0.predict(X_te), 0, None))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, target_sub in enumerate(["FD001", "FD004"]):
        ax = axes[idx]
        y_true, y_pred = rf_seed0_preds[target_sub]
        sub_rmse = rmse(y_true, y_pred)
        sub_nasa = nasa_score(y_true, y_pred)
        color = 'teal' if target_sub == 'FD001' else 'crimson'
        ax.scatter(y_true, y_pred, alpha=0.75, color=color, edgecolors='k', s=55, label='Test Engines')
        max_val = max(np.max(y_true), np.max(y_pred)) + 10
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Ideal Fit (y = x)')
        ax.set_title(f"Random Forest (Seed 0) on {target_sub}\nRMSE: {sub_rmse:.2f} | NASA Score: {sub_nasa:.1f}", fontsize=13)
        ax.set_xlabel("True Remaining Useful Life (RUL)", fontsize=11)
        ax.set_ylabel("Predicted Remaining Useful Life (RUL)", fontsize=11)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.tight_layout()
    scatter_path = os.path.join(fig_dir, "rf_scatter_FD001_FD004.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {scatter_path}", flush=True)

    # 2. Bar chart of RMSE per subset for both models
    sub_labels = subsets
    ridge_rmses = [runs_df[(runs_df['subset'] == s) & (runs_df['model'] == 'Ridge')]['rmse'].iloc[0] for s in sub_labels]
    rf_rmses = [runs_df[(runs_df['subset'] == s) & (runs_df['model'] == 'RandomForest')]['rmse'].mean() for s in sub_labels]
    rf_stds = [runs_df[(runs_df['subset'] == s) & (runs_df['model'] == 'RandomForest')]['rmse'].std() for s in sub_labels]

    x = np.arange(len(sub_labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width/2, ridge_rmses, width, label='Ridge Regression', color='steelblue', edgecolor='black')
    rects2 = plt.bar(x + width/2, rf_rmses, width, yerr=rf_stds, capsize=5, label='Random Forest (Mean ± Std)', color='forestgreen', edgecolor='black')

    plt.xlabel('C-MAPSS Subset', fontsize=12)
    plt.ylabel('Test RMSE (Cycles)', fontsize=12)
    plt.title('Baseline Test RMSE Comparison Across C-MAPSS Subsets', fontsize=14)
    plt.xticks(x, sub_labels, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)

    for rect in rects1:
        h = rect.get_height()
        plt.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        plt.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    barchart_path = os.path.join(fig_dir, "rmse_comparison_barchart.png")
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {barchart_path}", flush=True)

    total_elapsed = time.time() - total_start_time
    print(f"\nAll tasks finished in {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes).", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_baseline()
