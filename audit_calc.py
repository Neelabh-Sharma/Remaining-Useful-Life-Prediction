import os, re
import numpy as np
import pandas as pd
from scipy import stats

print("=== RUNNING PHASE 0 AUDIT CALCULATIONS ===")

# 1. Audit Table 2 differences vs Bootstrap observed delta
df_stats = pd.read_csv("results/stats.csv")
print("\n--- 1. Stats CSV vs Table 2 Differences ---")
for _, r in df_stats.iterrows():
    sub = r['subset']
    diff_seeds = r['diff_rmse_seeds_mean']
    obs_diff = r['observed_diff_engine_avg']
    print(f"[{sub}] Mean of seed diffs: {diff_seeds:.4f} | Ensemble observed diff: {obs_diff:.4f}")

# 2. Audit RF Global vs RF Cond on FD001 / FD003
df_base = pd.read_csv("results/baseline_runs.csv")
df_rf_cond = pd.read_csv("results/rf_cond_runs.csv")
print("\n--- 2. RF Global vs RF Cond on FD001/FD003 ---")
for sub in ["FD001", "FD003"]:
    g_scores = df_base[(df_base['subset']==sub) & (df_base['model']=='RandomForest')].sort_values('seed')['rmse'].values
    c_scores = df_rf_cond[df_rf_cond['subset']==sub].sort_values('seed')['rmse'].values
    print(f"[{sub}] RF Global RMSEs: {g_scores}")
    print(f"[{sub}] RF Cond   RMSEs: {c_scores}")
    print(f"[{sub}] Differences: {c_scores - g_scores}")

# 3. Audit k=9 vs k=6 on FD004
df_extra = pd.read_csv("results/extra_runs.csv")
df_gru_v2 = pd.read_csv("results/gru_runs_v2.csv")
print("\n--- 3. k=6 vs k=9 on FD004 ---")
k6_scores = df_gru_v2[(df_gru_v2['subset']=='FD004') & (df_gru_v2['method']=='GRU_cond')].sort_values('seed')['rmse'].values
k9_scores = df_extra[(df_extra['exp_type']=='ablation') & (df_extra['k']==9.0)].sort_values('seed')['rmse'].values
for s in range(3):
    print(f"Seed {s}: k=6 RMSE = {k6_scores[s]:.6f} | k=9 RMSE = {k9_scores[s]:.6f} | Diff = {k9_scores[s] - k6_scores[s]:.6f}")
print(f"k=6 Mean: {np.mean(k6_scores):.4f} +/- {np.std(k6_scores, ddof=1):.4f}")
print(f"k=9 Mean: {np.mean(k9_scores):.4f} +/- {np.std(k9_scores, ddof=1):.4f}")

# 4. Audit Inertia in cluster_metrics
cm_fd2 = pd.read_csv("results/cluster_metrics_FD002.csv")
cm_fd4 = pd.read_csv("results/cluster_metrics_FD004.csv")
print("\n--- 4. Inertia & Silhouette Metrics ---")
print("FD002 k=2..7 inertia:")
print(cm_fd2[['k', 'inertia', 'silhouette']])
print("FD004 k=2..7 inertia:")
print(cm_fd4[['k', 'inertia', 'silhouette']])

# 5. Audit Tuning Seed Overlap and Test Leakage
df_tune = pd.read_csv("results/stage2_tuning.csv")
print("\n--- 5. Stage 2 Tuning ---")
print(df_tune[df_tune['subset']=='FD004'][['config', 'method', 'seed', 'val_rmse', 'test_rmse']])

# 6. Audit Statistical Significance Claims
print("\n--- 6. Significance Claims ---")
print(f"FD002 Seed-level paired t-test: t={df_stats.loc[df_stats['subset']=='FD002', 't_stat_seed'].values[0]:.4f}, p={df_stats.loc[df_stats['subset']=='FD002', 't_pval_seed'].values[0]:.4f}")
print(f"FD004 Seed-level paired t-test: t={df_stats.loc[df_stats['subset']=='FD004', 't_stat_seed'].values[0]:.4f}, p={df_stats.loc[df_stats['subset']=='FD004', 't_pval_seed'].values[0]:.4f}")
