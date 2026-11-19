import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 1. Print per-seed RMSE and NASA for FD004 GRU_cond at k=3, 6, 9
df_gru = pd.read_csv("results/gru_runs.csv")
df_extra = pd.read_csv("results/extra_runs.csv")

print("=" * 80)
print("1. PER-SEED RMSE AND NASA FOR FD004 GRU_cond (k=3, 6, 9)")
print("=" * 80)

k3 = df_extra[(df_extra['exp_type'] == 'ablation') & (df_extra['k'] == 3.0)]
k6 = df_gru[(df_gru['subset'] == 'FD004') & (df_gru['method'] == 'GRU_cond')]
k9 = df_extra[(df_extra['exp_type'] == 'ablation') & (df_extra['k'] == 9.0)]

print("\n--- k=3 (from results/extra_runs.csv) ---")
for _, r in k3.iterrows():
    print(f"Seed {int(r['seed'])}: RMSE = {r['rmse']:.6f}, NASA = {r['nasa_score']:.2f}, Epochs = {int(r['epochs'])}, Time = {r['seconds']:.1f}s")
print(f"Mean ± Std: RMSE = {k3['rmse'].mean():.6f} ± {k3['rmse'].std():.6f}, NASA = {k3['nasa_score'].mean():.2f} ± {k3['nasa_score'].std():.2f}")

print("\n--- k=6 (from results/gru_runs.csv) ---")
for _, r in k6.iterrows():
    print(f"Seed {int(r['seed'])}: RMSE = {r['rmse']:.6f}, NASA = {r['nasa_score']:.2f}, Epochs = {int(r['epochs'])}, Time = {r['seconds']:.1f}s")
print(f"Mean ± Std: RMSE = {k6['rmse'].mean():.6f} ± {k6['rmse'].std():.6f}, NASA = {k6['nasa_score'].mean():.2f} ± {k6['nasa_score'].std():.2f}")

print("\n--- k=9 (from results/extra_runs.csv) ---")
for _, r in k9.iterrows():
    print(f"Seed {int(r['seed'])}: RMSE = {r['rmse']:.6f}, NASA = {r['nasa_score']:.2f}, Epochs = {int(r['epochs'])}, Time = {r['seconds']:.1f}s")
print(f"Mean ± Std: RMSE = {k9['rmse'].mean():.6f} ± {k9['rmse'].std():.6f}, NASA = {k9['nasa_score'].mean():.2f} ± {k9['nasa_score'].std():.2f}")

# 2. Distinct operating conditions
tr = pd.read_csv("data_clean/train_FD004.csv")[['op_setting_1', 'op_setting_2', 'op_setting_3']]
tr_rounded = tr.round(1)
distinct_ops = tr_rounded.value_counts().reset_index()
print("\n" + "=" * 80)
print(f"2. DISTINCT OPERATING CONDITIONS (ROUNDED TO 1 DECIMAL): {len(distinct_ops)} COMBINATIONS")
print("=" * 80)
print(distinct_ops.to_string(index=False))

# 3. KMeans k=6 and k=9
print("\n" + "=" * 80)
print("3. KMEANS(random_state=0, n_init=10) CLUSTERS ON FD004 TRAIN OP SETTINGS")
print("=" * 80)

km6 = KMeans(n_clusters=6, random_state=0, n_init=10).fit(tr)
c6 = pd.Series(km6.labels_).value_counts().sort_index()
print("\n--- k=6 Clusters ---")
for i in range(6):
    center = km6.cluster_centers_[i]
    print(f"Cluster {i}: size = {c6[i]:5d} ({c6[i]/len(tr)*100:5.2f}%), center = [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.1f}]")

km9 = KMeans(n_clusters=9, random_state=0, n_init=10).fit(tr)
c9 = pd.Series(km9.labels_).value_counts().sort_index()
print("\n--- k=9 Clusters ---")
for i in range(9):
    center = km9.cluster_centers_[i]
    print(f"Cluster {i}: size = {c9[i]:5d} ({c9[i]/len(tr)*100:5.2f}%), center = [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.1f}]")

# Cross-tabulate k=6 vs k=9 assignments
print("\n--- Cross-tabulation: Which k=6 clusters are split by k=9? ---")
ct = pd.crosstab(pd.Series(km6.labels_, name="k=6 cluster"), pd.Series(km9.labels_, name="k=9 cluster"))
print(ct.to_string())

