import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Publication styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.autolayout'] = True

# ==============================================================================
# 1(a) KMeans Silhouette & Inertia for k=2..12 on FD002 and FD004
# ==============================================================================
print(">>> Starting 1(a): KMeans Silhouette & Inertia for k=2..12...", flush=True)

k_range = list(range(2, 13))
subsets_k = ["FD002", "FD004"]
k_results = []

for sub in subsets_k:
    df_tr = pd.read_csv(f"data_clean/train_{sub}.csv")
    X_op = df_tr[['op_setting_1', 'op_setting_2', 'op_setting_3']].values
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        km.fit(X_op)
        inertia = km.inertia_
        sil = silhouette_score(X_op, km.labels_, sample_size=10000, random_state=0)
        k_results.append({
            "subset": sub,
            "k": k,
            "inertia": inertia,
            "silhouette": sil
        })
        print(f"  [{sub}] k={k:2d} | Inertia: {inertia:12.2f} | Silhouette: {sil:.6f}", flush=True)

df_k_metrics = pd.DataFrame(k_results)

# Plot k_selection
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

for sub, color, marker in [("FD002", "#1b9e77", "o"), ("FD004", "#d95f02", "s")]:
    sub_data = df_k_metrics[df_k_metrics["subset"] == sub]
    ax1.plot(sub_data["k"], sub_data["silhouette"], marker=marker, color=color, linewidth=1.8, markersize=6, label=sub)
    ax2.plot(sub_data["k"], sub_data["inertia"], marker=marker, color=color, linewidth=1.8, markersize=6, label=sub)

ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Silhouette Score (sample size = 10,000)")
ax1.set_xticks(k_range)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True)
ax1.axvline(6, color="gray", linestyle="--", alpha=0.7, label="True Regimes (k=6)")

ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Inertia (Sum of Squared Distances)")
ax2.set_xticks(k_range)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True)
ax2.axvline(6, color="gray", linestyle="--", alpha=0.7)

fig.savefig("figures/k_selection.png", dpi=300)
fig.savefig("figures/k_selection.pdf")
plt.close(fig)
print("Saved figures/k_selection.png and .pdf", flush=True)


# ==============================================================================
# 1(b) Error by true-RUL bin (0-30, 30-60, 60-125, >125)
# ==============================================================================
print("\n>>> Starting 1(b): Error by true-RUL bin...", flush=True)

bins = [0, 30, 60, 125, np.inf]
bin_labels = ["0-30", "30-60", "60-125", ">125"]

bin_rows = []

for sub in ["FD002", "FD004"]:
    # Load all 3 seeds predictions for global and cond
    preds_g = {}
    preds_c = {}
    
    for s in [0, 1, 2]:
        df_g = pd.read_csv(f"results/preds/{sub}_G_kNone_f1.0_s{s}.csv")
        df_c = pd.read_csv(f"results/preds/{sub}_C_kNone_f1.0_s{s}.csv")
        preds_g[s] = df_g
        preds_c[s] = df_c
        
    y_true = preds_g[0]["true_RUL"].values
    
    # Assign each engine to a bin based on true_RUL
    bin_idx = np.digitize(y_true, bins, right=True) - 1
    # Note: digitize with right=True for bins=[0, 30, 60, 125, inf]:
    # <= 30: index 0
    # 30 < x <= 60: index 1
    # 60 < x <= 125: index 2
    # > 125: index 3
    
    for b_i, b_label in enumerate(bin_labels):
        mask = (bin_idx == b_i)
        n_eng = int(np.sum(mask))
        
        rmse_g_seeds = []
        rmse_c_seeds = []
        
        for s in [0, 1, 2]:
            y_t_bin = preds_g[s].loc[mask, "true_RUL"].values
            p_g_bin = preds_g[s].loc[mask, "pred"].values
            p_c_bin = preds_c[s].loc[mask, "pred"].values
            
            rg = np.sqrt(np.mean((y_t_bin - p_g_bin) ** 2)) if len(y_t_bin) > 0 else np.nan
            rc = np.sqrt(np.mean((y_t_bin - p_c_bin) ** 2)) if len(y_t_bin) > 0 else np.nan
            
            rmse_g_seeds.append(rg)
            rmse_c_seeds.append(rc)
            
        bin_rows.append({
            "subset": sub,
            "bin": b_label,
            "n_engines": n_eng,
            "rmse_global_s0": rmse_g_seeds[0],
            "rmse_global_s1": rmse_g_seeds[1],
            "rmse_global_s2": rmse_g_seeds[2],
            "rmse_global_mean": np.mean(rmse_g_seeds),
            "rmse_global_std": np.std(rmse_g_seeds, ddof=1),
            "rmse_cond_s0": rmse_c_seeds[0],
            "rmse_cond_s1": rmse_c_seeds[1],
            "rmse_cond_s2": rmse_c_seeds[2],
            "rmse_cond_mean": np.mean(rmse_c_seeds),
            "rmse_cond_std": np.std(rmse_c_seeds, ddof=1),
            "rmse_diff_mean": np.mean(rmse_c_seeds) - np.mean(rmse_g_seeds)
        })

df_bins = pd.DataFrame(bin_rows)
df_bins.to_csv("results/error_by_bin.csv", index=False)
print("Saved results/error_by_bin.csv", flush=True)

# Plot error by bin
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

for ax, sub in zip([ax1, ax2], ["FD002", "FD004"]):
    sub_df = df_bins[df_bins["subset"] == sub]
    x = np.arange(len(bin_labels))
    width = 0.35
    
    ax.bar(x - width/2, sub_df["rmse_global_mean"], width, yerr=sub_df["rmse_global_std"],
           capsize=4, label="GRU Global", color="#d95f02", edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.bar(x + width/2, sub_df["rmse_cond_mean"], width, yerr=sub_df["rmse_cond_std"],
           capsize=4, label="GRU Condition-Aware", color="#1b9e77", edgecolor="black", linewidth=0.8, alpha=0.9)
    
    ax.set_title(f"{sub} (Engines per bin: {list(sub_df['n_engines'])})", fontsize=11)
    ax.set_xlabel("True RUL Bin (Cycles)")
    ax.set_ylabel("Test RMSE (Cycles)")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True)

fig.savefig("figures/error_by_bin.png", dpi=300)
fig.savefig("figures/error_by_bin.pdf")
plt.close(fig)
print("Saved figures/error_by_bin.png and .pdf", flush=True)


# ==============================================================================
# 1(c) Statistical Tests: Global vs Cond on FD002 and FD004
# ==============================================================================
print("\n>>> Starting 1(c): Statistical Tests & Paired Bootstrap...", flush=True)

stat_rows = []
p_values_for_holm = []

for sub in ["FD002", "FD004"]:
    # Per-seed RMSE values (from gru_runs_v2.csv)
    df_runs = pd.read_csv("results/gru_runs_v2.csv")
    sub_g = df_runs[(df_runs["subset"] == sub) & (df_runs["method"] == "GRU_global")].sort_values("seed")
    sub_c = df_runs[(df_runs["subset"] == sub) & (df_runs["method"] == "GRU_cond")].sort_values("seed")
    
    rmse_g = sub_g["rmse"].values # length 3
    rmse_c = sub_c["rmse"].values # length 3
    diff_seeds = rmse_c - rmse_g
    
    # 1. Paired t-test on per-seed RMSE
    t_stat, t_pval = stats.ttest_rel(rmse_c, rmse_g)
    
    # 2. Paired Wilcoxon on per-seed RMSE
    # With n=3, wilcoxon stat is min sum of ranks
    try:
        w_res = stats.wilcoxon(rmse_c, rmse_g)
        w_stat, w_pval = w_res.statistic, w_res.pvalue
    except Exception as e:
        w_stat, w_pval = np.nan, np.nan
        
    # 3. Paired bootstrap over test engines (10,000 resamples, seeds averaged)
    # Load all 3 seeds per-engine predictions
    preds_g_all = []
    preds_c_all = []
    for s in [0, 1, 2]:
        dg = pd.read_csv(f"results/preds/{sub}_G_kNone_f1.0_s{s}.csv")
        dc = pd.read_csv(f"results/preds/{sub}_C_kNone_f1.0_s{s}.csv")
        preds_g_all.append(dg["pred"].values)
        preds_c_all.append(dc["pred"].values)
        
    y_true = dg["true_RUL"].values
    n_eng = len(y_true)
    
    # Average predictions over the 3 seeds
    p_g_avg = np.mean(preds_g_all, axis=0)
    p_c_avg = np.mean(preds_c_all, axis=0)
    
    # Engine-level paired t-test and Wilcoxon on squared errors
    sq_err_g = (p_g_avg - y_true) ** 2
    sq_err_c = (p_c_avg - y_true) ** 2
    t_eng_stat, t_eng_pval = stats.ttest_rel(sq_err_c, sq_err_g)
    w_eng_res = stats.wilcoxon(sq_err_c, sq_err_g)
    w_eng_stat, w_eng_pval = w_eng_res.statistic, w_eng_res.pvalue
    
    # Bootstrap 95% CI of RMSE difference (RMSE_cond - RMSE_global)
    rng = np.random.RandomState(0)
    B = 10000
    boot_diffs = np.empty(B)
    
    for b in range(B):
        idx = rng.randint(0, n_eng, size=n_eng)
        y_b = y_true[idx]
        pg_b = p_g_avg[idx]
        pc_b = p_c_avg[idx]
        
        rmse_g_b = np.sqrt(np.mean((pg_b - y_b) ** 2))
        rmse_c_b = np.sqrt(np.mean((pc_b - y_b) ** 2))
        boot_diffs[b] = rmse_c_b - rmse_g_b
        
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    observed_diff = np.sqrt(np.mean((p_c_avg - y_true) ** 2)) - np.sqrt(np.mean((p_g_avg - y_true) ** 2))
    
    stat_rows.append({
        "subset": sub,
        "n_seeds": len(rmse_g),
        "n_engines": n_eng,
        "mean_rmse_global": np.mean(rmse_g),
        "mean_rmse_cond": np.mean(rmse_c),
        "diff_rmse_seeds_mean": np.mean(diff_seeds),
        "t_stat_seed": t_stat,
        "t_pval_seed": t_pval,
        "wilcoxon_stat_seed": w_stat,
        "wilcoxon_pval_seed": w_pval,
        "observed_diff_engine_avg": observed_diff,
        "boot_95ci_lower": ci_lower,
        "boot_95ci_upper": ci_upper,
        "t_stat_engine": t_eng_stat,
        "t_pval_engine": t_eng_pval,
        "wilcoxon_stat_engine": w_eng_stat,
        "wilcoxon_pval_engine": w_eng_pval
    })
    p_values_for_holm.append((sub, t_eng_pval, w_eng_pval, t_pval, w_pval))

df_stats = pd.DataFrame(stat_rows)

# Holm correction over the 2 subsets for engine-level and seed-level tests
# Order p-values from smallest to largest
p_eng_t = [df_stats.loc[df_stats["subset"]==s, "t_pval_engine"].values[0] for s in ["FD004", "FD002"]]
# Holm-Bonferroni: p_adj[0] = min(1, p[0]*2), p_adj[1] = min(1, max(p_adj[0], p[1]*1))
p_order = np.argsort(p_eng_t)
holm_t_eng = np.empty(2)
holm_t_eng[p_order[0]] = min(1.0, p_eng_t[p_order[0]] * 2)
holm_t_eng[p_order[1]] = min(1.0, max(holm_t_eng[p_order[0]], p_eng_t[p_order[1]] * 1))

df_stats["holm_adj_t_pval_engine"] = [holm_t_eng[1], holm_t_eng[0]] # FD002, FD004

df_stats.to_csv("results/stats.csv", index=False)
print("Saved results/stats.csv", flush=True)


# ==============================================================================
# 1(d) Per-seed table of all main results (results/per_seed.md & .tex)
# ==============================================================================
print("\n>>> Starting 1(d): Per-seed table of all main results...", flush=True)

df_gru_all = pd.read_csv("results/gru_runs_v2.csv")
df_rf_cond = pd.read_csv("results/rf_cond_runs.csv")
df_base = pd.read_csv("results/baseline_runs.csv")

# Combine all main runs into one unified dataframe
unified_rows = []

for _, r in df_base.iterrows():
    m = r['model']
    unified_rows.append({
        "subset": r['subset'],
        "method": f"{m}_global",
        "seed": str(r['seed']),
        "rmse": float(r['rmse']),
        "nasa_score": float(r['nasa_score']),
        "epochs": 1 if m == "Ridge" else "N/A"
    })

for _, r in df_rf_cond.iterrows():
    unified_rows.append({
        "subset": r['subset'],
        "method": "RF_cond",
        "seed": str(int(r['seed'])),
        "rmse": float(r['rmse']),
        "nasa_score": float(r['nasa_score']),
        "epochs": "N/A"
    })

for _, r in df_gru_all.iterrows():
    unified_rows.append({
        "subset": r['subset'],
        "method": r['method'],
        "seed": str(int(r['seed'])),
        "rmse": float(r['rmse']),
        "nasa_score": float(r['nasa_score']),
        "epochs": int(r['epochs'])
    })

df_unified = pd.DataFrame(unified_rows)
# Sort systematically
def parse_seed(x):
    if pd.isna(x) or str(x).lower() in ['none', 'nan', '']:
        return -1
    try:
        return int(float(x))
    except:
        return -1

df_unified['seed_sort'] = df_unified['seed'].apply(parse_seed)
df_unified = df_unified.sort_values(by=["subset", "method", "seed_sort"]).drop(columns=['seed_sort'])

# Markdown table
md_lines = []
md_lines.append("# Per-Seed Main Benchmark Results\n")
md_lines.append("| Subset | Method | Seed | Test RMSE | NASA Score | Epochs |")
md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
for _, r in df_unified.iterrows():
    ep_str = str(r['epochs'])
    seed_str = str(r['seed']) if str(r['seed']).lower() not in ['none', 'nan'] else '--'
    md_lines.append(f"| {r['subset']} | {r['method']} | {seed_str} | {r['rmse']:.4f} | {r['nasa_score']:.1f} | {ep_str} |")

with open("results/per_seed.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print("Saved results/per_seed.md", flush=True)

# LaTeX table
tex_lines = []
tex_lines.append("% Per-seed main results table")
tex_lines.append("\\begin{table*}[t]")
tex_lines.append("\\centering")
tex_lines.append("\\caption{Per-seed evaluation results across all main benchmark methods and subsets.}")
tex_lines.append("\\label{tab:per_seed_results}")
tex_lines.append("\\small")
tex_lines.append("\\begin{tabular}{lllccc}")
tex_lines.append("\\toprule")
tex_lines.append("\\textbf{Subset} & \\textbf{Method} & \\textbf{Seed} & \\textbf{Test RMSE} & \\textbf{NASA Score} & \\textbf{Epochs} \\\\")
tex_lines.append("\\midrule")

current_sub = None
for _, r in df_unified.iterrows():
    if current_sub is not None and r['subset'] != current_sub:
        tex_lines.append("\\midrule")
    current_sub = r['subset']
    method_fmt = r['method'].replace('_', '\\_')
    seed_fmt = str(r['seed']) if str(r['seed']).lower() not in ['none', 'nan'] else '--'
    tex_lines.append(f"{r['subset']} & {method_fmt} & {seed_fmt} & {r['rmse']:.4f} & {r['nasa_score']:.1f} & {r['epochs']} \\\\")

tex_lines.append("\\bottomrule")
tex_lines.append("\\end{tabular}")
tex_lines.append("\\end{table*}")

with open("results/per_seed.tex", "w") as f:
    f.write("\n".join(tex_lines) + "\n")
print("Saved results/per_seed.tex", flush=True)

print("\n>>> STAGE 1 COMPLETED SUCCESSFULLY! <<<", flush=True)
