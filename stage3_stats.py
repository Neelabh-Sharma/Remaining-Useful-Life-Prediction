import os
import numpy as np
import pandas as pd
from scipy import stats

PREDS_DIR = "results/preds"
V3_CSV = "results/gru_runs_v3.csv"
V2_CSV = "results/gru_runs_v2.csv"
STATS_V3_CSV = "results/stats_v3.csv"
STATS_NOTES_MD = "results/stats_notes.md"

df_v3 = pd.read_csv(V3_CSV)
df_v2 = pd.read_csv(V2_CSV)

print("Computing STAGE 3 Statistics...", flush=True)

rows_primary = []
p_vals_primary = {}

# 1. Primary Comparison: Global h=128 vs Cond h=64 across 10 seeds (0-9)
for sub in ["FD002", "FD004"]:
    sub_v3 = df_v3[df_v3["subset"] == sub]
    
    g_runs = sub_v3[(sub_v3["method"] == "GRU_global") & (sub_v3["hidden"] == 128)].sort_values("seed")
    c_runs = sub_v3[(sub_v3["method"] == "GRU_cond") & (sub_v3["hidden"] == 64)].sort_values("seed")
    
    seeds = g_runs["seed"].values
    assert list(seeds) == list(c_runs["seed"].values) == list(range(10)), f"Seeds mismatch for {sub}!"
    
    rmse_g = g_runs["rmse"].values
    rmse_c = c_runs["rmse"].values
    
    # Paired differences (cond - global)
    diffs = rmse_c - rmse_g
    n_seeds = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(n_seeds)
    
    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(rmse_c, rmse_g)
    
    # 95% t-interval over seeds (df = n_seeds - 1 = 9)
    t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
    ci_lower_seed = mean_diff - t_crit * se_diff
    ci_upper_seed = mean_diff + t_crit * se_diff
    
    # Exact two-sided Wilcoxon signed-rank test
    w_res = stats.wilcoxon(rmse_c, rmse_g, alternative='two-sided')
    w_stat, w_pval = w_res.statistic, w_res.pvalue
    
    # Engine-level paired bootstrap (10,000 resamples) on seed-averaged predictions
    preds_g_all = []
    preds_c_all = []
    for s in range(10):
        dg = pd.read_csv(os.path.join(PREDS_DIR, f"{sub}_G_h128_s{s}.csv"))
        dc = pd.read_csv(os.path.join(PREDS_DIR, f"{sub}_C_h64_s{s}.csv"))
        preds_g_all.append(dg["pred"].values)
        preds_c_all.append(dc["pred"].values)
        
    y_true = dg["true_RUL"].values
    n_engines = len(y_true)
    
    p_g_avg = np.mean(preds_g_all, axis=0)
    p_c_avg = np.mean(preds_c_all, axis=0)
    
    observed_diff_eng = np.sqrt(np.mean((p_c_avg - y_true) ** 2)) - np.sqrt(np.mean((p_g_avg - y_true) ** 2))
    
    rng = np.random.RandomState(0)
    B = 10000
    boot_diffs = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n_engines, size=n_engines)
        y_b = y_true[idx]
        pg_b = p_g_avg[idx]
        pc_b = p_c_avg[idx]
        boot_diffs[b] = np.sqrt(np.mean((pc_b - y_b) ** 2)) - np.sqrt(np.mean((pg_b - y_b) ** 2))
        
    ci_boot_lower = np.percentile(boot_diffs, 2.5)
    ci_boot_upper = np.percentile(boot_diffs, 97.5)
    
    p_vals_primary[sub] = t_pval
    
    rows_primary.append({
        "comparison_type": "Primary (Global h=128 vs Cond h=64, 10 seeds)",
        "subset": sub,
        "n_seeds": n_seeds,
        "n_engines": n_engines,
        "global_mean_rmse": np.mean(rmse_g),
        "global_std_rmse": np.std(rmse_g, ddof=1),
        "cond_mean_rmse": np.mean(rmse_c),
        "cond_std_rmse": np.std(rmse_c, ddof=1),
        "mean_rmse_diff": mean_diff,
        "std_rmse_diff": std_diff,
        "seed_t_stat": t_stat,
        "seed_t_pval": t_pval,
        "seed_t_95ci_lower": ci_lower_seed,
        "seed_t_95ci_upper": ci_upper_seed,
        "seed_wilcoxon_stat": w_stat,
        "seed_wilcoxon_pval": w_pval,
        "engine_observed_rmse_diff": observed_diff_eng,
        "engine_boot_95ci_lower": ci_boot_lower,
        "engine_boot_95ci_upper": ci_boot_upper,
        "engine_bootstrap_label": "conditional on the trained models, does not capture training randomness"
    })

# Holm correction over FD002 and FD004 for primary seed t-test
# Sort p-values
sorted_subs = sorted(p_vals_primary.keys(), key=lambda s: p_vals_primary[s])
holm_pvals = {}
m = len(sorted_subs)
for rank, s in enumerate(sorted_subs):
    adj_p = min(1.0, p_vals_primary[s] * (m - rank))
    holm_pvals[s] = adj_p

for r in rows_primary:
    r["seed_t_holm_adj_pval"] = holm_pvals[r["subset"]]

# 2. Secondary Comparison: Global h=64 vs Cond h=64 (seeds 0-2 from gru_runs_v2)
rows_secondary = []
for sub in ["FD002", "FD004"]:
    g_runs = df_v2[(df_v2["subset"] == sub) & (df_v2["method"] == "GRU_global")].sort_values("seed")
    c_runs = df_v2[(df_v2["subset"] == sub) & (df_v2["method"] == "GRU_cond")].sort_values("seed")
    
    rmse_g = g_runs["rmse"].values[:3]
    rmse_c = c_runs["rmse"].values[:3]
    diffs = rmse_c - rmse_g
    
    t_stat, t_pval = stats.ttest_rel(rmse_c, rmse_g)
    w_res = stats.wilcoxon(rmse_c, rmse_g, alternative='two-sided')
    
    rows_secondary.append({
        "comparison_type": "Secondary (Global h=64 vs Cond h=64, seeds 0-2)",
        "subset": sub,
        "n_seeds": 3,
        "n_engines": len(pd.read_csv(f"data_clean/test_RUL_{sub}.csv")),
        "global_mean_rmse": np.mean(rmse_g),
        "global_std_rmse": np.std(rmse_g, ddof=1),
        "cond_mean_rmse": np.mean(rmse_c),
        "cond_std_rmse": np.std(rmse_c, ddof=1),
        "mean_rmse_diff": np.mean(diffs),
        "std_rmse_diff": np.std(diffs, ddof=1),
        "seed_t_stat": t_stat,
        "seed_t_pval": t_pval,
        "seed_t_95ci_lower": np.mean(diffs) - stats.t.ppf(0.975, df=2) * (np.std(diffs, ddof=1) / np.sqrt(3)),
        "seed_t_95ci_upper": np.mean(diffs) + stats.t.ppf(0.975, df=2) * (np.std(diffs, ddof=1) / np.sqrt(3)),
        "seed_wilcoxon_stat": w_res.statistic,
        "seed_wilcoxon_pval": w_res.pvalue,
        "engine_observed_rmse_diff": np.nan,
        "engine_boot_95ci_lower": np.nan,
        "engine_boot_95ci_upper": np.nan,
        "engine_bootstrap_label": "N/A",
        "seed_t_holm_adj_pval": np.nan
    })

df_all_stats = pd.DataFrame(rows_primary + rows_secondary)
df_all_stats.to_csv(STATS_V3_CSV, index=False)
print(f"Saved {STATS_V3_CSV}", flush=True)

# Append comprehensive explanation to results/stats_notes.md
notes_text = """
# Detailed Methodology for Statistical Tests (Stage 3 / results/stats_v3.csv)

### 1. Primary Comparison: GRU_global (hidden=128) vs. GRU_cond (hidden=64) over 10 Seeds (Seeds 0-9)
- **Seed-Level Paired Tests (n=10):**
  - Evaluated on test RMSE per seed: $d_s = \\text{RMSE}_{\\text{cond}, s} - \\text{RMSE}_{\\text{global}, s}$ for seeds $s \\in \\{0, \\dots, 9\\}$.
  - Each seed corresponds to the exact same 80/20 engine-level train/validation partition.
  - **Paired $t$-test:** Standard Student's paired $t$-test ($df = 9$), testing null hypothesis $H_0: \\mu_d = 0$.
  - **95% $t$-interval:** $\\bar{d} \\pm t_{0.025, 9} \\times \\frac{s_d}{\\sqrt{10}}$, where $t_{0.025, 9} = 2.262157$.
  - **Exact Wilcoxon Signed-Rank Test:** Two-sided test on the 10 paired differences $d_s$.
  - **Holm-Bonferroni Correction:** Applied across the two multi-condition subsets FD002 and FD004.
- **Engine-Level Paired Bootstrap (B=10,000 resamples):**
  - Predictions are first averaged over all 10 seeds for each test engine: $p_{\\text{cond}, i} = \\frac{1}{10}\\sum_{s=0}^9 p_{\\text{cond}, s, i}$ and $p_{\\text{global}, i} = \\frac{1}{10}\\sum_{s=0}^9 p_{\\text{global}, s, i}$.
  - The statistic evaluated on each bootstrap sample $b$ (resampling test engine indices with replacement) is:
    $$\\Delta\\text{RMSE}^{(b)} = \\sqrt{\\frac{1}{N}\\sum_{i \\in \\text{sample}_b} (p_{\\text{cond}, i} - y_i)^2} - \\sqrt{\\frac{1}{N}\\sum_{i \\in \\text{sample}_b} (p_{\\text{global}, i} - y_i)^2}$$
  - The 95% bootstrap confidence interval is the 2.5th and 97.5th percentiles of $\\Delta\\text{RMSE}^{(b)}$.
  - **Conditioning Label:** This engine bootstrap is explicitly *conditional on the trained models, and does not capture training randomness*.

### 2. Secondary Comparison: GRU_global (hidden=64) vs. GRU_cond (hidden=64) over Seeds 0-2
- Evaluated on test RMSE per seed ($n=3$) from the original benchmark runs (`results/gru_runs_v2.csv`).
- Seed-level paired $t$-test ($df=2$), 95% $t$-interval, and exact Wilcoxon test.
"""

with open(STATS_NOTES_MD, "a") as f:
    f.write(notes_text)
print(f"Updated {STATS_NOTES_MD}", flush=True)
