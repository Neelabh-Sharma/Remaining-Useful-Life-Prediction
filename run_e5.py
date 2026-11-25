import os
import numpy as np
import pandas as pd
from scipy import stats
from utils import rmse, nasa_score

PREDS_DIR = "results/preds"
subsets = ["FD002", "FD004"]

print("=== RUNNING E5: ADVANCED STATISTICAL BENCHMARKS ===")

stats_rows = []

for sub in subsets:
    # 1. Load predictions across seeds 0, 1, 2
    preds_g = []
    preds_c = []
    for s in [0, 1, 2]:
        dg = pd.read_csv(f"{PREDS_DIR}/{sub}_G_kNone_f1.0_s{s}.csv")
        dc = pd.read_csv(f"{PREDS_DIR}/{sub}_C_kNone_f1.0_s{s}.csv")
        preds_g.append(dg['pred'].values)
        preds_c.append(dc['pred'].values)
        
    y_true = dg['true_RUL'].values
    N_eng = len(y_true)
    N_seeds = 3
    
    # Compute per-seed metrics
    rmse_g_seeds = np.array([rmse(y_true, p) for p in preds_g])
    rmse_c_seeds = np.array([rmse(y_true, p) for p in preds_c])
    nasa_g_seeds = np.array([nasa_score(y_true, p) for p in preds_g])
    nasa_c_seeds = np.array([nasa_score(y_true, p) for p in preds_c])
    
    diff_rmse_seeds = rmse_c_seeds - rmse_g_seeds
    mean_diff_rmse = np.mean(diff_rmse_seeds)
    std_diff_rmse = np.std(diff_rmse_seeds, ddof=1)
    
    # Seed-level paired t-test & Cohen's d
    t_seed, p_seed_t = stats.ttest_rel(rmse_c_seeds, rmse_g_seeds)
    cohen_d = mean_diff_rmse / std_diff_rmse if std_diff_rmse > 0 else np.nan
    w_seed_res = stats.wilcoxon(rmse_c_seeds, rmse_g_seeds, alternative='two-sided')
    
    # 2. Engine-level paired tests on squared error (seed-averaged predictions)
    pg_avg = np.mean(preds_g, axis=0)
    pc_avg = np.mean(preds_c, axis=0)
    
    sq_err_g = (pg_avg - y_true) ** 2
    sq_err_c = (pc_avg - y_true) ** 2
    t_eng, p_eng_t = stats.ttest_rel(sq_err_c, sq_err_g)
    w_eng_res = stats.wilcoxon(sq_err_c, sq_err_g, alternative='two-sided')
    
    # 3. Double Bootstrap: Resample BOTH engines AND seeds (B=10,000)
    rng = np.random.RandomState(42)
    B = 10000
    boot_rmse_diffs = np.empty(B)
    boot_nasa_diffs = np.empty(B)
    
    for b in range(B):
        # Resample seeds with replacement (3 seeds)
        s_idx = rng.randint(0, N_seeds, size=N_seeds)
        # Resample test engines with replacement (N_eng engines)
        e_idx = rng.randint(0, N_eng, size=N_eng)
        
        y_b = y_true[e_idx]
        # Average resampled seeds on resampled engines
        pg_b = np.mean([preds_g[s][e_idx] for s in s_idx], axis=0)
        pc_b = np.mean([preds_c[s][e_idx] for s in s_idx], axis=0)
        
        boot_rmse_diffs[b] = rmse(y_b, pc_b) - rmse(y_b, pg_b)
        boot_nasa_diffs[b] = nasa_score(y_b, pc_b) - nasa_score(y_b, pg_b)
        
    ci_double_lower = np.percentile(boot_rmse_diffs, 2.5)
    ci_double_upper = np.percentile(boot_rmse_diffs, 97.5)
    
    ci_nasa_lower = np.percentile(boot_nasa_diffs, 2.5)
    ci_nasa_upper = np.percentile(boot_nasa_diffs, 97.5)
    
    stats_rows.append({
        "subset": sub,
        "n_seeds": N_seeds,
        "n_engines": N_eng,
        "global_rmse_mean": np.mean(rmse_g_seeds),
        "global_rmse_std": np.std(rmse_g_seeds, ddof=1),
        "cond_rmse_mean": np.mean(rmse_c_seeds),
        "cond_rmse_std": np.std(rmse_c_seeds, ddof=1),
        "mean_seed_diff": mean_diff_rmse,
        "cohen_d": cohen_d,
        "seed_t_stat": t_seed,
        "seed_t_pval": p_seed_t,
        "seed_wilcoxon_stat": w_seed_res.statistic,
        "seed_wilcoxon_pval": w_seed_res.pvalue,
        "engine_t_stat": t_eng,
        "engine_t_pval": p_eng_t,
        "engine_wilcoxon_stat": w_eng_res.statistic,
        "engine_wilcoxon_pval": w_eng_res.pvalue,
        "double_boot_rmse_95ci_lower": ci_double_lower,
        "double_boot_rmse_95ci_upper": ci_double_upper,
        "global_nasa_mean": np.mean(nasa_g_seeds),
        "cond_nasa_mean": np.mean(nasa_c_seeds),
        "double_boot_nasa_95ci_lower": ci_nasa_lower,
        "double_boot_nasa_95ci_upper": ci_nasa_upper
    })

df_e5 = pd.DataFrame(stats_rows)

# Holm correction over the 2 subsets for seed t-test and engine t-test
p_seeds = df_e5['seed_t_pval'].values
p_engs = df_e5['engine_t_pval'].values

df_e5['seed_t_holm_adj'] = [min(1.0, 2 * p_seeds[0]) if p_seeds[0] < p_seeds[1] else min(1.0, p_seeds[0]),
                            min(1.0, 2 * p_seeds[1]) if p_seeds[1] < p_seeds[0] else min(1.0, p_seeds[1])]
df_e5['engine_t_holm_adj'] = [min(1.0, 2 * p_engs[0]) if p_engs[0] < p_engs[1] else min(1.0, p_engs[0]),
                              min(1.0, 2 * p_engs[1]) if p_engs[1] < p_engs[0] else min(1.0, p_engs[1])]

df_e5.to_csv("results/e5_rigorous_statistics.csv", index=False)
print("Saved results/e5_rigorous_statistics.csv")
print(df_e5[['subset', 'mean_seed_diff', 'cohen_d', 'seed_t_pval', 'engine_t_pval', 'double_boot_rmse_95ci_lower', 'double_boot_rmse_95ci_upper']])
