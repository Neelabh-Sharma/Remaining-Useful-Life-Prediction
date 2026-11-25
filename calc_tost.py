import numpy as np
import pandas as pd
from scipy import stats

# Data from frozen results
cond_10 = np.array([30.83188788422825, 30.911633878474216, 32.34916855513709])
glob_100 = np.array([30.831678129032545, 30.159886134266195, 32.40058728136396])

diffs_paired = cond_10 - glob_100
mean_diff = np.mean(diffs_paired)
std_diff = np.std(diffs_paired, ddof=1)
se_diff = std_diff / np.sqrt(len(diffs_paired))

# Two-sample Welch t-test (unpaired)
welch_res = stats.ttest_ind(cond_10, glob_100, equal_var=False)

# Paired t-test
paired_res = stats.ttest_rel(cond_10, glob_100)

delta = 1.5
# TOST paired
t1 = (mean_diff - (-delta)) / se_diff
p1 = 1 - stats.t.cdf(t1, df=len(diffs_paired)-1)
t2 = (mean_diff - delta) / se_diff
p2 = stats.t.cdf(t2, df=len(diffs_paired)-1)
p_tost_paired = max(p1, p2)

# TOST two-sample Welch
se_welch = np.sqrt(np.var(cond_10, ddof=1)/3 + np.var(glob_100, ddof=1)/3)
s1, s2 = np.var(cond_10, ddof=1), np.var(glob_100, ddof=1)
df_welch = (s1/3 + s2/3)**2 / ((s1/3)**2 / 2 + (s2/3)**2 / 2)
t1_w = (mean_diff - (-delta)) / se_welch
p1_w = 1 - stats.t.cdf(t1_w, df=df_welch)
t2_w = (mean_diff - delta) / se_welch
p2_w = stats.t.cdf(t2_w, df=df_welch)
p_tost_welch = max(p1_w, p2_w)

# 95% CI for difference
ci_lower_paired = mean_diff - stats.t.ppf(0.975, df=len(diffs_paired)-1) * se_diff
ci_upper_paired = mean_diff + stats.t.ppf(0.975, df=len(diffs_paired)-1) * se_diff

print("=== TOST EVALUATION (cond@10% vs global@100%) ===")
print(f"cond@10% mean: {np.mean(cond_10):.4f} +/- {np.std(cond_10, ddof=1):.4f}")
print(f"global@100% mean: {np.mean(glob_100):.4f} +/- {np.std(glob_100, ddof=1):.4f}")
print(f"Mean difference: {mean_diff:.4f} cycles")
print(f"Paired t-test: t={paired_res.statistic:.4f}, p={paired_res.pvalue:.4f}")
print(f"Welch t-test: t={welch_res.statistic:.4f}, df={df_welch:.2f}, p={welch_res.pvalue:.4f}")
print(f"Paired 95% CI for difference: [{ci_lower_paired:.4f}, {ci_upper_paired:.4f}]")
print(f"TOST paired (delta={delta}): p={p_tost_paired:.4f} (p1={p1:.4f}, p2={p2:.4f})")
print(f"TOST Welch (delta={delta}): p={p_tost_welch:.4f} (p1={p1_w:.4f}, p2={p2_w:.4f})")
if p_tost_paired < 0.05:
    print("EQUIVALENCE REJECTS NULL: Statistically equivalent within +/- 1.5 cycles.")
else:
    print("EQUIVALENCE FAILS TO REJECT NULL: Inconclusive / underpowered at N=3 seeds.")

# Also evaluate cond@25% vs global@100%
cond_25 = np.array([31.275789579441824, 30.650547850600816, 30.775365828650795])
diffs_25 = cond_25 - glob_100
mean_diff_25 = np.mean(diffs_25)
std_diff_25 = np.std(diffs_25, ddof=1)
se_diff_25 = std_diff_25 / np.sqrt(3)
t1_25 = (mean_diff_25 - (-delta)) / se_diff_25
p1_25 = 1 - stats.t.cdf(t1_25, df=2)
t2_25 = (mean_diff_25 - delta) / se_diff_25
p2_25 = stats.t.cdf(t2_25, df=2)
p_tost_25 = max(p1_25, p2_25)
print("\n=== TOST EVALUATION (cond@25% vs global@100%) ===")
print(f"cond@25% mean: {np.mean(cond_25):.4f} +/- {np.std(cond_25, ddof=1):.4f}")
print(f"Mean diff vs global@100%: {mean_diff_25:.4f}")
print(f"TOST paired p: {p_tost_25:.4f}")
