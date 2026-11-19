import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global matplotlib style for high publication quality
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.autolayout'] = True

os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------
# Figure (a): Label-Efficiency Curve (FD004)
# -------------------------------------------------------------
def plot_label_efficiency():
    # Read extra runs (10%, 25%, 50%)
    df_extra = pd.read_csv("results/extra_runs.csv")
    df_lbl = df_extra[df_extra["exp_type"] == "label_efficiency"]

    # Read 100% runs from gru_runs.csv / gru_runs_v2.csv on FD004
    df_gru = pd.read_csv("results/gru_runs.csv")
    df_gru100 = df_gru[df_gru["subset"] == "FD004"]

    # Compute mean and std for GRU_cond
    cond_stats = {}
    glob_stats = {}

    for frac in [0.10, 0.25, 0.50]:
        sub_c = df_lbl[(df_lbl["label_frac"] == frac) & (df_lbl["method"] == "GRU_cond")]
        cond_stats[int(frac * 100)] = (sub_c["rmse"].mean(), sub_c["rmse"].std())

        sub_g = df_lbl[(df_lbl["label_frac"] == frac) & (df_lbl["method"] == "GRU_global")]
        glob_stats[int(frac * 100)] = (sub_g["rmse"].mean(), sub_g["rmse"].std())

    # 100%
    sub_c100 = df_gru100[df_gru100["method"] == "GRU_cond"]
    cond_stats[100] = (sub_c100["rmse"].mean(), sub_c100["rmse"].std())

    sub_g100 = df_gru100[df_gru100["method"] == "GRU_global"]
    glob_stats[100] = (sub_g100["rmse"].mean(), sub_g100["rmse"].std())

    fractions = [10, 25, 50, 100]
    c_means = [cond_stats[f][0] for f in fractions]
    c_stds = [cond_stats[f][1] for f in fractions]
    g_means = [glob_stats[f][0] for f in fractions]
    g_stds = [glob_stats[f][1] for f in fractions]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(fractions, g_means, yerr=g_stds, fmt='o--', color='#d95f02',
                ecolor='#d95f02', elinewidth=1.5, capsize=4, capthick=1.5,
                linewidth=1.8, markersize=6, label='Global Normalization (Baseline)')
    ax.errorbar(fractions, c_means, yerr=c_stds, fmt='s-', color='#1b9e77',
                ecolor='#1b9e77', elinewidth=1.5, capsize=4, capthick=1.5,
                linewidth=1.8, markersize=6, label='Condition-Aware (Proposed)')

    ax.set_xlabel('Labeled Training Engines (%)')
    ax.set_ylabel('Test RMSE (Cycles)')
    ax.set_xticks(fractions)
    ax.set_xlim(5, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, loc='upper right')

    fig.savefig('figures/label_efficiency_curve.png', dpi=300)
    fig.savefig('figures/label_efficiency_curve.pdf')
    plt.close(fig)
    print("Saved figures/label_efficiency_curve.png and .pdf")

# -------------------------------------------------------------
# Figure (b): Cluster Ablation Bar Chart (FD004)
# -------------------------------------------------------------
def plot_ablation():
    df_extra = pd.read_csv("results/extra_runs.csv")
    df_abl = df_extra[df_extra["exp_type"] == "ablation"]
    df_gru = pd.read_csv("results/gru_runs.csv")

    # k=1 is GRU_global on FD004
    k1_df = df_gru[(df_gru["subset"] == "FD004") & (df_gru["method"] == "GRU_global")]
    k1_mean, k1_std = k1_df["rmse"].mean(), k1_df["rmse"].std()

    # k=3 from extra
    k3_df = df_abl[df_abl["k"] == 3]
    k3_mean, k3_std = k3_df["rmse"].mean(), k3_df["rmse"].std()

    # k=6 from gru_runs
    k6_df = df_gru[(df_gru["subset"] == "FD004") & (df_gru["method"] == "GRU_cond")]
    k6_mean, k6_std = k6_df["rmse"].mean(), k6_df["rmse"].std()

    # k=9 from extra
    k9_df = df_abl[df_abl["k"] == 9]
    k9_mean, k9_std = k9_df["rmse"].mean(), k9_df["rmse"].std()

    ks = ['k=1 (Global)', 'k=3', 'k=6 (Proposed)', 'k=9']
    means = [k1_mean, k3_mean, k6_mean, k9_mean]
    stds = [k1_std, k3_std, k6_std, k9_std]
    colors = ['#7570b3', '#386cb0', '#1b9e77', '#66a61e']

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(ks, means, yerr=stds, capsize=5, color=colors, width=0.55, edgecolor='black', linewidth=0.8, alpha=0.9)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 1.2, f'{m:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Number of Operating Clusters (k)')
    ax.set_ylabel('Test RMSE (Cycles)')
    ax.set_ylim(0, 36)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    fig.savefig('figures/cluster_ablation_barchart.png', dpi=300)
    fig.savefig('figures/cluster_ablation_barchart.pdf')
    plt.close(fig)
    print("Saved figures/cluster_ablation_barchart.png and .pdf")

# -------------------------------------------------------------
# Figure (c): Per-Subset RMSE Bar Chart (Global vs Cond)
# -------------------------------------------------------------
def plot_subset_comparison():
    df_gru = pd.read_csv("results/gru_runs.csv")
    subsets = ['FD001', 'FD002', 'FD003', 'FD004']

    g_means, g_stds = [], []
    c_means, c_stds = [], []

    for s in subsets:
        sg = df_gru[(df_gru["subset"] == s) & (df_gru["method"] == "GRU_global")]["rmse"]
        sc = df_gru[(df_gru["subset"] == s) & (df_gru["method"] == "GRU_cond")]["rmse"]
        g_means.append(sg.mean())
        g_stds.append(sg.std())
        c_means.append(sc.mean())
        c_stds.append(sc.std())

    x = np.arange(len(subsets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, g_means, width, yerr=g_stds, capsize=4, label='GRU Global',
           color='#d95f02', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.bar(x + width/2, c_means, width, yerr=c_stds, capsize=4, label='GRU Condition-Aware',
           color='#1b9e77', edgecolor='black', linewidth=0.8, alpha=0.9)

    ax.set_xlabel('C-MAPSS Dataset Subset')
    ax.set_ylabel('Test RMSE (Cycles)')
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylim(0, 36)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, loc='upper left')

    fig.savefig('figures/subset_rmse_comparison.png', dpi=300)
    fig.savefig('figures/subset_rmse_comparison.pdf')
    plt.close(fig)
    print("Saved figures/subset_rmse_comparison.png and .pdf")

if __name__ == "__main__":
    plot_label_efficiency()
    plot_ablation()
    plot_subset_comparison()

