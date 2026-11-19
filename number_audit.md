# Comprehensive Numerical Audit for `paper.tex`

Every numerical value reported in `paper.tex` is audited below against its source artifact, file path, table, and row index. No numbers were invented or estimated.

---

## 1. Dataset Characteristics (Section IV-A, Table I)
*Source: `data_clean/train_FD00x.csv` and `data_clean/test_RUL_FD00x.csv`*

| Parameter | Subset | Value in Paper | Source File | Exact Verification Query / Row |
| :--- | :--- | :--- | :--- | :--- |
| Conditions | FD001 | 1 | `data_clean/train_FD001.csv` | Single operating regime (Sea Level) |
| Conditions | FD002 | 6 | `data_clean/train_FD002.csv` | 6 discrete operating regimes |
| Conditions | FD003 | 1 | `data_clean/train_FD003.csv` | Single operating regime (Sea Level) |
| Conditions | FD004 | 6 | `data_clean/train_FD004.csv` | 6 discrete operating regimes |
| Fault Modes | FD001 | 1 | `project_log.md` | High Pressure Compressor (HPC) degradation |
| Fault Modes | FD002 | 1 | `project_log.md` | High Pressure Compressor (HPC) degradation |
| Fault Modes | FD003 | 2 | `project_log.md` | HPC and Fan degradation |
| Fault Modes | FD004 | 2 | `project_log.md` | HPC and Fan degradation |
| Train Engines | FD001 | 100 | `data_clean/train_FD001.csv` | `len(train_df['unit_number'].unique()) == 100` |
| Train Engines | FD002 | 260 | `data_clean/train_FD002.csv` | `len(train_df['unit_number'].unique()) == 260` |
| Train Engines | FD003 | 100 | `data_clean/train_FD003.csv` | `len(train_df['unit_number'].unique()) == 100` |
| Train Engines | FD004 | 249 | `data_clean/train_FD004.csv` | `len(train_df['unit_number'].unique()) == 249` |
| Test Engines | FD001 | 100 | `data_clean/test_RUL_FD001.csv`| `len(test_df['unit_number'].unique()) == 100` |
| Test Engines | FD002 | 259 | `data_clean/test_RUL_FD002.csv`| `len(test_df['unit_number'].unique()) == 259` |
| Test Engines | FD003 | 100 | `data_clean/test_RUL_FD003.csv`| `len(test_df['unit_number'].unique()) == 100` |
| Test Engines | FD004 | 248 | `data_clean/test_RUL_FD004.csv`| `len(test_df['unit_number'].unique()) == 248` |

---

## 2. Unsupervised Regime Clustering (Section IV-B, Fig. 1)
*Source: `results/cluster_metrics_FD002.csv` and `results/cluster_metrics_FD004.csv`*

| Metric | Dataset | Value in Paper | Source File | Row Index / Value |
| :--- | :--- | :--- | :--- | :--- |
| Silhouette ($k=6$) | FD002 | 0.999659 | `results/cluster_metrics_FD002.csv` | Row 5 (`k=6`): `0.9996587` |
| Silhouette ($k=6$) | FD004 | 0.999660 | `results/cluster_metrics_FD004.csv` | Row 5 (`k=6`): `0.9996598` |
| Inertia ($k=5$) | FD002 | $2.5 \times 10^5$ | `results/cluster_metrics_FD002.csv` | Row 4 (`k=5`): `246561.26` |
| Inertia ($k=5$) | FD004 | $2.8 \times 10^5$ | `results/cluster_metrics_FD004.csv` | Row 4 (`k=5`): `281440.25` |
| Inertia ($k=6$) | FD002 | 0.41 | `results/cluster_metrics_FD002.csv` | Row 5 (`k=6`): `0.4124` |
| Inertia ($k=6$) | FD004 | 0.47 | `results/cluster_metrics_FD004.csv` | Row 5 (`k=6`): `0.4682` |
| Over-cluster split ($k=9$) | FD004 | $<0.005$ | `project_log.md` | Distance between fractured sub-cluster centroids $\approx 0.005$ |

---

## 3. Main Benchmark Results under Protocol A (Section VI-A, Table II)
*Source: `results/final_tables.md` (Table 1), `results/baseline_runs.csv`, `results/rf_cond_runs.csv`, and `results/gru_runs_v2.csv`*

### FD001
- **Ridge (Global):** RMSE = `18.24`, NASA = `537.3` (`results/baseline_runs.csv`, Row 2)
- **RF (Global):** RMSE = `18.49 ± 0.13`, NASA = `878.4 ± 54.4` (`results/final_tables.md`, Table 1, Row 1)
- **RF (Cond):** RMSE = `18.48 ± 0.13`, NASA = `879.2 ± 59.0` (`results/final_tables.md`, Table 1, Row 1)
- **GRU (Global):** RMSE = `15.94 ± 0.29`, NASA = `463.8 ± 84.9` (`results/final_tables.md`, Table 1, Row 1)
- **GRU (Cond):** RMSE = `15.94 ± 0.29`, NASA = `463.8 ± 84.9` (`results/final_tables.md`, Table 1, Row 1)
- **$\Delta$ GRU RMSE:** `0.0%`

### FD002
- **Ridge (Global):** RMSE = `29.20`, NASA = `12905.2` (`results/baseline_runs.csv`, Row 4)
- **RF (Global):** RMSE = `30.89 ± 0.05`, NASA = `14547.0 ± 743.8` (`results/final_tables.md`, Table 1, Row 3)
- **RF (Cond):** RMSE = `29.93 ± 0.09`, NASA = `13751.6 ± 330.0` (`results/final_tables.md`, Table 1, Row 3)
- **GRU (Global):** RMSE = `28.98 ± 1.23`, NASA = `18033.3 ± 6705.3` (`results/final_tables.md`, Table 1, Row 3)
- **GRU (Cond):** RMSE = `27.07 ± 0.63`, NASA = `8573.4 ± 1276.2` (`results/final_tables.md`, Table 1, Row 3)
- **$\Delta$ GRU RMSE:** `-6.6%` ($[27.07 - 28.98] / 28.98 = -6.59\%$)

### FD003
- **Ridge (Global):** RMSE = `18.77`, NASA = `651.5` (`results/baseline_runs.csv`, Row 6)
- **RF (Global):** RMSE = `19.61 ± 0.23`, NASA = `1507.4 ± 56.7` (`results/final_tables.md`, Table 1, Row 5)
- **RF (Cond):** RMSE = `19.60 ± 0.26`, NASA = `1501.1 ± 62.2` (`results/final_tables.md`, Table 1, Row 5)
- **GRU (Global):** RMSE = `19.33 ± 2.89`, NASA = `1247.7 ± 557.7` (`results/final_tables.md`, Table 1, Row 5)
- **GRU (Cond):** RMSE = `19.33 ± 2.89`, NASA = `1247.7 ± 557.6` (`results/final_tables.md`, Table 1, Row 5)
- **$\Delta$ GRU RMSE:** `0.0%`

### FD004
- **Ridge (Global):** RMSE = `31.51`, NASA = `8652.6` (`results/baseline_runs.csv`, Row 8)
- **RF (Global):** RMSE = `32.04 ± 0.15`, NASA = `9008.6 ± 230.9` (`results/final_tables.md`, Table 1, Row 7)
- **RF (Cond):** RMSE = `30.32 ± 0.09`, NASA = `7593.8 ± 139.5` (`results/final_tables.md`, Table 1, Row 7)
- **GRU (Global):** RMSE = `31.13 ± 1.15`, NASA = `9214.6 ± 1550.8` (`results/final_tables.md`, Table 1, Row 7)
- **GRU (Cond):** RMSE = `26.79 ± 0.68`, NASA = `4950.5 ± 403.8` (`results/final_tables.md`, Table 1, Row 7)
- **$\Delta$ GRU RMSE:** `-13.9%` ($[26.79 - 31.13] / 31.13 = -13.94\%$)

---

## 4. Statistical Significance Metrics (Section VI-B)
*Source: `results/stats.csv`*

| Test Metric | Dataset | Value in Paper | Source File | Column / Exact Value |
| :--- | :--- | :--- | :--- | :--- |
| Seed-level paired t-test ($n=3$) | FD002 | $t = -2.544, p = 0.1259$ | `results/stats.csv`, Row 2 | `t_stat_seed`: -2.5444, `t_pval_seed`: 0.1259 |
| Seed-level paired t-test ($n=3$) | FD004 | $t = -12.663, p = 0.00618$| `results/stats.csv`, Row 3 | `t_stat_seed`: -12.6628, `t_pval_seed`: 0.00618 |
| Engine paired t-test (squared err) | FD002 | $t = -2.720, p = 0.00697$ | `results/stats.csv`, Row 2 | `t_stat_engine`: -2.7203, `t_pval_engine`: 0.006966 |
| Engine paired t-test (squared err) | FD004 | $t = -5.936, p = 9.82 \times 10^{-9}$ | `results/stats.csv`, Row 3 | `t_stat_engine`: -5.9364, `t_pval_engine`: $9.82 \times 10^{-9}$ |
| Engine Wilcoxon signed-rank | FD002 | $W = 13336.0, p = 0.00374$| `results/stats.csv`, Row 2 | `wilcoxon_stat_engine`: 13336.0, `wilcoxon_pval_engine`: 0.003737 |
| Engine Wilcoxon signed-rank | FD004 | $W = 8759.0, p = 3.50 \times 10^{-9}$ | `results/stats.csv`, Row 3 | `wilcoxon_stat_engine`: 8759.0, `wilcoxon_pval_engine`: $3.50 \times 10^{-9}$ |
| Holm-adjusted engine $t$ $p$-value | FD002 | $0.00697$ | `results/stats.csv`, Row 2 | `holm_adj_t_pval_engine`: 0.006966 |
| Holm-adjusted engine $t$ $p$-value | FD004 | $1.96 \times 10^{-8}$ | `results/stats.csv`, Row 3 | `holm_adj_t_pval_engine`: $1.96 \times 10^{-8}$ |
| Bootstrap observed $\Delta\text{RMSE}$| FD002 | $-1.665$ cycles | `results/stats.csv`, Row 2 | `observed_diff_engine_avg`: -1.6648 |
| Bootstrap 95% CI | FD002 | $[-2.80, -0.56]$ cycles | `results/stats.csv`, Row 2 | `boot_95ci_lower`: -2.7954, `boot_95ci_upper`: -0.5608 |
| Bootstrap observed $\Delta\text{RMSE}$| FD004 | $-4.275$ cycles | `results/stats.csv`, Row 3 | `observed_diff_engine_avg`: -4.2747 |
| Bootstrap 95% CI | FD004 | $[-5.65, -3.03]$ cycles | `results/stats.csv`, Row 3 | `boot_95ci_lower`: -5.6498, `boot_95ci_upper`: -3.0332 |

---

## 5. Label Scarcity Evaluation on FD004 (Section VI-C, Table III, Fig. 2)
*Source: `results/extra_runs.csv` and `results/final_tables.md` (Table 3)*

| Label Budget | Model | Metric | Value in Paper | Source File | Row Index / Query |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10% | GRU Global | RMSE | $37.49 \pm 0.79$ | `results/final_tables.md`, Table 3 | Mean of rows 8, 9, 10 in `extra_runs.csv` |
| 10% | GRU Cond | RMSE | $31.36 \pm 0.85$ | `results/final_tables.md`, Table 3 | Mean of rows 11, 12, 13 in `extra_runs.csv` |
| 10% | Reduction | $\Delta$ RMSE | $-16.3\%$ | `results/final_tables.md`, Table 3 | $[31.36 - 37.49] / 37.49 = -16.35\%$ |
| 25% | GRU Global | RMSE | $35.38 \pm 0.64$ | `results/final_tables.md`, Table 3 | Mean of rows 14, 15, 16 in `extra_runs.csv` |
| 25% | GRU Cond | RMSE | $30.90 \pm 0.33$ | `results/final_tables.md`, Table 3 | Mean of rows 17, 18, 19 in `extra_runs.csv` |
| 25% | Reduction | $\Delta$ RMSE | $-12.6\%$ | `results/final_tables.md`, Table 3 | $[30.90 - 35.38] / 35.38 = -12.66\%$ |
| 50% | GRU Global | RMSE | $33.24 \pm 1.03$ | `results/final_tables.md`, Table 3 | Mean of rows 20, 21, 22 in `extra_runs.csv` |
| 50% | GRU Cond | RMSE | $29.05 \pm 0.80$ | `results/final_tables.md`, Table 3 | Mean of rows 23, 24, 25 in `extra_runs.csv` |
| 50% | Reduction | $\Delta$ RMSE | $-12.6\%$ | `results/final_tables.md`, Table 3 | $[29.05 - 33.24] / 33.24 = -12.61\%$ |
| 100% | GRU Global | RMSE | $31.13 \pm 1.15$ | `results/final_tables.md`, Table 3 | Baseline global (100%) |
| 100% | GRU Cond | RMSE | $26.79 \pm 0.68$ | `results/final_tables.md`, Table 3 | Full condition-aware (100%) |
| 100% | Reduction | $\Delta$ RMSE | $-13.9\%$ | `results/final_tables.md`, Table 3 | $[26.79 - 31.13] / 31.13 = -13.94\%$ |

---

## 6. Cluster Count Granularity Ablation (Section VI-D, Table IV)
*Source: `results/extra_runs.csv` and `results/final_tables.md` (Table 2)*

| Cluster Count ($k$) | Description | Test RMSE | NASA Score | Relative to $k=1$ | Source File & Query |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $k=1$ | Global StandardScaler | $31.13 \pm 1.15$ | $9214.6 \pm 1550.8$ | Baseline (0.0%) | `results/final_tables.md`, Table 2, Row 1 |
| $k=3$ | Coarse Clustering | $28.79 \pm 0.86$ | $7145.3 \pm 878.9$ | $-7.5\%$ | Mean of rows 2, 3, 4 in `extra_runs.csv` |
| $k=6$ | True Physical Regimes | $26.79 \pm 0.68$ | $4950.5 \pm 403.8$ | $-13.9\%$ | Main benchmark cond $k=6$ |
| $k=9$ | Over-Clustering | $26.79 \pm 0.68$ | $4963.9 \pm 392.5$ | $-13.9\%$ | Mean of rows 5, 6, 7 in `extra_runs.csv` |

---

## 7. Error by Degradation Stage (Section VI-E, Fig. 3)
*Source: `results/error_by_bin.csv`*

| Subset | RUL Bin | Engines | GRU Global RMSE | GRU Cond RMSE | $\Delta$ RMSE (Mean) | Source File Row |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FD002 | $0$–$30$ | 61 | $6.94 \pm 0.34$ | $6.12 \pm 0.43$ | $-0.83$ | `results/error_by_bin.csv`, Row 2 |
| FD002 | $30$–$60$ | 41 | $16.91 \pm 3.16$ | $13.13 \pm 0.35$ | $-3.78$ | `results/error_by_bin.csv`, Row 3 |
| FD002 | $60$–$125$ | 100 | $18.41 \pm 1.18$ | $16.90 \pm 0.87$ | $-1.51$ | `results/error_by_bin.csv`, Row 4 |
| FD002 | $>125$ | 57 | $54.30 \pm 4.43$ | $51.61 \pm 1.89$ | $-2.70$ | `results/error_by_bin.csv`, Row 5 |
| FD004 | $0$–$30$ | 53 | $15.40 \pm 1.27$ | $7.60 \pm 0.86$ | $-7.81$ | `results/error_by_bin.csv`, Row 6 |
| FD004 | $30$–$60$ | 38 | $23.62 \pm 2.42$ | $16.91 \pm 1.83$ | $-6.72$ | `results/error_by_bin.csv`, Row 7 |
| FD004 | $60$–$125$ | 90 | $21.45 \pm 0.78$ | $18.37 \pm 0.81$ | $-3.08$ | `results/error_by_bin.csv`, Row 8 |
| FD004 | $>125$ | 67 | $49.63 \pm 1.49$ | $44.66 \pm 0.59$ | $-4.97$ | `results/error_by_bin.csv`, Row 9 |

---

## 8. Protocol A vs. Protocol B Comparison (Section VI-F, Table V)
*Source: `results/protocol_rmse.csv` and `results/final_tables.md` (Table 1b)*

| Subset | Method | Protocol A RMSE | Protocol B RMSE | Source File Verification |
| :--- | :--- | :--- | :--- | :--- |
| FD001 | GRU Global | $15.94 \pm 0.29$ | $14.66 \pm 0.48$ | `results/final_tables.md`, Protocol B |
| FD001 | GRU Cond | $15.94 \pm 0.29$ | $14.66 \pm 0.48$ | `results/final_tables.md`, Protocol B |
| FD002 | GRU Global | $28.98 \pm 1.23$ | $16.75 \pm 0.16$ | `results/final_tables.md`, Protocol B |
| FD002 | GRU Cond | $27.07 \pm 0.63$ | $14.46 \pm 0.19$ | `results/final_tables.md`, Protocol B |
| FD003 | GRU Global | $19.33 \pm 2.89$ | $18.16 \pm 2.74$ | `results/final_tables.md`, Protocol B |
| FD003 | GRU Cond | $19.33 \pm 2.89$ | $18.16 \pm 2.74$ | `results/final_tables.md`, Protocol B |
| FD004 | GRU Global | $31.13 \pm 1.15$ | $19.81 \pm 1.13$ | `results/final_tables.md`, Protocol B |
| FD004 | GRU Cond | $26.79 \pm 0.68$ | $14.95 \pm 0.96$ | `results/final_tables.md`, Protocol B |

---

## 9. Model Architecture & Preprocessing Hyperparameters (Section V)
*Source: `train.py`, `preprocess.py`, `utils.py`*

- **Sequence Window Length:** $30$ cycles (`utils.py`, `make_train_windows(..., window=30)`)
- **Sliding Window Stride:** $5$ cycles (`utils.py`, `make_train_windows(..., stride=5)`)
- **GRU Architecture:** 2 recurrent layers, hidden dimension $64$, dropout $0.2$ (`train.py`, `model.py`)
- **Target Normalization:** Piecewise linear target capped at $125$ cycles; scaled via $y / 125.0$ (`train.py`, lines 96, 99)
- **Optimizer & Learning Rate:** Adam optimizer, initial $lr = 2\times 10^{-3}$, gradient clipping $1.0$ (`train.py`, lines 115--116, 136)
- **LR Scheduler:** $\text{ReduceLROnPlateau}$ (mode='min', factor=0.5, patience=3) (`train.py`, line 117)
- **Batch Size:** $512$ for training (`run_main_v2.py`, line 40); $2048$ for evaluation (`train.py`, line 181)
- **Training Epochs:** Max $40$ epochs, early stopping patience $8$ (`train.py`, lines 46, 171)
- **Engine Holdout Validation Split:** $20\%$ of training engines held out by seed (`train.py`, line 73)
- **Epsilon Zero-Variance Threshold:** $10^{-10}$ (`preprocess.py`, lines 118, 150)
- **Random Seeds:** $0, 1, 2$ (`run_main_v2.py`, line 24; `run_extra.py`)
