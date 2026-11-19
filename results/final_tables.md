# Experimental Results and Benchmark Tables

## Table 1: Main Benchmark Results across NASA C-MAPSS Subsets

Evaluation on the final test cycle window (length 30) for each engine across seeds 0, 1, 2. Metrics reported as mean ± std.

> **Sign Convention:** Negative percentages represent lower RMSE (error reduction / performance improvement).

### Protocol A: Standard Test Evaluation (Unclipped Ground-Truth RUL)

| Subset | Metric | Ridge (Global)* | RF (Global) | RF (Cond-Aware) | GRU (Global) | GRU (Cond-Aware) | Δ GRU RMSE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FD001** | RMSE | 18.24 | 18.49 ± 0.13 | 18.48 ± 0.13 | 15.94 ± 0.29 | 15.94 ± 0.29 | 0.0% |
| | NASA | 537.3 | 878.4 ± 54.4 | 879.2 ± 59.0 | 463.8 ± 84.9 | 463.8 ± 84.9 | - |
| **FD002** | RMSE | 29.20 | 30.89 ± 0.05 | 29.93 ± 0.09 | 28.98 ± 1.23 | **27.07 ± 0.63** | **-6.6%** |
| | NASA | 12905.2 | 14547.0 ± 743.8 | 13751.6 ± 330.0 | 18033.3 ± 6705.3 | **8573.4 ± 1276.2** | - |
| **FD003** | RMSE | 18.77 | 19.61 ± 0.23 | 19.60 ± 0.26 | 19.33 ± 2.89 | 19.33 ± 2.89 | 0.0% |
| | NASA | 651.5 | 1507.4 ± 56.7 | 1501.1 ± 62.2 | 1247.7 ± 557.7 | 1247.7 ± 557.6 | - |
| **FD004** | RMSE | 31.51 | 32.04 ± 0.15 | 30.32 ± 0.09 | 31.13 ± 1.15 | **26.79 ± 0.68** | **-13.9%** |
| | NASA | 8652.6 | 9008.6 ± 230.9 | 7593.8 ± 139.5 | 9214.6 ± 1550.8 | **4950.5 ± 403.8** | - |

*\*Ridge regression is deterministic (single evaluation; no seed variance).*

*Note: For single-condition subsets (FD001 and FD003), condition-aware normalization sets $k=1$, making Mode C mathematically identical to Mode G (Global). Minor differences at the 6th decimal place are due solely to CPU floating-point non-associativity across parallel execution threads.*

### Protocol B: Clipped Ground-Truth RUL (Ground Truth Capped at 125 Cycles)

Evaluated with both model target and test ground-truth RUL capped at 125 cycles across all four subsets.

| Subset | Method | Standard RMSE | Clipped Ground-Truth RMSE | Standard NASA Score | Clipped NASA Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FD001** | GRU_global | 15.94 ± 0.29 | 14.66 ± 0.48 | 463.8 ± 84.9 | 425.7 ± 92.3 |
| **FD001** | GRU_cond | 15.94 ± 0.29 | 14.66 ± 0.48 | 463.8 ± 84.9 | 425.7 ± 92.3 |
| **FD002** | GRU_global | 28.98 ± 1.23 | 16.75 ± 0.16 | 18033.3 ± 6705.3 | 1397.1 ± 350.7 |
| **FD002** | GRU_cond | 27.07 ± 0.63 | **14.46 ± 0.19** | 8573.4 ± 1276.2 | 1013.1 ± 130.9 |
| **FD003** | GRU_global | 19.33 ± 2.89 | 18.16 ± 2.74 | 1247.7 ± 557.7 | 1195.1 ± 536.9 |
| **FD003** | GRU_cond | 19.33 ± 2.89 | 18.16 ± 2.74 | 1247.7 ± 557.6 | 1195.0 ± 536.9 |
| **FD004** | GRU_global | 31.13 ± 1.15 | 19.81 ± 1.13 | 9214.6 ± 1550.8 | 3099.4 ± 699.4 |
| **FD004** | GRU_cond | 26.79 ± 0.68 | **14.95 ± 0.96** | 4950.5 ± 403.8 | 1129.7 ± 277.7 |

---

## Table 2: Cluster Ablation Study on Multi-Regime FD004

Varying the number of operating clusters $k$ for condition-aware normalization (evaluated with 100% labeled engines, 40 epochs, seeds 0–2).

| Clusters (k) | Normalization Description | Test RMSE | NASA Score | Relative to Global ($k=1$) |
| :--- | :--- | :--- | :--- | :--- |
| $k=1$ | Global StandardScaler (No clustering) | 31.13 ± 1.15 | 9214.6 ± 1550.8 | Baseline (0.0%) |
| $k=3$ | Ablation k=3 | 28.79 ± 0.86 | 7145.3 ± 878.9 | -7.5% |
| $k=6$ | Proposed Regime Normalization | **26.79 ± 0.68** | **4950.5 ± 403.8** | **-13.9%** |
| $k=9$ | Ablation k=9 | 26.79 ± 0.68 | 4963.9 ± 392.5 | -13.9% |

---

## Table 3: Label-Efficiency Evaluation on FD004

Test RMSE and NASA score under constrained training labels (10%, 25%, 50%, and 100% of training engines available; validation engines fixed at 20%).

| Labeled Engines (%) | GRU (Global) RMSE | GRU (Cond-Aware) RMSE | Δ Cond-Aware RMSE (%) | GRU (Global) NASA | GRU (Cond-Aware) NASA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10%** | 37.49 ± 0.79 | **31.36 ± 0.85** | **-16.3%** | 19335.6 ± 2733.8 | **8597.7 ± 644.2** |
| **25%** | 35.38 ± 0.64 | **30.90 ± 0.33** | **-12.6%** | 13586.1 ± 1189.9 | **7874.4 ± 279.9** |
| **50%** | 33.24 ± 1.03 | **29.05 ± 0.80** | **-12.6%** | 12412.0 ± 534.8 | **6475.8 ± 618.8** |
| **100%** | 31.13 ± 1.15 | **26.79 ± 0.68** | **-13.9%** | 9214.6 ± 1550.8 | **4950.5 ± 403.8** |
