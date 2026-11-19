# Per-Regime Sensor Normalization on NASA C-MAPSS: Empirical & Reproducibility Study

This repository contains the complete codebase, data cleaning pipelines, experiments, statistical tests, and LaTeX source for the empirical research paper:

> **"Per-Regime Sensor Normalization for Remaining Useful Life Prediction under Label Scarcity and Test-Protocol Variation: An Empirical Study on C-MAPSS"**

---

## 1. Overview & Core Findings

In turbofan remaining useful life (RUL) prediction on NASA C-MAPSS:
- **Operating-regime normalization is high-leverage:** Normalizing sensors conditionally based on unsupervised $K$-Means clustering ($k=6$) fit on operating settings reduces test RMSE by **$-6.6\%$** on FD002 ($27.07$ vs. $28.98$) and **$-13.9\%$** on FD004 ($26.79$ vs. $31.13$).
- **Compensates for severe label scarcity:** On FD004, a condition-aware GRU trained on only **25% of labeled run-to-failure engines** achieves a test RMSE of **$30.90$**, matching a globally normalized model using **100% of labeled engines** ($31.13$).
- **Disentangling test protocols:** Post-hoc clipping of test ground truth at 125 cycles (Protocol B) drops nominal RMSE by $11$--$12$ cycles on multi-regime subsets without representing algorithmic improvement.

---

## 2. Dataset Acquisition & Integrity

Experiments must use the original NASA C-MAPSS archive files (`train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`).

> **Warning regarding third-party mirrors:** Common Hugging Face datasets mirrors of C-MAPSS corrupt evaluations by omitting true test RUL, stacking subsets without metadata, and duplicating engine unit numbers. Always clean and verify from the original NASA repository.

To prepare the clean dataset:
```bash
python prepare_data.py
```
This generates verified CSV files in `data_clean/`:
- `data_clean/train_FD00x.csv`
- `data_clean/test_FD00x.csv`
- `data_clean/test_RUL_FD00x.csv`

---

## 3. Repository File Map

```
├── data_clean/               # Preprocessed, verified NASA C-MAPSS data files
├── figures/                  # Publication-ready plots (.pdf and .png)
│   ├── k_selection.pdf               # Silhouette and inertia elbow (k=2..12)
│   ├── error_by_bin.pdf              # Error breakdown by degradation stage
│   ├── label_efficiency_curve.pdf    # Test RMSE across label fractions (10-100%)
│   ├── cluster_ablation_barchart.pdf # Cluster count ablation (k=1, 3, 6, 9)
│   └── fd003_validation_curves.pdf   # Validation loss curves diagnosing early stopping
├── logs/                     # Timestamped unbuffered execution logs
├── results/                  # Experimental results, metrics, and test predictions
│   ├── preds/                        # Per-engine test prediction CSVs
│   ├── baseline_runs.csv             # Deterministic Ridge & RF global baselines
│   ├── rf_cond_runs.csv              # Condition-aware Random Forest runs
│   ├── gru_runs_v2.csv               # Main benchmark 3-seed GRU runs
│   ├── extra_runs.csv                # Cluster ablation and label efficiency runs
│   ├── protocol_rmse.csv             # Protocol A (standard) vs Protocol B (clipped)
│   ├── error_by_bin.csv              # Error per true-RUL degradation bin
│   ├── stats.csv                     # Paired t-tests, Wilcoxon, and bootstrap CIs
│   ├── stats_notes.md                # Audit notes on per-engine test statistics
│   ├── per_seed.md / per_seed.tex    # Full per-seed benchmark breakdown
│   └── final_tables.md               # Formatted benchmark tables
├── model.py                  # PyTorch model definitions (GRURegressor)
├── preprocess.py             # Feature extraction, KMeans clustering, epsilon rule
├── utils.py                  # Sliding window generator, RMSE, NASA score
├── train.py                  # PyTorch training loop with engine-level validation
├── baseline.py               # Ridge and Random Forest baseline runner
├── run_main_v2.py            # Main benchmark execution runner (seeds 0-2)
├── run_extra.py              # Ablation (k=3, 9) and label scarcity (10, 25, 50%) runner
├── stage1_analysis.py        # Clustering silhouette, error by bin, paired statistics
├── stage2_tune.py            # Validation-only tuning of GRU hyperparameters
├── check_latex.py            # Rigorous LaTeX syntax, brace, and asset validator
├── paper.tex                 # Complete research paper (IEEEtran conference format)
├── references.bib            # BibTeX bibliography file
├── number_audit.md           # Numerical audit tracing all paper numbers to source files
└── README.md                 # This documentation file
```

---

## 4. Commands in Exact Execution Order

All experiments can be reproduced sequentially from scratch:

```bash
# 1. Clean raw NASA C-MAPSS data into data_clean/
python prepare_data.py

# 2. Run deterministic Ridge and Random Forest global baselines
python baseline.py

# 3. Run Condition-Aware Random Forest baselines (seeds 0, 1, 2)
python rf_cond.py

# 4. Run Main GRU Benchmarks across FD001-FD004 (seeds 0, 1, 2; batch 512; 40 epochs)
python -u run_main_v2.py > logs/run_main_v2.log 2>&1

# 5. Run Cluster Ablation (k=3, 9) and Label Scarcity (10%, 25%, 50%) on FD004
python -u run_extra.py > logs/run_extra.log 2>&1

# 6. Compute Protocol A vs Protocol B comparisons from saved predictions
python compute_protocols.py

# 7. Execute Stage 1 Analysis: Silhouette (k=2..12), Error by RUL Bin, and Bootstrap CIs
python stage1_analysis.py

# 8. Execute Stage 2 Validation-Only Hyperparameter Tuning
python stage2_tune.py

# 9. Validate LaTeX structure, brace balance, figure assets, and citation keys
python check_latex.py
```

---

## 5. Experimental Hyperparameters & Seeds

- **Random Seeds:** `0, 1, 2` for all main benchmark comparisons.
- **Windowing:** Sequence length $= 30$ cycles, stride $= 5$ cycles.
- **GRU Architecture:** 2 layers, hidden dimension $= 64$, dropout $= 0.2$.
- **Training Protocol:** Adam optimizer, $lr = 2\times 10^{-3}$, batch size $= 512$, maximum $40$ epochs, gradient clipping at $1.0$, $\text{ReduceLROnPlateau}$ (patience 3, factor 0.5), early stopping patience $8$.
- **Target Scaling:** Scaled via $y / 125.0$ during training; unscaled by $125.0$ at evaluation.
- **Validation Split:** $20\%$ of training engines held out by random seed (validation engines never reduced).
- **Clustering:** $K$-Means with $k=6$ (for multi-regime FD002/FD004) fit on training operating settings with `random_state=0`. Epsilon variance threshold $= 10^{-10}$.
- **Hardware:** Standard CPU multi-threading (`torch.set_num_threads(os.cpu_count())`). No GPU required.

---

## 6. Numerical Audit & Verification

Every numerical figure, percentage, and $p$-value reported in `paper.tex` is audited line-by-line in [number_audit.md](file:///c:/Users/lenovo/Desktop/cmapss-paper/number_audit.md).
