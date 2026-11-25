# CLEANUP MANIFEST

**Date:** 2026-11-23  
**Repo:** `cmapss-paper`  

---

## 1. Frozen Baseline Files (Preserved & Verified via `FROZEN.sha256`)
- `data_clean/train_FD001.csv` .. `data_clean/train_FD004.csv`
- `data_clean/test_FD001.csv` .. `data_clean/test_FD004.csv`
- `data_clean/test_RUL_FD001.csv` .. `data_clean/test_RUL_FD004.csv`
- `results/gru_runs.csv`, `results/gru_runs_v2.csv`, `results/gru_runs_v3.csv`
- `results/baseline_runs.csv`
- `results/rf_cond_runs.csv`
- `results/extra_runs.csv`
- `results/protocol_rmse.csv`
- `results/cluster_metrics_FD002.csv`, `results/cluster_metrics_FD004.csv`
- `results/rul_bin_analysis.csv`

## 2. Confirmatory Experimental Results Files (Preserved)
- `results/e1_pilot_runs.csv` (15 runs across C1, C2, C3, C4, C5)
- `results/e1_pilot_summary.csv`
- `results/tier1a_100epochs_runs.csv` (100-epoch sensitivity runs)
- `results/tier1c_ridge_runs.csv` (Linear Ridge regression runs)
- `results/e5_rigorous_statistics.csv` (Double bootstrap and hypothesis tests)
- `results/mechanism_variance_explained.csv` (Sensor between-regime variance decomposition)

## 3. Publication Manuscript & Audit Files (Preserved)
- `paper.tex` (Final revised manuscript, IEEEtran conference format)
- `references.bib` (26 verified entries with full DOIs)
- `references_verification.csv` (Verification audit table)
- `claims_evidence.csv` (Evidence mapping table)
- `AUDIT.md` (Comprehensive manuscript audit)
- `REVIEW_RESPONSE.md` (Point-by-point rebuttal to 14 hostile objections)
- `STATE.md` (Project state summary)
- `E1_REPORT.md` (E1 experimental report)
- `equivalence_protocol.md` (TOST pre-registered protocol)
- `FROZEN.sha256` (Cryptographic checksum manifest)

## 4. Archival and Cleanup Actions
- Delete temporary probe/test scripts: `test_m14.py`, `test_m3.py`, `test_m9.py`, `make_frozen.py`, `audit_calc.py`
- Remove Python bytecode caches (`__pycache__`)
- Retain core modular pipeline in `scripts/`: `run_e1_pilot.py`, `run_tier1a.py`, `e1_baselines.py`, `e2_e3_label_scarcity_tost.py`, `e4_mechanism.py`, `e5_e6_statistics.py`
