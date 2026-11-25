# PROJECT REPOSITORY STATE & BENCHMARK AUDIT

**Updated:** 2026-11-23  
**Target Paper:** *Operating-Condition Normalization in C-MAPSS Turbofan Prognostics: A Controlled Empirical Ablation, Label Scarcity, and Evaluation-Protocol Study*  
**Author:** Mr. Neelabh Sharma (Department of Mechanical and Aerospace Engineering, PHM Laboratory)  
**Role:** Co-Author, Statistician, and Internal Reviewer 2  

---

## 1. Executive Summary & Gate Status

- **Gate 0 Verification:** **PASSED**
  - **M14 (Input Dimensionality):** Verified exact feature vector dimensions:
    - C1 (Global): 24 features (21 sensors + 3 settings minus 3 zero-variance channels).
    - C2 (Per-Regime): 24 features (identical dynamic channels).
    - C3 (Global + One-Hot): 30 features (24 base features + 6 unscaled binary 0.0/1.0 indicators).
    - C4 (Global + Standardized Settings): 27 features (24 base features + 3 standardized operating settings).
    - C5 (Per-Regime + One-Hot): 30 features (24 per-regime features + 6 unscaled binary 0.0/1.0 indicators).
  - **M3 (Engine Split Validation):** Verified 80/20 train/validation split at engine unit level:
    - Total fleet FD004: 249 engines.
    - Training engines: 199. Validation engines: 50.
    - Label scarcity fractions (10%, 25%, 50%, 100%): correspond to 20, 50, 100, 199 labeled training engines (with 50 validation engines fixed).
  - **M1 ($\eta^2$ Between-Regime Variance):** Recomputed and confirmed that for all dynamic active sensors, between-regime variance under Mode C is $\sim 0$ ($< 10^{-24}$). The v2 numbers (14.29% and 9.52%) were mathematical artifacts of constant sensors triggering the epsilon rule ($3/21$ and $2/21$).
  - **M9 (Error Share Distribution):** Total squared error reduction across 248 test engines on FD004 is $60{,}899.71$:
    - $0$--$30$ cycles: $14.66\%$ share ($15.02 \to 7.35$ RMSE).
    - $30$--$60$ cycles: $17.21\%$ share ($23.08 \to 16.03$ RMSE).
    - $60$--$125$ cycles: $16.98\%$ share ($20.95 \to 18.03$ RMSE).
    - $>125$ cycles: $51.15\%$ share ($49.52 \to 44.58$ RMSE).
    - Near-failure engines ($0$--$60$ cycles) account for $31.87\%$ of total squared error reduction.

---

## 2. Decision Tree & Confirmatory Branch Status

- **Branch Confirmed:** **Branch A**
  - *Hypothesis Tested (Tier 1A):* Does extending training to 100 epochs eliminate the gap between global (C1) and per-regime (C2) models on FD004?
  - *Result:* 
    - C1 reached validation plateau (mean stopped epoch $77.7 \pm 23.0$), yielding test RMSE $30.23 \pm 0.06$.
    - C2 reached validation plateau (mean stopped epoch $54.3 \pm 19.4$), yielding test RMSE $26.65 \pm 0.43$.
    - The performance gap at 100 epochs is $30.23 - 26.65 = \mathbf{3.58\text{ cycles}} \ge 3.0$ cycles.
    - **Conclusion:** Branch A is decisively confirmed. Global normalization's deficit is not an undertraining artifact.

---

## 3. Inventory of Executed Experiments & Results Files

| Experiment | Status | Result File | Summary / Key Findings |
|:---|:---:|:---|:---|
| **Pilot Runs (Seeds 0-2)** | Frozen | `results/gru_runs_v3.csv`, `results/protocol_rmse.csv` | FD001: 15.94; FD002: 28.98 -> 27.07 (-6.6%); FD003: 19.33; FD004: 31.13 -> 26.79 (-13.9%). |
| **E1 Pilot (Auxiliary Inputs)** | Executed | `results/e1_pilot_runs.csv`, `results/e1_pilot_summary.csv` | C1: 31.13; C2: 26.79; C3: 30.65; C4: 31.18; C5: 27.02. Auxiliary features fail to close gap (>5 pooled std worse than C2). |
| **Tier 1A (100-Epoch Sensitivity)** | Executed | `results/tier1a_100epochs_runs.csv` | C1: 30.23 +/- 0.06; C2: 26.65 +/- 0.43. Gap = 3.58 cycles >= 3.0 cycles. |
| **Tier 1C (Linear Ridge Baseline)** | Executed | `results/tier1c_ridge_runs.csv` | Ridge FD002: 32.36 -> 28.92 (-10.6%); Ridge FD004: 35.72 -> 30.53 (-14.5%). Confirms linear generality. |
| **Label Scarcity & TOST (E2/E3)** | Executed | `results/extra_runs.csv`, `calc_tost.py` | 10%: 37.49 -> 31.36; 25%: 35.38 -> 30.90; 50%: 33.24 -> 29.05. Paired TOST for 10% cond vs 100% global (delta=1.5): p=0.0198 < 0.05. |
| **Mechanism / Variance Analysis (E4)**| Executed | `results/mechanism_variance_explained.csv` | Raw telemetry variance explained by regime is 99.2%; Mode C active sensor between-regime variance is ~0. |
| **Statistical Resampling (E5/E6)** | Executed | `results/e5_rigorous_statistics.csv` | Double bootstrap B=10,000 95% CIs: FD002 [-3.61, -0.23]; FD004 [-5.80, -2.89]. |

---

## 4. Manuscript, Code & Data Verification

- **Manuscript:** `paper.tex` (Complete, IEEEtran conference format, updated author Mr. Neelabh Sharma, Branch A reframed, zero promotional words).
- **Bibliography:** `references.bib` (26 verified entries with full DOIs, checked against real records).
- **Audit Documentation:**
  - `AUDIT.md`: Complete audit of v2 defects and recomputations.
  - `REVIEW_RESPONSE.md`: Point-by-point rebuttal addressing 14 hostile reviewer objections.
  - `claims_evidence.csv`: 44 rows mapping every manuscript claim to exact result files and rows.
  - `references_verification.csv`: Verification status and DOIs for all 26 references.
  - `FROZEN.sha256`: Cryptographic integrity checksums for all pilot and experimental files.
