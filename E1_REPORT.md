# E1 Pilot Report: Regime-Information Baselines vs. Normalization on FD004

**Date:** 2026-11-22  
**Commit:** `0c323b0c7e9e83228b00e1fce5c9cd9197fef71e`  
**Pipeline:** GRU (2 layers, $h=64$, dropout 0.2), window 30, stride 5, batch 512, Adam lr 2e-3, target $y/125.0$, early stopping 8 (max 40 epochs).

---
## 1. Step 0 Pre-Run Checks
1. **Pilot Reproduction (Seed 0):**
   - **C1 (Global):** Test RMSE (Protocol A) = **30.8317**, NASA = **8908.6**, Epochs = **40** (matches saved pilot seed 0 exactly).
   - **C2 (Per-Regime):** Test RMSE (Protocol A) = **26.3267**, NASA = **4711.7**, Epochs = **39** (matches saved pilot seed 0 exactly).
2. **Train/Validation Engine Split Counts:**
   - FD004 total fleet: **249** engines.
   - Seed-level 20% holdout split: **199** training engines, **50** validation engines (exactly 199/50).
3. **Data Leakage & Clustering Check:**
   - $K$-Means ($k=6$) fit strictly on training operating settings (`train_df[op_cols]`).
   - Test engines mapped via nearest training centroids with zero test statistics leakage.

---

## 2. Experimental Results across Conditions (Seeds 0, 1, 2)

### Table 1: Summary Statistics (Mean $\pm$ Standard Deviation across Seeds 0, 1, 2)
| Condition | Description | Mean Val RMSE | Test RMSE (Protocol A) | Test RMSE (Protocol B) | NASA Score | Mean Epochs | Mean Wall Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1** | Global Normalization | $18.26 \pm 1.35$ | $31.13 \pm 1.15$ | $19.81 \pm 1.13$ | $9214.6 \pm 1550.8$ | $40.0$ | $455.2\text{s}$ |
| **C2** | Per-Regime Normalization ($k=6$) | $\mathbf{13.94 \pm 0.83}$ | $\mathbf{26.79 \pm 0.68}$ | $\mathbf{14.95 \pm 0.96}$ | $\mathbf{4950.5 \pm 403.8}$ | $39.7$ | $269.0\text{s}$ |
| **C3** | Global + Regime One-Hot ($6$ dims) | $17.72 \pm 1.09$ | $30.65 \pm 0.25$ | $19.32 \pm 0.43$ | $9835.3 \pm 1666.2$ | $40.0$ | $301.1\text{s}$ |
| **C4** | Global + Standardized Raw Settings | $18.26 \pm 1.39$ | $31.18 \pm 0.86$ | $19.89 \pm 0.96$ | $9372.0 \pm 980.7$ | $40.0$ | $299.3\text{s}$ |
| **C5** | Per-Regime + Regime One-Hot | $13.87 \pm 0.57$ | $27.02 \pm 0.18$ | $14.67 \pm 0.55$ | $5116.5 \pm 139.1$ | $34.7$ | $267.5\text{s}$ |

---

## 3. Paired Seed-Level Differences

### Table 2: Raw Per-Seed Metrics and Paired Deltas (Test RMSE Protocol A)
| Condition | Seed 0 | Seed 1 | Seed 2 | Mean vs. C1 ($\Delta_{\text{C1}}$) | Mean vs. C2 ($\Delta_{\text{C2}}$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **C1** (Global) | $30.83$ | $30.16$ | $32.40$ | $0.00$ | $+4.34$ |
| **C2** (Per-Regime) | $26.33$ | $26.48$ | $27.57$ | $-4.34$ | $0.00$ |
| **C3** (Global + One-Hot) | $30.62$ | $30.92$ | $30.41$ | $-0.48$ ($-0.21, +0.75, -1.99$) | $+3.86$ ($+4.29, +4.43, +2.84$) |
| **C4** (Global + Raw Settings) | $30.78$ | $30.59$ | $32.16$ | $+0.05$ ($-0.05, +0.43, -0.24$) | $+4.39$ ($+4.46, +4.11, +4.59$) |
| **C5** (Per-Regime + One-Hot) | $26.88$ | $27.23$ | $26.96$ | $-4.11$ ($-3.96, -2.93, -5.44$) | $+0.23$ ($+0.55, +0.75, -0.61$) |

- **Comparison vs. Pooled Standard Deviation:**
  - Pooled standard deviation between C2 and C3: $s_{\text{pooled}} = 0.5115$. Mean difference $\Delta(\text{C3} - \text{C2}) = +3.86$ cycles, which is **$7.54$ pooled standard deviations** worse than C2.
  - Pooled standard deviation between C2 and C4: $s_{\text{pooled}} = 0.7719$. Mean difference $\Delta(\text{C4} - \text{C2}) = +4.39$ cycles, which is **$5.68$ pooled standard deviations** worse than C2.

---

## 4. Validation vs. Test Ranking Disagreements
Across the 5 conditions within each seed:
- **Seed 0:** C2 and C5 swap ranks 1 and 2 (Val: C5 $14.11$ vs. C2 $14.18$; Test: C2 $26.33$ vs. C5 $26.88$). Both decisively outperform global variants C1, C3, and C4 on both validation and test.
- **Seed 1:** C1, C3, and C4 invert ranks between validation and test (C3 Val: $16.82$ rank 3, but Test: $30.92$ rank 5; C1 Val: $17.13$ rank 4, Test: $30.16$ rank 3).
- **Seed 2:** C1 and C4 swap ranks 4 and 5; C2 and C5 swap ranks 1 and 2 (Val: C5 $14.28$ rank 1, C2 $14.62$ rank 2; Test: C2 $27.57$ rank 2, C5 $26.96$ rank 1).
- **Key Observation:** Validation RMSE cleanly separates the per-regime family (C2, C5 $\approx 13.9$) from the global family (C1, C3, C4 $\approx 17.7$--$18.3$) in perfect agreement with test RMSE. However, fine-grained ranking within each cluster of methods exhibits mild validation-test rank inversion.

---

## 5. Factual Conclusion (Strict Criterion Evaluation)
Criterion **(ii)** unequivocally holds: **C3 and C4 are clearly worse than C2 by more than 2 pooled standard deviations ($7.54$ and $5.68$ pooled standard deviations, respectively).** 

Providing the model with auxiliary regime information—whether as one-hot indicators (C3) or standardized raw operating settings (C4)—reduces test RMSE by at most $0.48$ cycles over global baseline C1 and fails to recover the $4.34$-cycle gain achieved by per-regime normalization. Per-regime normalization (C2) and its one-hot hybrid (C5) achieve essentially identical test performance ($26.79$ vs. $27.02$, $\Delta = +0.23$ cycles). Therefore, operating-condition normalization provides a structural preprocessing benefit that cannot be replicated merely by providing regime indicators as extra input features to a recurrent model.
