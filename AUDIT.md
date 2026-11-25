# AUDIT OF v2 MANUSCRIPT: COMPREHENSIVE FINDINGS MATRIX

**Date:** 2026-11-22  
**Role:** Senior PHM/RUL Researcher, Statistician, Internal Reviewer 2  
**Target Submission:** Defensible Empirical Prognostics Benchmark  

This audit documents every critical methodological, statistical, algorithmic, and textual defect identified in the v2 manuscript (*"Operating-Condition Normalization in C-MAPSS Turbofan Prognostics: A Controlled Empirical Ablation, Label Scarcity, and Evaluation-Protocol Study"*), accompanied by exact recomputations, root-cause analyses, and corrections.

---

## 1. Summary of Findings

| Item | Category | Claim in v2 | Audited Reality / Recomputed Value | Verdict & Correction |
|:---|:---|:---|:---|:---|
| **a) $\eta^2$ Metric** | Statistical / Mechanism | $\eta^2 = 14.29\%$ (FD002) and $9.52\%$ (FD004) residual regime variance after per-regime norm. | Exactly equals $3/21$ and $2/21$ (sensor count ratio). True active sensor between-regime variance under Mode C is mathematically $\sim 0$ ($< 10^{-24}$). The $3/21$ and $2/21$ were constant sensors ($0/0$ division artifacts handled by epsilon rule). | **Refuted / Corrected:** Delete $14.29\%$ / $9.52\%$ claim. State that per-regime z-scoring eliminates between-regime sensor variance to machine precision ($\eta^2 \approx 0$). Remove unsupported inference about recurrent hidden states. |
| **b) TOST Equivalence** | Statistical | Cond@25% ($30.90 \pm 0.33$) equivalent to Global@100% ($31.13 \pm 1.15$) within $\pm 1.5$ cycles ($n=3$). | Welch $90\%$ CI is $[-2.07, +1.61]$ (width $3.68$, $p = 0.0946$). Pooled $90\%$ CI is $[-1.70, +1.24]$ ($p = 0.0699$). Neither fits within $[-1.5, +1.5]$ at $\alpha=0.05$. Pilot $n=3$ has insufficient power ($df=2.33$). | **Refuted:** v2 overclaimed statistical equivalence for pilot runs. Rephrase as "directionally within nominal range but equivalence test does not reject non-equivalence at $\alpha=0.05$ with $n=3$ ($p=0.095$)". Confirmatory protocol specified. |
| **c) Engine Counts** | Methodology | $18, 44, 89, 179$ engines remain after a "20% holdout" at $10\%, 25\%, 50\%, 100\%$. | Split logic held out $20\%$ of training engines ($50$ engines) as validation set, leaving $199$ base training engines. Applying fractions yields $20, 50, 100, 199$ training engines ($8\%, 20\%, 40\%, 80\%$ of total fleet of $249$). The numbers $18, 44, 89, 179$ resulted from taking fractions of total engines minus validation twice. | **Clarified & Corrected:** Explicitly report both training and validation engine sets: at $10\%$, $20$ train + $50$ val; at $25\%$, $50$ train + $50$ val; at $50\%$, $100$ train + $50$ val; at $100\%$, $199$ train + $50$ val. |
| **d) Hyperparameter Tuning** | Protocol / Leakage | $h=64$ retained without test consultation; tuning seeds disjoint from evaluation seeds. | In `stage2_tune.py`, tuning was evaluated on seeds $0$ and $1$ on FD004. Global was improved by $3.66\%$ on validation ($h=128$), while Cond improved by only $1.07\%$ (below $2\%$ threshold). However, seeds $0$ and $1$ overlap pilot benchmark seeds ($0, 1, 2$). | **Flagged:** Fully disclose that pilot tuning on seeds $0, 1$ shared initialization seeds with pilot evaluation. Confirmatory runs mandate disjoint seeds ($100, 101$ for tuning, $10$--$19$ for testing). |
| **e) $k=9$ vs. $k=6$** | Reproducibility | $k=9$ gives $26.79 \pm 0.68$, identical to $k=6$; claimed identical predictions. | Both $k=6$ and $k=9$ did run independently. Predictions differ per engine by up to $1.62$ cycles (mean difference $\approx 0.22$ cycles). By numerical coincidence, their 2-decimal rounded means and standard deviations coincide ($26.79 \pm 0.68$). | **Verified & Nuanced:** Confirmed that $k=9$ actually executed with 9 clusters; clarify that individual engine predictions differ slightly ($\text{MAE} \approx 0.22$ cycles) while aggregate test RMSE is practically identical. |
| **f) RF on $k=1$ Discrepancy** | Reproducibility | RF Global ($18.49 \pm 0.13$) vs RF Cond ($18.48 \pm 0.13$) on FD001. Footnote blames std divisors. | `preprocess.py` Mode G used `StandardScaler` ($\text{ddof}=0$), whereas Mode C loop used pandas `.std()` ($\text{ddof}=1$). With $N \approx 20{,}000$, $\sqrt{N/(N-1)} \approx 1.000024$. Combined with multi-threaded `RandomForestRegressor(n_jobs=-1)` tree order, this produced the $0.01$ RMSE variation. | **Verified:** The footnote explanation is confirmed correct and mathematically verified. |
| **g) Abstract / Intro Overclaims** | Scientific Integrity | "Robust gains" across all multi-regime subsets; unconditioned claims of significance. | On FD002, seed-level paired $t$-test has $p = 0.126$ ($n=3$); double-bootstrap CI $[-3.61, -0.23]$ is close to zero; NASA score CI $[-26883, +154]$ includes zero. Gain on FD002 is modest ($-6.6\%$). | **Corrected:** Replaced promotional phrasing. Abstract and intro explicitly report FD002 as a modest reduction ($-6.6\%$) with seed-level $p=0.126$ and NASA score CI spanning zero. Headline strong significance is restricted to FD004 ($-13.9\%$). |
| **h) Omitted Baselines** | Transparency | Omission of regime one-hot, raw settings inputs, min-max, and alternate architectures. | v2 deleted the limitation disclosure admitting that one-hot regime indicators and raw operating setting inputs were not evaluated. | **Restored:** Restored prominent disclosure in Limitations and experimental design. |
| **i) Formatting & LaTeX Errors** | Presentation | Literal Markdown `**99.19%**` in LaTeX; bibtex references missing/uncompiled. | Checked `paper.tex` line 282: contained literal markdown bold markers. Citation keys `gupta2024auditing`, `pasa2019`, `stagnn2024` needed audit. | **Fixed:** Removed all literal markdown. Balanced braces and verified all 12 BibTeX entries in `references.bib`. |
| **j) Protocol B Framing** | Metric Interpretation | Protocol B metric reduction is "entirely an artifact of truncating healthy early-life engines". | Under Protocol B, FD004 RMSE drops from $19.81$ (global) to $14.95$ (cond)---a relative reduction of $-24.5\%$, which is *larger* than under Protocol A ($-13.9\%$). | **Refined:** The baseline level shift ($\approx 11$ cycles) is an artifact of truncation, but the *relative benefit* of regime normalization is preserved and amplified under Protocol B. |

---

## 2. Detailed Technical Evidence

### 2.1 The $\eta^2$ Dissection (Item a)
In `run_e4.py`, $\eta^2 = SS_{\text{between}} / SS_{\text{total}}$ was calculated across all 21 sensor columns for FD002 and FD004:
- Active dynamic sensors (e.g., sensor 2, 3, 4, 7, 8, etc.) have $\eta^2 \approx 0.992$ under raw and global normalization.
- In condition-aware mode (`mode='C'`), each active sensor is centered to its cluster mean: $\mu_{j,c} = \frac{1}{N_c}\sum_{i \in c} x_{ij}$. Thus, within each cluster, the sample mean is identically 0. Consequently, $SS_{\text{between}} = \sum_c N_c (\bar{x}_c - \bar{x})^2 \equiv 0$, giving $\eta^2 = 0.0$ to machine precision ($\sim 10^{-26}$).
- However, for sensors with constant values within certain clusters:
  - On FD002, 3 sensors (sensor 1, 5, 19) triggered the epsilon rule ($c_{\text{std}} < 1e-10 \to c_{\text{std}} = 1.0$), resulting in unscaled cluster offsets where the code returned $\eta^2 = 1.0$.
  - Average over all 21 sensors: $(3 \times 1.0 + 18 \times 0.0) / 21 = 3/21 = 0.142857 = 14.29\%$.
  - On FD004, exactly 2 sensors (sensor 1, 5) triggered this, yielding $(2 \times 1.0 + 19 \times 0.0) / 21 = 2/21 = 0.095238 = 9.52\%$.
- **Conclusion:** The numbers $14.29\%$ and $9.52\%$ in v2 were mathematical artifacts reflecting sensor counts ($3/21$ and $2/21$), not residual physical variance. In truth, per-regime z-scoring eliminates between-regime variance across all active sensors.

### 2.2 TOST Equivalence Analysis (Item b)
- Group 1 (FD004 Cond @ 25% labels): $m_1 = 30.9009$, $s_1 = 0.3323$, $n_1 = 3$.
- Group 2 (FD004 Global @ 100% labels): $m_2 = 31.1307$, $s_2 = 1.1528$, $n_2 = 3$.
- Observed Difference: $\Delta = -0.2298$ cycles.
- Equivalence Margin: $\delta = \pm 1.5$ cycles.
- Standard Error of Difference: $SE = \sqrt{0.3323^2/3 + 1.1528^2/3} = 0.6907$.
- Welch Satterthwaite degrees of freedom: $df = 2.327$.
- Critical $t$ at $\alpha = 0.05$ (one-sided, $90\%$ confidence level): $t_{\text{crit}} = 2.659$.
- Welch $90\%$ Confidence Interval: $[-0.230 - 2.659(0.691), -0.230 + 2.659(0.691)] = [-2.066, +1.606]$.
- Since the upper bound ($+1.606$) exceeds $+1.5$ and the lower bound ($-2.066$) falls below $-1.5$, **the hypothesis of non-equivalence cannot be rejected at $\alpha = 0.05$** ($p_{\text{TOST}} = 0.0946$).
- **Conclusion:** While the empirical difference ($-0.23$) is small, claiming formal statistical equivalence under $n=3$ is statistically invalid. Confirmatory 10-seed testing is required.

---

## 3. Confirmatory Audits & Additions (Phase 0, E1 Pilot, Tier 1A, Tier 1C, M9)

### 3.1 Input Dimensions & Integrity (M14)
- **C1 (Global):** 24 features (21 sensors + 3 operational settings, with 3 constant sensors dropped -> 24 dynamic channels).
- **C2 (Per-Regime):** 24 features (identical sensor dimensionality).
- **C3 (Global + One-Hot):** 30 features (24 base features + 6 one-hot regime columns, verified unscaled binary 0.0/1.0).
- **C4 (Global + Standardized Settings):** 27 features (24 base features + 3 standardized operating settings, fit strictly on training set).
- **C5 (Per-Regime + One-Hot):** 30 features (24 per-regime standardized features + 6 one-hot regime columns).

### 3.2 Linear Model Generalization (Tier 1C)
- Per-regime normalization is verified to benefit linear models without neural capacity:
  - FD002 Ridge RMSE: $32.36 \to 28.92$ ($-10.6\%$ reduction).
  - FD004 Ridge RMSE: $35.72 \to 30.53$ ($-14.5\%$ reduction).
- Proves that regime standardization is a fundamental data-space transformation, not an idiosyncratic neural artifact.

### 3.3 100-Epoch Extended Training Sensitivity (Tier 1A)
- At 100 epochs, global model C1 reaches validation convergence (mean stopped epoch $77.7 \pm 23.0$, val RMSE $17.41 \pm 0.26$, test RMSE $30.23 \pm 0.06$).
- Per-regime model C2 reaches validation convergence (mean stopped epoch $54.3 \pm 19.4$, val RMSE $13.70 \pm 0.61$, test RMSE $26.65 \pm 0.43$).
- The performance gap at 100 epochs is $30.23 - 26.65 = \mathbf{3.58\text{ cycles}} \ge 3.0$ cycles, confirming **Branch A**. Extended training does not eliminate the normalization advantage.

### 3.4 Degradation Error Share Decomposition (M9)
- Total squared error reduction across 248 test engines on FD004: $60{,}899.71$.
- Bin $0$--$30$ cycles: $14.66\%$ share ($15.02 \to 7.35$ RMSE).
- Bin $30$--$60$ cycles: $17.21\%$ share ($23.08 \to 16.03$ RMSE).
- Bin $60$--$125$ cycles: $16.98\%$ share ($20.95 \to 18.03$ RMSE).
- Bin $>125$ cycles: $51.15\%$ share ($49.52 \to 44.58$ RMSE).
- Near-failure engines ($0$--$60$ cycles) account for $31.87\%$ of total squared error reduction.

### 3.5 Paired TOST Equivalence at 10% Labels
- Under paired comparisons across seeds, cond@10% ($31.36 \pm 0.85$) vs global@100% ($31.13 \pm 1.15$) yields mean paired delta $+0.2335$ cycles ($95\%$ CI: $[-0.8832, +1.3502]$).
- Because the entire $95\%$ CI is strictly within $\pm 1.5$ cycles, paired TOST rejects non-equivalence ($p = 0.0198 < 0.05$).
- Under two-sample Welch TOST, $p = 0.1031$, showing power constraints at $n=3$ seeds.
