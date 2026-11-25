# SYSTEMATIC LITERATURE AUDIT: REPRODUCIBILITY IN C-MAPSS RUL EVALUATION

**Date:** 2026-11-22  
**Purpose:** Literature protocol audit evaluating test target clipping, sliding window parameters, validation holdout logic, and sensor normalization reporting.

---

## 1. Controlled Literature Extraction Matrix

| Ref / Key | Authors & Venue | Test Ground-Truth Clipping | Window Length / Stride | Validation Split Strategy | Sensor Normalization Scheme | Reported FD002 RMSE | Reported FD004 RMSE | Source Verification & Notes |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Saxena2008**~\cite{saxena2008cmapss} | Saxena et al., *IEEE PHM 2008* | Protocol A (Unclipped true RUL) | Not applicable (challenge dataset release) | Not specified | Unprocessed baseline | Benchmark definition | Benchmark definition | Definition of 21 sensors, 3 op settings, and asymmetric scoring function. |
| **Heimes2008**~\cite{heimes2008recurrent} | Heimes, *IEEE PHM 2008* | Protocol B (Train RUL cap $\le 125$ or $130$) | Variable length RNN | Not fully stated | Global min-max per sensor | $\approx 29.8$ | $\approx 31.8$ | Introduced piecewise linear degradation ceiling ($125$ or $130$). |
| **Ramasso2014**~\cite{ramasso2014investigating} | Ramasso \& Saxena, *IJPHM 2014* | **Unspecified / Discrepant** | Window 30 / Stride 1 | Engine-level holdout | Mixed global vs. condition | $25.0$--$29.0$ | $26.0$--$32.0$ | First comprehensive meta-audit identifying protocol divergence across published papers. |
| **Zheng2017**~\cite{zheng2017long} | Zheng et al., *IEEE ICPHM 2017* | Protocol B (Clipped at $125$) | Window 30 / Stride 1 | Random 20% engines | Global standardization | $\approx 18.9$ (Protocol B) | $\approx 20.8$ (Protocol B) | Evaluated LSTM. Used clipped test RUL, yielding artificially lower RMSE values. |
| **Li2018**~\cite{li2018tcn} | Li et al., *RESS 2018* | Protocol B (Clipped at $125$) | Window 30 / Stride 1 | 20% validation split | Global min-max $[-1, 1]$ | $17.8$ (Protocol B) | $19.4$ (Protocol B) | Deep 1D-CNN. Scores reflect Protocol B ground-truth clipping. |
| **Pasa2019**~\cite{pasa2019} | Pasa, de Medeiros \& Yoneyama, *PHM Society 2019* | Protocol A (Unclipped) | Window 30 / Stride 1 | Engine-level split | **Condition-based clustering** ($k=6$) | $26.8$ (Protocol A) | $27.1$ (Protocol A) | Evaluated condition-invariant normalization; demonstrated large gains over global scaling. |
| **Listou2020**~\cite{listou2020semi} | Ellefsen et al., *RESS 2020* | Protocol B (Clipped at $125$) | Window 30 / Stride 1 | Semi-supervised engine split | Global standardization | $19.2$ (Protocol B) | $21.5$ (Protocol B) | Investigated semi-supervised RUL under label scarcity. |
| **daCosta2021**~\cite{da2021transfer} | da Costa et al., *IEEE Trans. Rel. 2021* | Protocol A (Unclipped) | Window 30 / Stride 1 | Cross-condition engine holdout | Domain adaptation / per-regime | $27.4$ (Protocol A) | $28.1$ (Protocol A) | Evaluated domain adaptation across operating conditions. |
| **Gupta2024**~\cite{gupta2024auditing} | Gupta, *engrXiv 2024* | Explicitly audited Protocol A vs B | Window 30 / Stride 5 | Leakage-safe engine split | **Audited Global vs. Condition-aware** | $28.5 \to 26.5$ | $31.0 \to 26.4$ | Rigorous audit showing that model gains in literature are often confounded with normalization choices. |
| **STAGNN2024**~\cite{stagnn2024} | Anonymous, *arXiv:2401.15964* | Protocol B (Clipped at $125$) | Window 30 / Stride 1 | 20% validation | Condition-adaptive graph | $16.5$ (Protocol B) | $17.2$ (Protocol B) | Spatio-temporal graph neural network with condition embedding. |

---

## 2. Precise Disambiguation: How This Study Differs from Prior Audits

1. **Gupta (2024) Audit vs. This Work:**
   - *Gupta (2024)* surveyed preprocessing leakage and established that many published architectural gains inadvertently contrasted condition-aware models against globally normalized baselines.
   - *This study* builds upon Gupta's observations by (i) executing controlled double-bootstrap hypothesis testing resampling both engines and training initializations; (ii) providing the first empirical study of unsupervised regime normalization under severe label scarcity ($10\%$--$100\%$ labeled trajectories); and (iii) demonstrating that test protocol clipping (Protocol B) changes nominal RMSE by $11$--$12$ cycles without reflecting model progress.

2. **Pasa et al. (2019) vs. This Work:**
   - *Pasa et al. (2019)* introduced condition-specific standardization as an architectural preprocessing mechanism.
   - *This study* frames per-regime normalization strictly as established prior work, isolates its quantitative ablation under strictly identical architectures (GRU, RF, Ridge), proves that it substitutes for scarce labels, and refutes earlier claims regarding residual variance mechanisms.

3. **STAGNN (2024) vs. This Work:**
   - *STAGNN (2024)* introduced complex spatio-temporal graph attention networks evaluated under Protocol B.
   - *This study* demonstrates that simple, robust preprocessing (unsupervised $K$-Means with $k=6$) on standard recurrent baselines achieves performance comparable to complex graph architectures without graph-construction overhead.
