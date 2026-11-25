# CHANGELOG: AUDIT AND REVISION HISTORY

## Version 3.0 (Confirmatory Submission Overhaul) - 2026-11-22

### 1. Statistical & Methodological Corrections
- **$\eta^2$ Regime Variance Decomposition (Phase 0 Audit Item a):**
  - Identified that previously reported values ($14.29\%$ on FD002 and $9.52\%$ on FD004) were mathematical artifacts of sensor counting ($3/21$ and $2/21$) due to constant sensors triggering epsilon divisions.
  - Corrected text to reflect that on all dynamic active sensors, per-regime z-scoring drives between-cluster variance to machine precision ($\eta^2 \approx 0$).
  - Deleted unsupported inferences that recurrent neural networks expend representational capacity tracking flight envelope shifts.
- **TOST Equivalence Testing (Phase 0 Audit Item b):**
  - Corrected overclaimed formal statistical equivalence under pilot $n=3$ seeds ($p=0.095$).
  - Clarified that while the empirical difference ($-0.23$ cycles) falls within the $\pm 1.5$ margin, formal equivalence requires confirmatory 10-seed testing.
  - Pre-registered `equivalence_protocol.md` defining exact Welch TOST and double-bootstrap equivalence testing procedures.
- **Engine Split Accounting (Phase 0 Audit Item c):**
  - Corrected reported engine counts from $18, 44, 89, 179$ to the actual codebase counts: $20, 50, 100, 199$ training engines alongside $50$ fixed validation engines ($249$ engines total on FD004).
- **Hyperparameter Tuning Transparency (Phase 0 Audit Item d):**
  - Disclosed that pilot tuning on FD004 utilized seeds $0$ and $1$, which overlapped pilot evaluation seeds. Pre-specified disjoint seeds ($100, 101$) for confirmatory tuning.
- **Cluster Count Analysis ($k=9$ vs. $k=6$, Phase 0 Audit Item e):**
  - Verified that $k=9$ ran independently. Documented that engine-level predictions differ by an average of $0.22$ cycles ($\max 1.62$ cycles), while fleet-level rounded RMSE matches $26.79$ cycles.
- **Protocol B Contextualization (Phase 0 Audit Item j):**
  - Nuanced the discussion of Protocol B ground-truth clipping: while the nominal $11$-cycle reduction is an artifact of truncating healthy engines, the relative normalization benefit is preserved and amplified ($-24.5\%$ on FD004).

### 2. Manuscript Prose, Integrity, and Framing
- **Title and Framing:** Reframed manuscript title to *"Operating-Condition Normalization in C-MAPSS Turbofan Prognostics: A Controlled Empirical Ablation, Label Scarcity, and Evaluation-Protocol Study"*, framing per-regime normalization strictly as prior work and highlighting empirical ablation and evaluation protocol audits.
- **Abstract & Tone (Phase 0 Audit Items g, i):**
  - Toned down promotional vocabulary ("exceptionally high-leverage", "dramatically", "scientifically invalid").
  - Explicitly reported modest gains on FD002 ($-6.6\%$, seed-level $p=0.126$, NASA CI spanning zero) while reporting robust significance on FD004 ($-13.9\%$).
  - Removed literal markdown syntax (`**99.19%**`) from `paper.tex`.
- **Limitations & Omitted Baselines (Phase 0 Audit Item h):**
  - Restored comprehensive disclosure of omitted baselines (one-hot regime indicators, raw settings inputs, min-max scaling, and alternate architectures).

### 3. Deliverables and Repositories
- Created `AUDIT.md` providing line-by-line verification of v2 findings.
- Created `equivalence_protocol.md` establishing pre-specified TOST procedures.
- Created `claims_evidence.csv` providing a complete traceability matrix for all numerical claims.
- Created `references_to_verify.md` documenting source literature extraction.
- Created `REVIEW_RESPONSE.md` providing point-by-point rebuttal to Reviewer 2 objections and re-scoring matrix.
- Updated `references.bib` with verified metadata and DOIs.
