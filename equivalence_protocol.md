# PRE-SPECIFIED EQUIVALENCE TESTING PROTOCOL (TOST)

**Protocol Registration Date:** 2026-11-22  
**Target Benchmark:** C-MAPSS FD004 Label Scarcity Study  
**Hypothesis:** An unsupervised condition-aware GRU trained on a scarce labeled engine fraction $x\%$ ($x \in \{10, 25, 50\}$) achieves predictive accuracy statistically equivalent to a globally normalized GRU trained on $100\%$ labeled engines.

---

## 1. Specification of Equivalence Margin ($\delta$)
- **Baseline Metric:** Mean test RMSE of GRU Global @ 100% labeled engines on FD004 ($31.13$ cycles).
- **Equivalence Bound:** $\delta = 1.5$ cycles ($\approx 4.82\%$ of baseline test RMSE).
- **Engineering Justification:** In turbofan fleet maintenance scheduling, an RUL deviation of $1.5$ flight cycles is well within the typical inspection interval (typically $25$--$50$ cycles) and reflects practically negligible predictive divergence.

---

## 2. Statistical Hypotheses (TOST Framework)
Let $\mu_{\text{cond}, x\%}$ be the true expected test RMSE for condition-aware GRU trained on $x\%$ labels, and $\mu_{\text{global}, 100\%}$ be the true expected test RMSE for global GRU trained on $100\%$ labels.
Define the mean difference:
$$\Delta = \mu_{\text{cond}, x\%} - \mu_{\text{global}, 100\%}$$

The null hypothesis of non-equivalence is composite:
$$H_{01}: \Delta \le -\delta \quad \text{or} \quad H_{02}: \Delta \ge +\delta$$
The alternative hypothesis of equivalence is:
$$H_1: -\delta < \Delta < +\delta$$

Equivalence at significance level $\alpha = 0.05$ is established if and only if both one-sided null hypotheses $H_{01}$ and $H_{02}$ are rejected at $\alpha = 0.05$. Operationally, this corresponds to checking whether the two-sided $(1 - 2\alpha) = 90\%$ confidence interval for $\Delta$ lies strictly within $(-\delta, +\delta) = (-1.5, +1.5)$.

---

## 3. Test Procedures

### A. Seed-Level Two-Sample TOST (Welch Form)
For confirmatory seeds ($n=10$ independent random train/val splits and model initializations):
1. Compute sample means $\bar{y}_1, \bar{y}_2$ and sample standard deviations $s_1, s_2$.
2. Compute Welch standard error:
   $$SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$
3. Compute Satterthwaite degrees of freedom $\nu$:
   $$\nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$$
4. Construct the $90\%$ confidence interval:
   $$[\bar{y}_1 - \bar{y}_2 - t_{0.95, \nu} \cdot SE, \quad \bar{y}_1 - \bar{y}_2 + t_{0.95, \nu} \cdot SE]$$
5. Reject non-equivalence if $CI_{90\%} \subset (-1.5, +1.5)$.

### B. Nested Subsets-and-Seeds Double Bootstrap
To account simultaneously for engine-subset sampling variance and neural training variance:
1. Resample $B = 10{,}000$ times with replacement over engine subsets and training seeds.
2. Form the empirical distribution of $\Delta^* = \text{RMSE}^*_{\text{cond}, x\%} - \text{RMSE}^*_{\text{global}, 100\%}$.
3. Obtain the $90\%$ BCa (or percentile) bootstrap interval $[q_{0.05}^*, q_{0.95}^*]$.
4. Declare robust equivalence if $[q_{0.05}^*, q_{0.95}^*] \subset (-1.5, +1.5)$.

---

## 4. Decision Rule and Null Reporting
- If the $90\%$ CI is strictly contained within $(-1.5, +1.5)$, report equivalence at margin $\delta = 1.5$.
- If the $90\%$ CI extends outside $(-1.5, +1.5)$, **do not claim equivalence**. Explicitly state that non-equivalence cannot be ruled out at the pre-specified margin $\delta=1.5$ under $\alpha=0.05$, and report the exact CI width and TOST $p$-value ($p = \max(p_1, p_2)$).
