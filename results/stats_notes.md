# Notes on Statistical Tests (1c)

1. **Per-engine quantity tested**: The engine-level paired t-test and Wilcoxon signed-rank test were performed on **per-engine squared error**, $(p_{\text{avg}} - y_{\text{true}})^2$.
2. **Seed averaging**: **Yes**, predictions across the 3 seeds were averaged per engine ($p_{\text{avg}} = \frac{1}{3}\sum_{s=0}^2 p_s$) before evaluating engine-level squared errors, paired hypothesis tests, and the paired bootstrap resamples.
3. **Bootstrap statistic**: In each bootstrap resample $b$, paired engine indices were resampled with replacement to compute the difference in root-mean-squared error, $\text{RMSE}_{\text{cond}}^{(b)} - \text{RMSE}_{\text{global}}^{(b)}$.
