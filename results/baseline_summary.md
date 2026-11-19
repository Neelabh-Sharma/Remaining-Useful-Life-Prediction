# NASA C-MAPSS Baseline Benchmark Summary

Evaluation on test engine final cycles (window=30, stride=5, target clipped at 125, RF n_estimators=100, max_depth=20, seeds 0, 1, 2).

| Subset | Model | Test RMSE | Test NASA Score |
| :--- | :--- | :--- | :--- |
| FD001 | Ridge | 18.24 | 537.3 |
| FD001 | Random Forest | 18.49 ± 0.13 | 878.4 ± 54.4 |
| FD002 | Ridge | 29.20 | 12905.2 |
| FD002 | Random Forest | 30.89 ± 0.05 | 14547.0 ± 743.8 |
| FD003 | Ridge | 18.77 | 651.5 |
| FD003 | Random Forest | 19.61 ± 0.23 | 1507.4 ± 56.7 |
| FD004 | Ridge | 31.51 | 8652.6 |
| FD004 | Random Forest | 32.04 ± 0.15 | 9008.6 ± 230.9 |

*Notes: Random Forest metrics are reported as mean ± std across 3 random seeds (0, 1, 2). Ridge regression is deterministic.*
